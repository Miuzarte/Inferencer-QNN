// htpbench 对比 QNN HTP 各性能档位 (power config) 下的单帧 Execute 延迟。
//
// 为什么需要它: 用真实 streamer + PC metrics 做 A/B 要跨进程重启, 一轮 >1 分钟,
// 而且手机的 CPU/热状态会漂移。htpbench 在单进程内轮询多档位
// (default -> burst -> ... -> default -> ...), 几十秒出结果, 把漂移抵消掉。
//
// 两个关键设计:
//
//  1. -interval 默认 33ms, 复现 30fps 源的真实占空比。背靠背 (-interval 0) 会让
//     DSP 一直醒着, 从而掩盖 sleepLatency / rpcPolling 的全部收益。
//  2. 投票是进程级的、不可撤销: 一旦下发过档位, "不下发" 的 default 就恢复不回来。
//     所以 default 只在"本进程还没投过任何票"的轮次里测量 (纯 default 跑法因此能拿到
//     完整时间序列用于看热衰减)。要换基线就重跑进程。
//
// 用法 (Termux):
//
//	./htpbench -ctx models/ctx_v73.bin -rounds 3 -iters 120 -interval 33ms \
//	  -profiles default,burst,burst_nosleep,sustained_high_performance,balanced
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	tjpeg "Inferencer/jpeg"
	"Inferencer/logger"
	"Inferencer/qnn"
	"Inferencer/qnn/perf"
	"Inferencer/yolo"
	"github.com/rs/zerolog"
)

var log = logger.New("HtpBench")

func main() {
	ctxPath := flag.String("ctx", "models/ctx_v73.bin", "QNN context binary")
	libDir := flag.String("lib", "/data/data/com.termux/files/usr/lib", "QNN libs dir")
	arch := flag.Int("arch", 73, "HTP arch")
	profilesFlag := flag.String("profiles", "default,burst",
		"逗号分隔的档位名, all = 全部 ("+perf.NamesString()+")")
	iters := flag.Int("iters", 120, "每轮每档测量帧数")
	rounds := flag.Int("rounds", 3, "轮询轮数 (抵消热/时钟漂移)")
	interval := flag.Duration("interval", 33*time.Millisecond,
		"每帧间隔 (复现 30fps 占空比); 0 = 背靠背, 会掩盖 DSP 休眠唤醒成本")
	warmup := flag.Int("warmup", 20, "换档后的预热帧数 (丢弃不计)")
	rpcLatency := flag.Int("rpc-control-latency", -1,
		"-1 = 用档位默认, 0 = 不下发该项, >0 = us")
	rpcPolling := flag.Int("rpc-polling-time", -1,
		"-1 = 用档位默认, 0 = 不下发该项, >0 = us")
	imagePath := flag.String("image", "",
		"可选 640x640 JPEG 作为输入; 缺省用确定性伪随机输入")
	jsonOut := flag.String("json", "", "可选, 结果落盘为 JSON")
	hashOutput := flag.Bool("hash-output", true,
		"每轮结束后打印输出张量的 FNV-1a 哈希 (用来证明档位不改变数值结果)")
	dryRun := flag.Bool("dry-run", false, "只打印解析后的档位参数, 不加载 QNN")
	verbose := flag.Bool("v", false, "verbose")
	flag.Parse()
	if *verbose {
		logger.SetGlobalLevel(zerolog.TraceLevel)
	}

	cfgs, err := resolveProfiles(*profilesFlag, *rpcLatency, *rpcPolling)
	if err != nil {
		log.Error().Err(err).Msg("bad -profiles")
		os.Exit(1)
	}
	if *dryRun {
		printProfiles(cfgs)
		return
	}

	ctxBin, err := os.ReadFile(*ctxPath)
	if err != nil {
		log.Error().Err(err).Msg("read ctx")
		os.Exit(1)
	}

	log.Info().Str("lib", *libDir).Int("arch", *arch).Msg("creating QNN session")
	sess, err := qnn.Create(*libDir, *arch, *verbose)
	if err != nil {
		log.Error().Err(err).Msg("qnn create failed")
		os.Exit(1)
	}
	defer sess.Close()

	perfReady := true
	if err := sess.PerfInit(); err != nil {
		perfReady = false
		if errors.Is(err, qnn.ErrPerfUnsupported) {
			log.Warn().Err(err).Msg("HTP perf infrastructure unavailable, 只有 default 档可测")
		} else {
			log.Error().Err(err).Msg("perf init failed, 只有 default 档可测")
		}
	}
	if pi, perr := sess.PlatformInfo(); perr != nil {
		log.Warn().Err(perr).Msg("platform info unavailable")
	} else {
		log.Info().Str("platform", pi.String()).Msg("device")
	}

	if err := sess.LoadBinary(ctxBin); err != nil {
		log.Error().Err(err).Msg("load binary failed")
		os.Exit(1)
	}
	_, inDims, _, outDims, inDtype, inScale, inOffset,
		outDtype, outScale, outOffset, err := sess.IOInfo()
	if err != nil {
		log.Error().Err(err).Msg("io info failed")
		os.Exit(1)
	}
	log.Info().
		Interface("in_dims", inDims).Int("in_dtype", inDtype).
		Float32("in_scale", inScale).Int32("in_offset", inOffset).
		Interface("out_dims", outDims).Int("out_dtype", outDtype).
		Float32("out_scale", outScale).Int32("out_offset", outOffset).
		Msg("graph IO")

	inBytes := numel(inDims) * dtypeBytes(inDtype)
	outBytes := numel(outDims) * dtypeBytes(outDtype)
	if inBytes <= 0 || outBytes <= 0 {
		log.Error().Interface("in_dims", inDims).Interface("out_dims", outDims).
			Msg("无法从 dims/dtype 推出缓冲大小")
		os.Exit(1)
	}
	in := make([]byte, inBytes)
	out := make([]byte, outBytes)
	if err := fillInput(in, *imagePath); err != nil {
		log.Warn().Err(err).Msg("输入准备失败, 改用确定性伪随机输入")
		fillDeterministic(in)
	}

	// 预热: 首次 Execute 会加载 Skel / 预热 DSP, 不计入
	if _, err := sess.Execute(in, out); err != nil {
		log.Error().Err(err).Msg("warmup execute failed")
		os.Exit(1)
	}

	// runFrames 按 interval 节流跑 n 帧; record 为 true 时返回每帧 Execute 的 ms。
	runFrames := func(n int, record bool) ([]float64, error) {
		var samples []float64
		if record {
			samples = make([]float64, 0, n)
		}
		next := time.Now()
		for i := 0; i < n; i++ {
			if *interval > 0 {
				if wait := time.Until(next); wait > 0 {
					time.Sleep(wait)
				}
				next = next.Add(*interval)
				if now := time.Now(); now.After(next) {
					next = now.Add(*interval) // 落后太多就重新对齐
				}
			}
			ms, err := sess.Execute(in, out)
			if err != nil {
				return nil, fmt.Errorf("execute #%d: %w", i, err)
			}
			if record {
				samples = append(samples, ms)
			}
		}
		return samples, nil
	}

	n := len(cfgs)
	perRound := make([][][]float64, n)
	all := make([][]float64, n)
	failed := make([]string, n)
	hashes := make([]map[uint64]int, n)
	for i := range hashes {
		hashes[i] = map[uint64]int{}
	}

	fmt.Printf("htpbench: iters=%d rounds=%d interval=%v warmup=%d\n",
		*iters, *rounds, *interval, *warmup)
	fmt.Printf("%-28s %5s %5s %8s %8s %8s %8s %8s %16s\n",
		"profile", "round", "n", "avg_ms", "p50", "p90", "max", "min", "out_fnv")

	// voted: 本进程是否已经下发过任何档位。投票不可撤销, 所以一旦投过票,
	// 后面的 default 就不再是"未投票基线", 只能跳过。
	// (-profiles default -rounds N 这种纯基线跑法因此可以拿到完整时间序列)
	voted := false
	for r := 0; r < *rounds; r++ {
		for i, cfg := range cfgs {
			if failed[i] != "" {
				continue
			}
			if perf.IsDefault(cfg.Name) && voted {
				continue
			}
			if cfg.HasAny() {
				if !perfReady {
					failed[i] = "perf infrastructure unavailable"
					fmt.Printf("%-28s %5s   -- %s\n", cfg.Name, "-", failed[i])
					continue
				}
				degraded, aerr := sess.PerfApply(cfg)
				if aerr != nil {
					failed[i] = aerr.Error()
					fmt.Printf("%-28s %5d   -- %v\n", cfg.Name, r+1, aerr)
					continue
				}
				voted = true
				if degraded {
					log.Warn().Str("profile", cfg.Name).
						Msg("部分配置项被拒, 已退化为只下发 DCVS_V3")
				}
			}
			if _, err := runFrames(*warmup, false); err != nil {
				failed[i] = err.Error()
				fmt.Printf("%-28s %5d   -- %v\n", cfg.Name, r+1, err)
				continue
			}
			samples, err := runFrames(*iters, true)
			if err != nil {
				failed[i] = err.Error()
				fmt.Printf("%-28s %5d   -- %v\n", cfg.Name, r+1, err)
				continue
			}
			perRound[i] = append(perRound[i], samples)
			all[i] = append(all[i], samples...)
			st := statsOf(samples)
			hashStr := "-"
			if *hashOutput {
				h := fnv64a(out)
				hashes[i][h]++
				hashStr = fmt.Sprintf("%016x", h)
			}
			fmt.Printf("%-28s %5d %5d %8.2f %8.2f %8.2f %8.2f %8.2f %16s\n",
				cfg.Name, r+1, len(samples), st.Avg, st.P50, st.P90, st.Max, st.Min, hashStr)
		}
	}

	printSummary(cfgs, all, failed, *iters, *rounds)
	if *hashOutput {
		printHashCheck(cfgs, hashes)
	}

	if *jsonOut != "" {
		if err := writeJSON(*jsonOut, cfgs, all, failed, *iters, *rounds, *interval); err != nil {
			log.Error().Err(err).Msg("写 json 失败")
		} else {
			log.Info().Str("path", *jsonOut).Msg("结果已落盘")
		}
	}

	// Termux 下正常 return 会让 QNN/厂商库的 atexit 清理触发 SIGABRT
	// (与 cmd/streamer 同一个坑: "signal arrived during cgo execution"),
	// 直接 os.Exit 跳过 defer 与 atexit。stdout 无缓冲, 不会丢日志。
	os.Exit(0)
}

// resolveProfiles 解析 -profiles 并应用 -rpc-* 覆盖 (-1 = 保留档位默认)。
func resolveProfiles(list string, rpcLatency, rpcPolling int) ([]perf.Config, error) {
	list = strings.TrimSpace(list)
	var names []string
	if list == "" || list == "all" {
		names = perf.Names()
	} else {
		for _, p := range strings.Split(list, ",") {
			if p = strings.TrimSpace(p); p != "" {
				names = append(names, p)
			}
		}
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("空档位列表, 可用: %s", perf.NamesString())
	}
	out := make([]perf.Config, 0, len(names))
	for _, name := range names {
		cfg, err := perf.Lookup(name)
		if err != nil {
			return nil, err
		}
		if rpcLatency != -1 {
			cfg.RpcControlLatency = rpcLatency
		}
		if rpcPolling != -1 {
			cfg.RpcPollingTime = rpcPolling
		}
		out = append(out, cfg)
	}
	return out, nil
}

func printProfiles(cfgs []perf.Config) {
	fmt.Printf("%-28s %10s %8s %9s %9s %8s %8s %7s %7s\n",
		"profile", "powerMode", "dcvsEn", "sleepLat", "sleepDis", "corner", "rpcLat", "rpcPol", "hasAny")
	for _, c := range cfgs {
		corner := "-"
		if c.HasDcvs() {
			corner = fmt.Sprintf("%#x/%#x/%#x", c.CoreMin, c.CoreTarget, c.CoreMax)
		}
		fmt.Printf("%-28s %10s %8d %9d %9d %8s %8d %7d %7t\n",
			c.Name, modeName(c.PowerMode), c.DcvsEnable, c.SleepLatency, c.SleepDisable,
			corner, c.RpcControlLatency, c.RpcPollingTime, c.HasAny())
	}
}

func modeName(m int) string {
	switch m {
	case 0:
		return "-"
	case perf.POWER_MODE_ADJUST_UP_DOWN:
		return "ADJ_UP_DOWN"
	case perf.POWER_MODE_ADJUST_ONLY_UP:
		return "ADJ_ONLY_UP"
	case perf.POWER_MODE_POWER_SAVER:
		return "PWR_SAVER"
	case perf.POWER_MODE_POWER_SAVER_AGGRESSIVE:
		return "PWR_SAVE_AGG"
	case perf.POWER_MODE_PERFORMANCE:
		return "PERFORMANCE"
	case perf.POWER_MODE_DUTY_CYCLE:
		return "DUTY_CYCLE"
	}
	return fmt.Sprintf("%#x", m)
}

// stats 是单组样本的汇总。
type stats struct {
	N              int     `json:"n"`
	Avg            float64 `json:"avg_ms"`
	P50            float64 `json:"p50_ms"`
	P90            float64 `json:"p90_ms"`
	Min            float64 `json:"min_ms"`
	Max            float64 `json:"max_ms"`
	Std            float64 `json:"std_ms"`
	DeltaVsDefault float64 `json:"delta_vs_default_pct,omitempty"`
}

func statsOf(samples []float64) stats {
	n := len(samples)
	if n == 0 {
		return stats{}
	}
	sorted := make([]float64, n)
	copy(sorted, samples)
	sort.Float64s(sorted)
	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	avg := sum / float64(n)
	var sq float64
	for _, v := range sorted {
		d := v - avg
		sq += d * d
	}
	return stats{
		N:   n,
		Avg: avg,
		P50: percentile(sorted, 0.50),
		P90: percentile(sorted, 0.90),
		Min: sorted[0],
		Max: sorted[n-1],
		Std: math.Sqrt(sq / float64(n)),
	}
}

func percentile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(q * float64(len(sorted)-1))
	if idx < 0 {
		idx = 0
	}
	if idx > len(sorted)-1 {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// benchRow 是汇总表的一行 (某个档位跨轮次的汇总, 或一条失败原因)。
type benchRow struct {
	name string
	st   stats
	err  string
}

// printHashCheck 汇总输出张量哈希: 同一档位的多轮哈希应当一致, 跨档位也应当一致 —
// 不一致说明档位影响了数值结果 (那就不只是"调频", 而是改变了硬件行为)。
func printHashCheck(cfgs []perf.Config, hashes []map[uint64]int) {
	fmt.Printf("\n---- 输出张量哈希 (FNV-1a 64) ----\n")
	global := map[uint64]int{}
	for i, c := range cfgs {
		if len(hashes[i]) == 0 {
			continue
		}
		strs := make([]string, 0, len(hashes[i]))
		for h, cnt := range hashes[i] {
			strs = append(strs, fmt.Sprintf("%016x x%d", h, cnt))
			global[h] += cnt
		}
		sort.Strings(strs)
		fmt.Printf("%-28s %s\n", c.Name, strings.Join(strs, ", "))
	}
	switch len(global) {
	case 0:
	case 1:
		fmt.Printf("所有档位输出完全一致: 投票只改频率/电压, 不改变数值结果\n")
	default:
		fmt.Printf("!! 出现 %d 个不同的输出哈希: 档位改变了数值结果, 需要单独确认\n", len(global))
	}
}

// fnv64a 是 FNV-1a 64 位哈希, 用来比对输出张量是否逐字节一致。
func fnv64a(b []byte) uint64 {
	const (
		offset = 14695981039346656037
		prime  = 1099511628211
	)
	h := uint64(offset)
	for _, v := range b {
		h ^= uint64(v)
		h *= prime
	}
	return h
}

func printSummary(cfgs []perf.Config, all [][]float64, failed []string,
	iters, rounds int) {
	rows := make([]benchRow, 0, len(cfgs))
	for i, c := range cfgs {
		if failed[i] != "" {
			rows = append(rows, benchRow{name: c.Name, err: failed[i]})
			continue
		}
		rows = append(rows, benchRow{name: c.Name, st: statsOf(all[i])})
	}

	// 基线: default 档优先, 否则第一行
	base := -1
	for i := range rows {
		if rows[i].err == "" && rows[i].st.N > 0 && perf.IsDefault(rows[i].name) {
			base = i
			break
		}
	}
	if base < 0 {
		for i := range rows {
			if rows[i].err == "" && rows[i].st.N > 0 {
				base = i
				break
			}
		}
	}
	baseAvg := 0.0
	if base >= 0 {
		baseAvg = rows[base].st.Avg
	}

	fmt.Printf("\n---- summary (iters=%d rounds=%d, avg over rounds) ----\n", iters, rounds)
	fmt.Printf("%-28s %6s %8s %8s %8s %8s %10s\n",
		"profile", "n", "avg_ms", "p50", "p90", "std", "vs "+rowName(rows, base))
	for i := range rows {
		if rows[i].err != "" {
			fmt.Printf("%-28s %6s %8s %8s %8s %8s %10s\n",
				rows[i].name, "-", "FAILED", "", "", "", rows[i].err)
			continue
		}
		delta := "-"
		if base >= 0 && i != base && baseAvg > 0 && rows[i].st.N > 0 {
			delta = fmt.Sprintf("%+.1f%%", (rows[i].st.Avg-baseAvg)/baseAvg*100)
			rows[i].st.DeltaVsDefault = (rows[i].st.Avg - baseAvg) / baseAvg * 100
		}
		fmt.Printf("%-28s %6d %8.2f %8.2f %8.2f %8.3f %10s\n",
			rows[i].name, rows[i].st.N, rows[i].st.Avg, rows[i].st.P50,
			rows[i].st.P90, rows[i].st.Std, delta)
	}
	if base >= 0 && !perf.IsDefault(rows[base].name) {
		fmt.Printf("\n注: 没有 default 档, 基线取 %s\n", rows[base].name)
	} else if base >= 0 {
		fmt.Printf("\n注: 投票是进程级的且不可撤销, default 只在投第一票之前有效\n")
	}
	fmt.Printf("注: 口径为 graphExecute 的 wall time, 与 PC metrics 的 stream_inference_ms 同义\n")
}

func rowName(rows []benchRow, base int) string {
	if base < 0 {
		return "base"
	}
	return rows[base].name
}

type jsonReport struct {
	Iters    int           `json:"iters"`
	Rounds   int           `json:"rounds"`
	Interval string        `json:"interval"`
	Profiles []jsonProfile `json:"profiles"`
}

type jsonProfile struct {
	Name    string  `json:"name"`
	Summary stats   `json:"summary"`
	Rounds  []stats `json:"rounds"`
	Error   string  `json:"error,omitempty"`
}

func writeJSON(path string, cfgs []perf.Config, all [][]float64, failed []string,
	iters, rounds int, interval time.Duration) error {
	rep := jsonReport{Iters: iters, Rounds: rounds, Interval: interval.String()}
	for i, c := range cfgs {
		jp := jsonProfile{Name: c.Name, Error: failed[i]}
		if jp.Error == "" {
			jp.Summary = statsOf(all[i])
		}
		for _, samples := range roundsOf(all[i], iters) {
			jp.Rounds = append(jp.Rounds, statsOf(samples))
		}
		rep.Profiles = append(rep.Profiles, jp)
	}
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// roundsOf 把扁平样本按 iters 切回每轮 (default 只有第 1 轮, 后面轮次缺).
func roundsOf(samples []float64, iters int) [][]float64 {
	if iters <= 0 {
		return nil
	}
	var out [][]float64
	for len(out)*iters < len(samples) {
		lo := len(out) * iters
		hi := lo + iters
		if hi > len(samples) {
			hi = len(samples)
		}
		out = append(out, samples[lo:hi])
	}
	return out
}

func numel(dims []uint32) int {
	if len(dims) == 0 {
		return 0
	}
	n := 1
	for _, d := range dims {
		if d == 0 {
			return 0
		}
		n *= int(d)
	}
	return n
}

// dtypeBytes 与 qnn/bridge.c 的 dtype_size 保持一致 (Qnn_DataType_t 的字面值见
// qnn-headers/QNN/QnnTypes.h:134-177)。
func dtypeBytes(dt int) int {
	switch dt {
	case 0x0216, 0x0116, 0x0016, 0x0316, 0x0416: // FLOAT_16, UINT_16, INT_16, SFIXED/UFIXED_16
		return 2
	case 0x0232, 0x0132, 0x0032, 0x0332, 0x0432: // FLOAT_32, UINT_32, INT_32, SFIXED/UFIXED_32
		return 4
	case 0x0064, 0x0164, 0x0264: // INT_64, UINT_64, FLOAT_64
		return 8
	}
	return 1
}

// fillInput 用 -image 准备输入 (须是 640x640 的 JPEG); 路径为空时返回错误由调用方兜底。
func fillInput(in []byte, imagePath string) error {
	if strings.TrimSpace(imagePath) == "" {
		return fmt.Errorf("未指定 -image")
	}
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return err
	}
	rgb, w, h, err := tjpeg.DecodeJPEGFlags(data, tjpeg.FlagsFast)
	if err != nil {
		return err
	}
	if w != yolo.ModelSize || h != yolo.ModelSize {
		return fmt.Errorf("图像是 %dx%d, 需要 %dx%d", w, h, yolo.ModelSize, yolo.ModelSize)
	}
	if len(in) != len(rgb)*2 {
		return fmt.Errorf("输入缓冲 %d 字节与 RGB %d 字节不匹配", len(in), len(rgb))
	}
	yolo.RGBToQuint16(rgb, in)
	return nil
}

// fillDeterministic 用 xorshift32 填满输入: 内容与延迟无关, 但要是非退化的分布。
func fillDeterministic(b []byte) {
	s := uint32(0x9E3779B9)
	for i := 0; i+1 < len(b); i += 2 {
		s ^= s << 13
		s ^= s >> 17
		s ^= s << 5
		v := uint16(s)
		b[i] = byte(v)
		b[i+1] = byte(v >> 8)
	}
}
