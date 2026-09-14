// streamer 在 Termux 中连接 GoCVStreamer WebSocket，做 QNN HTP 远程推理。
//
// 协议（与 flutterApp/androidApp 一致）：
//
//	收: [4B frame_id LE][JPEG]（640x640 流帧）
//	回: {"frame_id":N,"detections":[...],"inference_ms":M}
//
// 两种处理模式:
//
//	默认 (串行): read -> decode -> quant -> execute -> post -> write 一条线程走完。
//	  单帧延迟最低 (没有跨线程交接与缓冲弹跳), 但吞吐上限 = 1/(cpu+exec)。
//	-pipeline: 拆成 解码+量化 / NPU execute / 后处理+回包 三段 goroutine。
//	  同帧内 decode -> execute 有数据依赖, 单帧 wall time 不变, 但吞吐上限变成
//	  1/max(cpu,exec) (实测 30fps 下 cpu+~1.2ms、latency+~1.7ms 的代价换来
//	  60fps 源也能 58.6fps 不掉帧)。段间只留 1 帧, 挤掉最旧的: 过时的帧没有价值。
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	tjpeg "Inferencer/jpeg"
	"Inferencer/logger"
	"Inferencer/qnn"
	"Inferencer/qnn/perf"
	"Inferencer/yolo"
	"github.com/coder/websocket"
	"github.com/rs/zerolog"
)

var log = logger.New("Streamer")

type remoteDetection struct {
	X1        float64 `json:"x1"`
	Y1        float64 `json:"y1"`
	X2        float64 `json:"x2"`
	Y2        float64 `json:"y2"`
	Score     float64 `json:"score"`
	Class     int     `json:"class"`
	ClassName string  `json:"class_name"`
}

type remoteResult struct {
	FrameID     uint64            `json:"frame_id"`
	Detections  []remoteDetection `json:"detections"`
	InferenceMs float64           `json:"inference_ms"`

	// 分段耗时 (ms): CpuMs = 本帧各 CPU 段之和 (解码+量化+后处理),
	// 让 PC 端能把"手机自己的计算"从网络延迟里摘掉
	CpuMs    float64 `json:"cpu_ms,omitempty"`
	DecodeMs float64 `json:"decode_ms,omitempty"`
	QuantMs  float64 `json:"quant_ms,omitempty"`
	PostMs   float64 `json:"post_ms,omitempty"`
	// ReadMs = 等待下一帧的时间, 持续接近 0 说明手机侧已饱和 (在排队)
	ReadMs float64 `json:"read_ms,omitempty"`
	// QueueMs = 本帧在流水线队列里等的时间 (串行模式恒为 0, 不是网络)
	QueueMs float64 `json:"queue_ms,omitempty"`
	// FrameBytes = 收到的 JPEG 字节数 (A/B 时用它对照画面内容变化)
	FrameBytes int `json:"frame_bytes,omitempty"`
}

// bench 分段计时（-bench 开启后每 5s 输出各阶段 avg/max，单位 ms）。
const (
	stRead = iota
	stDecode
	stResize
	stToFloat
	stQuant
	stExecute
	stDequant
	stArgmax
	stNMS
	stWrite
	numStages
)

var stageNames = [numStages]string{
	"read", "decode", "resize", "tofloat", "quant",
	"execute", "dequant", "argmax", "nms", "write",
}

var benchSum [numStages]atomic.Uint64
var benchMax [numStages]atomic.Uint64
var benchFrames atomic.Uint64

func addBench(st int, d time.Duration) {
	us := uint64(d.Microseconds())
	benchSum[st].Add(us)
	for {
		m := benchMax[st].Load()
		if us <= m || benchMax[st].CompareAndSwap(m, us) {
			break
		}
	}
}

// frameJob 是一帧的工作单元: 缓冲复用, 各段把耗时写回同一个结构, 最后一起上报。
type frameJob struct {
	id        uint64
	bytes     int
	readDur   time.Duration
	decodeDur time.Duration
	quantDur  time.Duration
	postDur   time.Duration
	queueDur  time.Duration
	execMs    float64
	queuedAt  time.Time

	// 快路径 (640x640) 为 true; 否则用下面这组做归一化
	fast       bool
	scale      float32
	padX, padY int
	srcW, srcH int

	in  []byte
	out []byte
}

// reset 只清每帧变化的字段, 缓冲保留复用。
func (job *frameJob) reset() {
	job.id = 0
	job.bytes = 0
	job.readDur, job.decodeDur, job.quantDur, job.postDur, job.queueDur = 0, 0, 0, 0, 0
	job.execMs = 0
	job.fast = true
	job.scale, job.padX, job.padY, job.srcW, job.srcH = 0, 0, 0, 0, 0
}

// connHolder 在 reader (重连时替换连接) 与 writer (回包) 之间共享当前连接。
type connHolder struct {
	mu sync.RWMutex
	c  *websocket.Conn
}

func (h *connHolder) set(c *websocket.Conn) {
	h.mu.Lock()
	h.c = c
	h.mu.Unlock()
}

func (h *connHolder) get() *websocket.Conn {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.c
}

// offer 把 job 放进容量 1 的交接队列; 队列已满时丢弃最旧的那个并返回它 (调用方回收)。
// 实时场景下过时的帧没有价值, 丢旧保新可以避免延迟累积。
func offer(ch chan *frameJob, job *frameJob) (dropped *frameJob) {
	for {
		select {
		case ch <- job:
			return nil
		default:
		}
		select {
		case old := <-ch:
			return old
		default:
		}
	}
}

// pinnedMask/pinnedTID 记录 worker 线程成功绑定到的核集合 (0 表示未绑定)。
var (
	pinnedMask atomic.Int64
	pinnedTID  atomic.Int64
)

func setAffinityMask(mask uint64) error {
	_, _, e := syscall.Syscall(syscall.SYS_SCHED_SETAFFINITY, 0,
		uintptr(unsafe.Sizeof(mask)), uintptr(unsafe.Pointer(&mask)))
	if e != 0 {
		return e
	}
	return nil
}

// parseAffinity 解析 -affinity: "-1"/空=不绑, "7"=单核, "4-7"=范围, "4,5,6,7"=列表。
//
// 两代机器的快核都在末尾序号 (小米13: 3小+4大+1超大, 超大核=7;
// 小米17: 6大+2超大, 超大核=6,7), 所以用集合描述可以照搬。
func parseAffinity(s string) (uint64, []int, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == "-1" {
		return 0, nil, nil
	}
	var mask uint64
	var cpus []int
	add := func(n int) error {
		if n < 0 || n > 63 {
			return fmt.Errorf("cpu %d out of range", n)
		}
		if mask&(1<<uint(n)) == 0 {
			mask |= 1 << uint(n)
			cpus = append(cpus, n)
		}
		return nil
	}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if lo, hi, ok := strings.Cut(part, "-"); ok {
			l, err1 := strconv.Atoi(strings.TrimSpace(lo))
			h, err2 := strconv.Atoi(strings.TrimSpace(hi))
			if err1 != nil || err2 != nil || l > h {
				return 0, nil, fmt.Errorf("bad range %q", part)
			}
			for i := l; i <= h; i++ {
				if err := add(i); err != nil {
					return 0, nil, err
				}
			}
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return 0, nil, fmt.Errorf("bad cpu %q", part)
		}
		if err := add(n); err != nil {
			return 0, nil, err
		}
	}
	sort.Ints(cpus)
	return mask, cpus, nil
}

// pinThreadMask 把当前 goroutine 固定到 mask 里的核。
//
// Android 调度器只有在进程产生持续负载后才会把大核加入可调度集合，
// 因此设置失败时忙转一段时间再重试；绑定期间该 goroutine 会被 LockOSThread 固定住。
func pinThreadMask(mask uint64, timeout time.Duration) error {
	if mask == 0 {
		return nil
	}
	runtime.LockOSThread()
	deadline := time.Now().Add(timeout)
	lastErr := errors.New("timeout")
	for time.Now().Before(deadline) {
		if err := setAffinityMask(mask); err == nil {
			pinnedMask.Store(int64(mask))
			tid, _, _ := syscall.Syscall(syscall.SYS_GETTID, 0, 0, 0)
			pinnedTID.Store(int64(tid))
			log.Info().Uint64("mask", mask).Msg("pinned worker thread")
			return nil
		} else {
			lastErr = err
		}
		// 忙转 250ms：给调度器制造持续负载，促使它放开大核
		spinEnd := time.Now().Add(250 * time.Millisecond)
		for time.Now().Before(spinEnd) {
		}
	}
	return fmt.Errorf("pin mask %#x: %w", mask, lastErr)
}

// readThreadStat 读线程的累计 CPU 时间（tick）与当前所在核。
// Android 上普通用户读不了 /proc/stat，但可以读自己线程的 stat。
func readThreadStat(tid int64) (cpuTicks, procNo int64) {
	b, err := os.ReadFile(fmt.Sprintf("/proc/self/task/%d/stat", tid))
	if err != nil {
		return 0, -1
	}
	f := strings.Fields(string(b))
	if len(f) < 39 {
		return 0, -1
	}
	ut, _ := strconv.ParseInt(f[13], 10, 64)
	st, _ := strconv.ParseInt(f[14], 10, 64)
	p, _ := strconv.ParseInt(f[38], 10, 64)
	return ut + st, p
}

func main() {
	wsURL := flag.String("url", "ws://192.168.1.100:9090/stream", "GoCVStreamer WS URL")
	ctxPath := flag.String("ctx", "models/ctx_v73.bin", "QNN context binary")
	libDir := flag.String("lib", "/data/data/com.termux/files/usr/lib", "QNN libs dir")
	arch := flag.Int("arch", 73, "HTP arch")
	conf := flag.Float64("conf", 0.45, "confidence threshold")
	classID := flag.Int("class", 0, "只回传指定类别（0=person，-1=全部）")
	verbose := flag.Bool("v", false, "verbose")
	bench := flag.Bool("bench", false, "per-stage timing stats")
	affinity := flag.String("affinity", "-1",
		"固定 worker 线程的 CPU 集合: -1=不绑, 7=单核, 4-7=范围, 4,5,6,7=列表 (小米13 快核=3-7, 小米17 快核=2-7)")
	pipeline := flag.Bool("pipeline", false, "三段流水线 (吞吐上限 = 1/max(CPU前处理, NPU execute), 代价是 30fps 下 cpu+1.2ms/latency+1.7ms)")
	compress := flag.Bool("compress", false, "enable permessage-deflate (JPEG 不可压, 默认关)")
	fastjpeg := flag.Bool("fastjpeg", true, "turbojpeg FASTDCT|FASTUPSAMPLE 快解码 (精度略降)")
	argmaxAll := flag.Bool("argmax-all", false, "后处理对全部 80 类求 argmax (默认只算 -class 指定的那一类, 更快)")
	htpPerf := flag.String("htp-perf", "default",
		"HTP 性能档位: "+perf.NamesString()+" (default = 不下发, 保持 QNN 默认行为)")
	rpcLatency := flag.Int("rpc-control-latency", -1,
		"覆盖 RPC control latency (us): -1 = 用档位默认, 0 = 不下发该项")
	rpcPolling := flag.Int("rpc-polling-time", -1,
		"覆盖 RPC polling time (us, 上限 9999): -1 = 用档位默认, 0 = 不下发该项")
	htpInfo := flag.Bool("htp-info", false, "启动时打印 HTP 性能基础设施与平台信息")
	flag.Parse()
	if *verbose {
		logger.SetGlobalLevel(zerolog.TraceLevel)
	}

	// 档位名先解析: 拼错就立刻退出, 不要静默降级成"没有投票"
	perfCfg, err := perf.Lookup(*htpPerf)
	if err != nil {
		log.Error().Err(err).Msg("bad -htp-perf")
		os.Exit(1)
	}
	if *rpcLatency != -1 {
		perfCfg.RpcControlLatency = *rpcLatency
	}
	if *rpcPolling != -1 {
		perfCfg.RpcPollingTime = *rpcPolling
	}
	if *rpcPolling > 9999 {
		log.Error().Int("rpc_polling_time", *rpcPolling).
			Msg("rpc-polling-time 上限是 9999us")
		os.Exit(1)
	}

	affinityMask, affinityCPUs, err := parseAffinity(*affinity)
	if err != nil {
		log.Error().Err(err).Str("affinity", *affinity).Msg("bad -affinity")
		os.Exit(1)
	}
	if affinityMask != 0 {
		log.Info().Interface("cpus", affinityCPUs).Uint64("mask", affinityMask).Msg("affinity configured")
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

	// HTP 性能档位必须在 LoadBinary 之前下发 (与 QIDK 的
	// deviceCreate -> perf -> contextCreate 顺序一致)。
	// 不支持时只告警并降级, 不能让设备差异变成启动失败。
	perfReady := false
	if perr := sess.PerfInit(); perr != nil {
		log.Warn().Err(perr).Str("htp_perf", perfCfg.Name).
			Msg("HTP perf infrastructure unavailable, falling back to no vote")
	} else {
		perfReady = true
		infraType, powerConfigID, ok := sess.PerfStatus()
		log.Info().
			Int("infra_type", infraType).
			Uint32("power_config_id", powerConfigID).
			Bool("power_config_ok", ok).
			Str("htp_perf", perfCfg.Name).
			Msg("HTP perf infrastructure ready")
	}
	if *htpInfo || perfCfg.HasAny() {
		if pi, perr := sess.PlatformInfo(); perr != nil {
			log.Warn().Err(perr).Msg("platform info unavailable")
		} else {
			log.Info().Str("platform", pi.String()).Msg("device")
		}
	}
	if perfCfg.HasAny() {
		if !perfReady {
			log.Warn().Str("htp_perf", perfCfg.Name).
				Msg("perf infrastructure unavailable, profile not applied")
		} else if degraded, aerr := sess.PerfApply(perfCfg); aerr != nil {
			log.Warn().Err(aerr).Str("htp_perf", perfCfg.Name).Msg("perf apply failed")
		} else if degraded {
			log.Warn().Str("htp_perf", perfCfg.Name).
				Msg("perf applied with degradation (only DCVS_V3 accepted)")
		} else {
			log.Info().Str("htp_perf", perfCfg.Name).Msg("perf applied")
		}
	}

	if err := sess.LoadBinary(ctxBin); err != nil {
		log.Error().Err(err).Msg("load binary failed")
		os.Exit(1)
	}
	_, inDims, _, outDims, inDtype, inScale, inOffset, outDtype, outScale, outOffset, err :=
		sess.IOInfo()
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

	// 实时解码标志: FASTDCT + FASTUPSAMPLE, 精度略降但更快
	jpegFlags := 0
	if *fastjpeg {
		jpegFlags = tjpeg.FlagsFast
	}

	// 复用的帧缓冲 (每组 2.4MB in + 1.4MB out); 三种模式共用
	const pipelineDepth = 3
	idle := make(chan *frameJob, pipelineDepth)
	for i := 0; i < pipelineDepth; i++ {
		idle <- &frameJob{
			in:  make([]byte, 640*640*3*2),
			out: make([]byte, 84*8400*2),
		}
	}
	toExec := make(chan *frameJob, 1)
	toPost := make(chan *frameJob, 1)
	conn := &connHolder{}
	var reconnectGen atomic.Uint64
	var dropped atomic.Uint64

	// Termux 下 QNN 库在进程退出阶段（内部线程清理）会触发 SIGABRT，
	// 即使 qnn_destroy 已跳过 QnnContext_free 也一样。因此收到
	// ctrl+c/SIGTERM 时直接 os.Exit，跳过 Go 正常退出路径与 QNN 清理。
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Info().Str("signal", sig.String()).Msg("received signal, exiting without QNN cleanup")
		os.Exit(0)
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 每帧的 CPU 前处理: 解码 + 量化, 结果写进 job.in
	preprocess := func(job *frameJob, jpegData []byte) error {
		td := time.Now()
		rgb, imgW, imgH, jerr := tjpeg.DecodeJPEGFlags(jpegData, jpegFlags)
		job.decodeDur = time.Since(td)
		addBench(stDecode, job.decodeDur)

		if jerr == nil && imgW == yolo.ModelSize && imgH == yolo.ModelSize {
			// 快路径：640x640 JPEG -> RGB -> quint16 LUT，跳过 letterbox/tofloat/dequant。
			tq := time.Now()
			yolo.RGBToQuint16(rgb, job.in)
			job.quantDur = time.Since(tq)
			addBench(stQuant, job.quantDur)
			job.fast = true
			return nil
		}

		// 回退路径：任意尺寸 JPEG，走 stdlib 解码 + letterbox + float 量化。
		if jerr != nil {
			log.Warn().Err(jerr).Msg("jpeg decode failed, fallback")
		}
		timg := time.Now()
		img, srcW, srcH, derr := yolo.Decode(jpegData)
		job.decodeDur += time.Since(timg)
		addBench(stDecode, time.Since(timg))
		if derr != nil {
			log.Error().Err(derr).Msg("decode failed")
			return derr
		}
		t2 := time.Now()
		rgba, scale, padX, padY := yolo.Letterbox(img, srcW, srcH)
		addBench(stResize, time.Since(t2))
		t3 := time.Now()
		input := yolo.ToFloat(rgba)
		addBench(stToFloat, time.Since(t3))
		t4 := time.Now()
		quantizeInput(input, job.in, inScale, inOffset)
		addBench(stQuant, time.Since(t4))
		job.quantDur = time.Since(t2)
		job.fast = false
		job.scale, job.padX, job.padY = scale, padX, padY
		job.srcW, job.srcH = srcW, srcH
		return nil
	}

	// 每帧的后处理: argmax + NMS + 归一化 + 类别过滤
	postprocess := func(job *frameJob) []remoteDetection {
		var boxes []yolo.DetBox
		if job.fast {
			t7 := time.Now()
			var cands []yolo.Candidate
			if *argmaxAll {
				cands = yolo.ArgmaxU16(job.out, float32(*conf), outScale)
			} else {
				// 只算 -class 指定的那一类: 少扫 79/80 的输出, person-only 时省 ~2ms
				cands = yolo.ArgmaxU16Class(job.out, float32(*conf), outScale, *classID)
			}
			addBench(stArgmax, time.Since(t7))
			t8 := time.Now()
			picked := yolo.NMS(cands, 0.7)
			addBench(stNMS, time.Since(t8))
			boxes = yolo.Normalize(picked, 1, 0, 0, yolo.ModelSize, yolo.ModelSize)
		} else {
			t6 := time.Now()
			out := dequantizeOutput(job.out, outScale, outOffset)
			addBench(stDequant, time.Since(t6))
			t7 := time.Now()
			cands := yolo.Argmax(out, float32(*conf))
			addBench(stArgmax, time.Since(t7))
			t8 := time.Now()
			picked := yolo.NMS(cands, 0.7)
			addBench(stNMS, time.Since(t8))
			boxes = yolo.Normalize(picked, job.scale, job.padX, job.padY, job.srcW, job.srcH)
		}

		dets := make([]remoteDetection, 0, len(boxes))
		for _, b := range boxes {
			if *classID >= 0 && b.Class != *classID {
				continue
			}
			dets = append(dets, remoteDetection{
				X1: float64(b.X1), Y1: float64(b.Y1),
				X2: float64(b.X2), Y2: float64(b.Y2),
				Score: float64(b.Score), Class: b.Class, ClassName: b.ClassName,
			})
		}
		return dets
	}

	var frames atomic.Uint64
	var msSum atomic.Uint64
	var lastMS atomic.Uint64

	msOf := func(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

	// report 组包 + 回包 (写失败只记日志, 由 reader 段负责重连)
	report := func(c *websocket.Conn, job *frameJob, dets []remoteDetection) {
		res := remoteResult{
			FrameID:     job.id,
			Detections:  dets,
			InferenceMs: job.execMs,
			CpuMs:       msOf(job.decodeDur + job.quantDur + job.postDur),
			DecodeMs:    msOf(job.decodeDur),
			QuantMs:     msOf(job.quantDur),
			PostMs:      msOf(job.postDur),
			ReadMs:      msOf(job.readDur),
			QueueMs:     msOf(job.queueDur),
			FrameBytes:  job.bytes,
		}
		payload, _ := json.Marshal(res)
		tw := time.Now()
		if c != nil {
			if err := c.Write(ctx, websocket.MessageText, payload); err != nil {
				log.Error().Err(err).Msg("write failed")
				c.Close(websocket.StatusAbnormalClosure, "write error")
			}
		}
		addBench(stWrite, time.Since(tw))
		frames.Add(1)
		benchFrames.Add(1)
		msSum.Add(uint64(job.execMs * 1000))
		lastMS.Store(uint64(job.execMs * 1000))
	}

	// dialLoop 负责重连; 每次连上后调用 onConn (重新确认绑核)
	dialLoop := func(onConn func()) {
		for {
			log.Info().Str("url", *wsURL).Msg("connecting")
			compMode := websocket.CompressionDisabled
			if *compress {
				compMode = websocket.CompressionContextTakeover
			}
			c, _, derr := websocket.Dial(ctx, *wsURL, &websocket.DialOptions{
				CompressionMode: compMode,
			})
			if derr != nil {
				log.Warn().Err(derr).Msg("dial failed, retry in 3s")
				select {
				case <-ctx.Done():
					return
				case <-time.After(3 * time.Second):
				}
				continue
			}
			log.Info().Msg("connected")
			c.SetReadLimit(16 << 20) // 640x640 JPEG 帧可达百 KB 级，默认 32KB 会直接断连
			conn.set(c)
			reconnectGen.Add(1)
			if onConn != nil {
				onConn()
			}
			return
		}
	}

	// 读一帧并占一个 job; 返回 nil 表示这帧该丢 (池空或非二进制帧)
	readFrame := func(c *websocket.Conn) *frameJob {
		for {
			t0 := time.Now()
			mt, data, rerr := c.Read(ctx)
			readDur := time.Since(t0)
			addBench(stRead, readDur)
			if rerr != nil {
				log.Error().Err(rerr).Msg("read failed, reconnect")
				c.Close(websocket.StatusNormalClosure, "reconnect")
				conn.set(nil)
				return nil
			}
			if mt != websocket.MessageBinary || len(data) < 5 {
				continue
			}
			// 池空 = 自己还在忙: 丢掉这一帧但继续读, 保持 TCP 排空 (不读会积压)
			var job *frameJob
			select {
			case job = <-idle:
			default:
				dropped.Add(1)
				continue
			}
			job.reset()
			job.id = uint64(binary.LittleEndian.Uint32(data[:4]))
			job.bytes = len(data) - 4
			job.readDur = readDur
			if err := preprocess(job, data[4:]); err != nil {
				idle <- job
				continue
			}
			return job
		}
	}

	// 串行: 一条线程走完全程, 单帧延迟最低
	serialStage := func() {
		if affinityMask != 0 {
			if err := pinThreadMask(affinityMask, 20*time.Second); err != nil {
				log.Warn().Err(err).Msg("pin failed, continue without pinning")
			}
		}
		var lastGen uint64
		for {
			reGen := func() {
				if gen := reconnectGen.Load(); gen != lastGen {
					lastGen = gen
					if affinityMask != 0 {
						// 断线期间调度器可能收窄了可调度核集, 重连后重新确认绑定
						if err := pinThreadMask(affinityMask, 8*time.Second); err != nil {
							log.Warn().Err(err).Msg("re-pin failed, continue")
						}
					}
				}
			}
			dialLoop(reGen)
			c := conn.get()
			if c == nil {
				select {
				case <-ctx.Done():
					return
				case <-time.After(time.Second):
				}
				continue
			}
			for {
				job := readFrame(c)
				if job == nil {
					if ctx.Err() != nil {
						return
					}
					break // 读失败: 回到外层重连
				}
				te := time.Now()
				execMs, eerr := sess.Execute(job.in, job.out)
				addBench(stExecute, time.Since(te))
				if eerr != nil {
					log.Error().Err(eerr).Msg("execute failed")
					idle <- job
					continue
				}
				job.execMs = execMs
				tp := time.Now()
				dets := postprocess(job)
				job.postDur = time.Since(tp)
				report(c, job, dets)
				idle <- job
			}
		}
	}

	// 流水线第一段: 收帧 + 解码 + 量化 (纯 CPU)
	readStage := func() {
		for {
			dialLoop(nil)
			c := conn.get()
			if c == nil {
				select {
				case <-ctx.Done():
					return
				case <-time.After(time.Second):
				}
				continue
			}
			for {
				job := readFrame(c)
				if job == nil {
					if ctx.Err() != nil {
						return
					}
					break
				}
				job.queuedAt = time.Now()
				if old := offer(toExec, job); old != nil {
					idle <- old // 回收被挤掉的那帧
				}
			}
		}
	}

	// 流水线第二段: NPU execute (worker 线程绑核在这一段)
	execStage := func() {
		if affinityMask != 0 {
			if err := pinThreadMask(affinityMask, 20*time.Second); err != nil {
				log.Warn().Err(err).Msg("pin failed, continue without pinning")
			}
		}
		var lastGen uint64
		for {
			select {
			case <-ctx.Done():
				return
			case job := <-toExec:
				if gen := reconnectGen.Load(); gen != lastGen {
					lastGen = gen
					if affinityMask != 0 {
						if err := pinThreadMask(affinityMask, 8*time.Second); err != nil {
							log.Warn().Err(err).Msg("re-pin failed, continue")
						}
					}
				}
				job.queueDur += time.Since(job.queuedAt)
				te := time.Now()
				execMs, eerr := sess.Execute(job.in, job.out)
				addBench(stExecute, time.Since(te))
				if eerr != nil {
					log.Error().Err(eerr).Msg("execute failed")
					idle <- job
					continue
				}
				job.execMs = execMs
				job.queuedAt = time.Now()
				if old := offer(toPost, job); old != nil {
					idle <- old
				}
			}
		}
	}

	// 流水线第三段: 后处理 + 回包
	postStage := func() {
		if affinityMask != 0 {
			if err := pinThreadMask(affinityMask, 20*time.Second); err != nil {
				log.Warn().Err(err).Msg("pin post failed, continue without pinning")
			}
		}
		for {
			select {
			case <-ctx.Done():
				return
			case job := <-toPost:
				job.queueDur += time.Since(job.queuedAt)
				tp := time.Now()
				dets := postprocess(job)
				job.postDur = time.Since(tp)
				report(conn.get(), job, dets)
				idle <- job
			}
		}
	}

	// 统计协程
	var lastAffTicks int64
	go func() {
		t := time.NewTicker(5 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				f := frames.Swap(0)
				ms := msSum.Swap(0)
				avg := float64(0)
				if f > 0 {
					avg = float64(ms) / float64(f) / 1000
				}
				log.Info().
					Float64("fps", float64(f)/5).
					Uint64("frames", f).
					Uint64("dropped", dropped.Swap(0)).
					Bool("pipeline", *pipeline).
					Str("htp_perf", perfCfg.Name).
					Float64("avg_infer_ms", avg).
					Float64("last_infer_ms", float64(lastMS.Load())/1000).
					Msg("stats")
				if *bench {
					bf := benchFrames.Swap(0)
					ev := log.Info().Uint64("frames", bf)
					for i := 0; i < numStages; i++ {
						s := benchSum[i].Swap(0)
						m := benchMax[i].Swap(0)
						a := float64(0)
						if bf > 0 {
							a = float64(s) / float64(bf) / 1000
						}
						ev = ev.
							Float64(stageNames[i]+"_avg", a).
							Float64(stageNames[i]+"_max", float64(m)/1000)
					}
					ev.Msg("bench")
				}
				if affinityMask != 0 {
					if tid := pinnedTID.Load(); tid > 0 {
						ticks, procNo := readThreadStat(tid)
						if lastAffTicks != 0 && ticks >= lastAffTicks {
							cpuSec := float64(ticks-lastAffTicks) / 100 // USER_HZ=100
							pct := cpuSec / 5 * 100
							if pct > 100 {
								pct = 100
							}
							freq := ""
							if procNo >= 0 {
								freqB, _ := os.ReadFile(fmt.Sprintf(
									"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", procNo))
								freq = strings.TrimSpace(string(freqB))
							}
							log.Info().
								Int64("mask", pinnedMask.Load()).
								Int64("proc", procNo).
								Float64("busy_pct", pct).
								Str("freq", freq).
								Msg("affinity")
						}
						lastAffTicks = ticks
					}
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	if *pipeline {
		go readStage()
		go execStage()
		go postStage()
	} else {
		go serialStage()
	}
	<-ctx.Done()
}

// quantizeInput float(0..1) -> quint16
func quantizeInput(v []float32, out []byte, scale float32, offset int32) {
	for i, f := range v {
		q := int32(f/scale+0.5) - offset
		if q < 0 {
			q = 0
		} else if q > 65535 {
			q = 65535
		}
		u := uint16(q)
		out[i*2] = byte(u)
		out[i*2+1] = byte(u >> 8)
	}
}

// dequantizeOutput quint16 -> float32
func dequantizeOutput(b []byte, scale float32, offset int32) []float32 {
	v := make([]float32, len(b)/2)
	for i := range v {
		u := uint16(b[i*2]) | uint16(b[i*2+1])<<8
		v[i] = (float32(int32(u)) + float32(offset)) * scale
	}
	return v
}
