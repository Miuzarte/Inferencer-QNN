// streamer 在 Termux 中连接 GoCVStreamer WebSocket，做 QNN HTP 远程推理。
//
// 协议（与 flutterApp/androidApp 一致）：
//
//	收: [4B frame_id LE][JPEG]（640x640 流帧）
//	回: {"frame_id":N,"detections":[...],"inference_ms":M}
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
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	tjpeg "Inferencer/jpeg"
	"Inferencer/qnn"
	"Inferencer/yolo"
	"github.com/coder/websocket"
)

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

// pinnedCPU 记录推理线程成功绑定到的 CPU（-1 表示未绑定）。
var (
	pinnedCPU atomic.Int64
	pinnedTID atomic.Int64
)

func setAffinityMask(mask uint64) error {
	_, _, e := syscall.Syscall(syscall.SYS_SCHED_SETAFFINITY, 0,
		uintptr(unsafe.Sizeof(mask)), uintptr(unsafe.Pointer(&mask)))
	if e != 0 {
		return e
	}
	return nil
}

// pinThread 把当前 goroutine 固定到指定 CPU。
//
// Android 调度器只有在进程产生持续负载后才会把大核加入可调度集合，
// 因此设置失败时忙转一段时间再重试；绑定期间该 goroutine 会被
// LockOSThread 固定住（推理主循环就在这个 goroutine 上）。
func pinThread(cpu int, timeout time.Duration) error {
	runtime.LockOSThread()
	deadline := time.Now().Add(timeout)
	lastErr := errors.New("timeout")
	for time.Now().Before(deadline) {
		if err := setAffinityMask(uint64(1) << uint(cpu)); err == nil {
			pinnedCPU.Store(int64(cpu))
			tid, _, _ := syscall.Syscall(syscall.SYS_GETTID, 0, 0, 0)
			pinnedTID.Store(int64(tid))
			fmt.Printf("pinned inference thread to cpu%d\n", cpu)
			return nil
		} else {
			lastErr = err
		}
		// 忙转 250ms：给调度器制造持续负载，促使它放开大核
		spinEnd := time.Now().Add(250 * time.Millisecond)
		for time.Now().Before(spinEnd) {
		}
	}
	return fmt.Errorf("pin cpu%d: %w", cpu, lastErr)
}

// readThreadStat 读推理线程的累计 CPU 时间（tick）与当前所在核。
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
	affinity := flag.Int("affinity", -1, "pin inference thread to CPU N (e.g. 7 = prime core; -1 = disabled)")
	flag.Parse()

	ctxBin, err := os.ReadFile(*ctxPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read ctx:", err)
		os.Exit(1)
	}

	fmt.Printf("creating QNN session (lib=%s arch=%d)...\n", *libDir, *arch)
	sess, err := qnn.Create(*libDir, *arch, *verbose)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer sess.Close()
	if err := sess.LoadBinary(ctxBin); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	_, inDims, _, outDims, inDtype, inScale, inOffset, outDtype, outScale, outOffset, err :=
		sess.IOInfo()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("graph IO: %v (dtype=%d scale=%f off=%d) -> %v (dtype=%d scale=%f off=%d)\n",
		inDims, inDtype, inScale, inOffset, outDims, outDtype, outScale, outOffset)

	inBytes := make([]byte, 640*640*3*2)
	outBytes := make([]byte, 84*8400*2)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if *affinity >= 0 {
		if err := pinThread(*affinity, 20*time.Second); err != nil {
			fmt.Fprintf(os.Stderr, "warning: %v (continue without pinning)\n", err)
		}
	}

	var frames atomic.Uint64
	var msSum atomic.Uint64
	var lastMS atomic.Uint64
	var lastAffTicks int64

	// 统计协程
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
				fmt.Printf("[stats] fps=%.1f frames=%d avg_infer_ms=%.2f last_infer_ms=%.2f\n",
					float64(f)/5, f, avg, float64(lastMS.Load())/1000)
				if *bench {
					bf := benchFrames.Swap(0)
					fmt.Printf("[bench] frames=%d", bf)
					for i := 0; i < numStages; i++ {
						s := benchSum[i].Swap(0)
						m := benchMax[i].Swap(0)
						avg := float64(0)
						if bf > 0 {
							avg = float64(s) / float64(bf) / 1000
						}
						fmt.Printf(" %s=%.2f/%.2f", stageNames[i], avg, float64(m)/1000)
					}
					fmt.Println()
				}
				if *affinity >= 0 {
					tid := pinnedTID.Load()
					if tid > 0 {
						ticks, procNo := readThreadStat(tid)
						if lastAffTicks != 0 && ticks >= lastAffTicks {
							cpuSec := float64(ticks-lastAffTicks) / 100 // USER_HZ=100
							pct := cpuSec / 5 * 100
							if pct > 100 {
								pct = 100
							}
							freqB, _ := os.ReadFile(fmt.Sprintf(
								"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", *affinity))
							fmt.Printf("[aff] cpu%d pinned=%d proc=%d busy=%.1f%% freq=%s\n",
								*affinity, pinnedCPU.Load(), procNo, pct,
								strings.TrimSpace(string(freqB)))
						}
						lastAffTicks = ticks
					}
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	// 断线重连循环
	for {
		fmt.Printf("connecting %s ...\n", *wsURL)
		c, _, err := websocket.Dial(ctx, *wsURL, &websocket.DialOptions{
			CompressionMode: websocket.CompressionContextTakeover,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "dial: %v (retry in 3s)\n", err)
			select {
			case <-ctx.Done():
				return
			case <-time.After(3 * time.Second):
			}
			continue
		}
		fmt.Println("connected")
		c.SetReadLimit(16 << 20) // 640x640 JPEG 帧可达百 KB 级，默认 32KB 会直接断连

		if *affinity >= 0 {
			// 断线重连期间调度器可能收窄了大核，连接成功后重新确认绑定
			if err := pinThread(*affinity, 8*time.Second); err != nil {
				fmt.Fprintf(os.Stderr, "warning: %v (continue without pinning)\n", err)
			}
		}

		for {
			t0 := time.Now()
			mt, data, err := c.Read(ctx)
			addBench(stRead, time.Since(t0))
			if err != nil {
				fmt.Fprintf(os.Stderr, "read: %v (reconnect)\n", err)
				c.Close(websocket.StatusNormalClosure, "reconnect")
				break
			}
			if mt != websocket.MessageBinary || len(data) < 5 {
				continue
			}
			frameID := uint64(binary.LittleEndian.Uint32(data[:4]))
			jpegData := data[4:]

			var ms float64
			var boxes []yolo.DetBox

			t1 := time.Now()
			rgb, imgW, imgH, jerr := tjpeg.DecodeJPEG(jpegData)
			addBench(stDecode, time.Since(t1))

			if jerr == nil && imgW == yolo.ModelSize && imgH == yolo.ModelSize {
				// 快路径：640x640 JPEG -> RGB -> quint16 LUT，跳过 letterbox/tofloat/dequant。
				t4 := time.Now()
				yolo.RGBToQuint16(rgb, inBytes)
				addBench(stQuant, time.Since(t4))

				t5 := time.Now()
				ms, err = sess.Execute(inBytes, outBytes)
				addBench(stExecute, time.Since(t5))
				if err != nil {
					fmt.Fprintf(os.Stderr, "execute: %v\n", err)
					continue
				}
				t7 := time.Now()
				cands := yolo.ArgmaxU16(outBytes, float32(*conf), outScale)
				addBench(stArgmax, time.Since(t7))
				t8 := time.Now()
				picked := yolo.NMS(cands, 0.7)
				addBench(stNMS, time.Since(t8))
				boxes = yolo.Normalize(picked, 1, 0, 0, yolo.ModelSize, yolo.ModelSize)
			} else {
				// 回退路径：任意尺寸 JPEG，走 stdlib 解码 + letterbox + float 量化。
				if jerr != nil {
					fmt.Fprintf(os.Stderr, "jpeg decode: %v (fallback)\n", jerr)
				}
				img, srcW, srcH, derr := yolo.Decode(jpegData)
				if derr != nil {
					fmt.Fprintf(os.Stderr, "decode: %v\n", derr)
					continue
				}
				t2 := time.Now()
				rgba, scale, padX, padY := yolo.Letterbox(img, srcW, srcH)
				addBench(stResize, time.Since(t2))
				t3 := time.Now()
				input := yolo.ToFloat(rgba)
				addBench(stToFloat, time.Since(t3))
				t4 := time.Now()
				quantizeInput(input, inBytes, inScale, inOffset)
				addBench(stQuant, time.Since(t4))
				t5 := time.Now()
				ms, err = sess.Execute(inBytes, outBytes)
				addBench(stExecute, time.Since(t5))
				if err != nil {
					fmt.Fprintf(os.Stderr, "execute: %v\n", err)
					continue
				}
				t6 := time.Now()
				out := dequantizeOutput(outBytes, outScale, outOffset)
				addBench(stDequant, time.Since(t6))
				t7 := time.Now()
				cands := yolo.Argmax(out, float32(*conf))
				addBench(stArgmax, time.Since(t7))
				t8 := time.Now()
				picked := yolo.NMS(cands, 0.7)
				addBench(stNMS, time.Since(t8))
				boxes = yolo.Normalize(picked, scale, padX, padY, srcW, srcH)
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
			res := remoteResult{FrameID: frameID, Detections: dets, InferenceMs: ms}
			t9 := time.Now()
			payload, _ := json.Marshal(res)
			err = c.Write(ctx, websocket.MessageText, payload)
			addBench(stWrite, time.Since(t9))
			if err != nil {
				fmt.Fprintf(os.Stderr, "write: %v\n", err)
				c.Close(websocket.StatusAbnormalClosure, "write error")
				break
			}
			frames.Add(1)
			benchFrames.Add(1)
			msSum.Add(uint64(ms * 1000))
			lastMS.Store(uint64(ms * 1000))
		}
	}
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
