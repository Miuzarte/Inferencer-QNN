package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"image/jpeg"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"Inferencer/yolo"

	"github.com/coder/websocket"
)

type FrameResult struct {
	FrameID    uint32           `json:"frame_id"`
	Detections []yolo.Detection `json:"detections"`
}

func main() {
	var (
		host       = flag.String("host", "localhost:8080", "streamer WebSocket address")
		modelPath  = flag.String("model", "/data/data/com.termux/files/home/.local/lib/yolo26n_qnn_v81.onnx", "model path")
		ortLib     = flag.String("ort-lib", "/data/data/com.termux/files/home/.local/lib/libonnxruntime.so", "ONNX Runtime library path")
		qnnLib     = flag.String("qnn-lib", "/data/data/com.termux/files/home/.local/lib/libonnxruntime_providers_qnn.so", "QNN EP plugin path")
		qnnHtp     = flag.String("qnn-htp", "/vendor/lib64/libQnnHtp.so", "QNN HTP backend path")
		confThresh = flag.Float64("conf", 0.45, "confidence threshold")
		maxRetries = flag.Int("retry", 0, "max connect retries (0=infinite)")
	)
	flag.Parse()

	log.SetFlags(log.Ltime)
	log.Printf("Inferencer starting (Go %s/%s)", runtime.Version(), runtime.GOARCH)

	cfg := yolo.DefaultConfig()
	if *modelPath != "" {
		cfg.ModelPath = *modelPath
	}
	if *ortLib != "" {
		cfg.OrtLibPath = *ortLib
	}
	if *qnnLib != "" {
		cfg.QnnLibPath = *qnnLib
	}
	if *qnnHtp != "" {
		cfg.QnnHtpPath = *qnnHtp
	}
	if *confThresh > 0 {
		cfg.ConfThresh = *confThresh
	}

	det, err := yolo.New(cfg)
	if err != nil {
		log.Fatalf("Init detector: %v", err)
	}
	defer det.Close()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	var conn *websocket.Conn
	retries := 0
	for {
		conn, err = connect(ctx, *host)
		if err == nil {
			break
		}
		if *maxRetries > 0 && retries >= *maxRetries {
			log.Fatalf("Max retries exceeded: %v", err)
		}
		retries++
		wait := time.Duration(min(retries, 10)) * time.Second
		log.Printf("Connect failed (attempt %d): %v, retrying in %v", retries, err, wait)
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
	}
	defer conn.Close(websocket.StatusNormalClosure, "done")
	log.Printf("Connected to %s", *host)

	readLoop(ctx, conn, det)
}

func connect(ctx context.Context, host string) (*websocket.Conn, error) {
	url := "ws://" + host + "/stream"
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	conn, _, err := websocket.Dial(ctx, url, &websocket.DialOptions{
		CompressionMode: websocket.CompressionContextTakeover,
	})
	return conn, err
}

func readLoop(ctx context.Context, conn *websocket.Conn, det *yolo.Detector) {
	var totalFrames uint64
	var totalTime time.Duration

	for {
		msgType, data, err := conn.Read(ctx)
		if err != nil {
			log.Printf("Read error: %v", err)
			return
		}

		if msgType != websocket.MessageBinary {
			continue
		}

		if len(data) < 4 {
			log.Printf("Frame too short: %d bytes", len(data))
			continue
		}

		frameID := binary.LittleEndian.Uint32(data[:4])
		jpegData := data[4:]

		img, err := jpeg.Decode(bytes.NewReader(jpegData))
		if err != nil {
			log.Printf("JPEG decode error: %v", err)
			continue
		}

		start := time.Now()
		detections, err := det.Detect(img)
		elapsed := time.Since(start)

		if err != nil {
			log.Printf("Detect error: %v", err)
			continue
		}

		totalFrames++
		totalTime += elapsed

		if totalFrames%100 == 0 {
			avg := totalTime / time.Duration(totalFrames)
			log.Printf("Frame %d: %d detections, %v (avg %v)",
				frameID, len(detections), elapsed.Round(time.Microsecond), avg.Round(time.Microsecond))
		}

		result := FrameResult{
			FrameID:    frameID,
			Detections: detections,
		}
		jsonData, err := json.Marshal(result)
		if err != nil {
			log.Printf("JSON marshal error: %v", err)
			continue
		}

		if err := conn.Write(ctx, websocket.MessageText, jsonData); err != nil {
			log.Printf("Write error: %v", err)
			return
		}
	}
}
