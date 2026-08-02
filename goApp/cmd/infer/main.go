// infer 在 Termux 中用 QNN HTP 跑 yolo26n context binary，对单张 JPEG 输出检测框。
//
// 用法:
//
//	go run ./cmd/infer -image test.jpg \
//	  -lib /data/data/com.termux/files/usr/lib -ctx models/ctx_v73.bin
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"time"

	"Inferencer/qnn"
	"Inferencer/yolo"
)

type detOut struct {
	X1, Y1, X2, Y2, Score float32 `json:",string"`
	Class                 int
	ClassName             string
}

func main() {
	imagePath := flag.String("image", "", "input JPEG path")
	ctxPath := flag.String("ctx", "models/ctx_v73.bin", "QNN context binary")
	libDir := flag.String("lib", "/data/data/com.termux/files/usr/lib", "QNN libs dir")
	arch := flag.Int("arch", 73, "HTP arch")
	conf := flag.Float64("conf", 0.45, "confidence threshold")
	outPath := flag.String("out", "", "optional output JPEG with boxes")
	verbose := flag.Bool("v", false, "verbose")
	flag.Parse()

	if *imagePath == "" {
		fmt.Fprintln(os.Stderr, "usage: infer -image test.jpg [-ctx models/ctx_v73.bin] [-lib ...]")
		os.Exit(1)
	}

	jpegData, err := os.ReadFile(*imagePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read image:", err)
		os.Exit(1)
	}
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
	inName, inDims, outName, outDims, inDtype, inScale, inOffset,
		outDtype, outScale, outOffset, err := sess.IOInfo()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("graph IO: %s %v (dtype=%d scale=%f off=%d) -> %s %v (dtype=%d scale=%f off=%d)\n",
		inName, inDims, inDtype, inScale, inOffset,
		outName, outDims, outDtype, outScale, outOffset)

	input, scale, padX, padY, srcW, srcH, err := yolo.Preprocess(jpegData)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	inBytes := quantizeInput(input, inScale, inOffset)
	outBytes := make([]byte, 84*8400*2)

	// 预热一次（首次加载 Skel/上下文）
	if _, err := sess.Execute(inBytes, outBytes); err != nil {
		fmt.Fprintln(os.Stderr, "warmup:", err)
		os.Exit(1)
	}

	const iters = 5
	var lastMS float64
	var boxes []yolo.DetBox
	for i := 0; i < iters; i++ {
		start := time.Now()
		ms, err := sess.Execute(inBytes, outBytes)
		if err != nil {
			fmt.Fprintln(os.Stderr, "execute:", err)
			os.Exit(1)
		}
		_ = start
		lastMS = ms
	}
	out := dequantizeOutput(outBytes, outScale, outOffset)
	boxes = yolo.Postprocess(out, scale, padX, padY, srcW, srcH, float32(*conf))

	fmt.Printf("inference_ms=%.2f detections=%d src=%dx%d letterbox=(%d,%d) scale=%.4f\n",
		lastMS, len(boxes), srcW, srcH, padX, padY, scale)
	dets := make([]detOut, 0, len(boxes))
	for _, b := range boxes {
		dets = append(dets, detOut{b.X1, b.Y1, b.X2, b.Y2, b.Score, b.Class, b.ClassName})
		fmt.Printf("  %-16s score=%.3f box=(%.3f, %.3f, %.3f, %.3f)\n",
			b.ClassName, b.Score, b.X1, b.Y1, b.X2, b.Y2)
	}
	if j, err := json.Marshal(dets); err == nil {
		fmt.Printf("json=%s\n", j)
	}

	if *outPath != "" {
		if err := drawBoxes(*imagePath, *outPath, boxes); err != nil {
			fmt.Fprintln(os.Stderr, "draw:", err)
		}
	}
}

// quantizeInput 将 float32（0..1）按 scale/offset 量化为 quint16。
func quantizeInput(v []float32, scale float32, offset int32) []byte {
	b := make([]byte, len(v)*2)
	for i, f := range v {
		q := int32(math.Round(float64(f/scale))) - offset
		if q < 0 {
			q = 0
		} else if q > 65535 {
			q = 65535
		}
		u := uint16(q)
		b[i*2] = byte(u)
		b[i*2+1] = byte(u >> 8)
	}
	return b
}

// dequantizeOutput 将 quint16 输出反量化为 float32。
func dequantizeOutput(b []byte, scale float32, offset int32) []float32 {
	v := make([]float32, len(b)/2)
	for i := range v {
		u := uint16(b[i*2]) | uint16(b[i*2+1])<<8
		v[i] = (float32(int32(u)) + float32(offset)) * scale
	}
	return v
}

func drawBoxes(srcPath, dstPath string, boxes []yolo.DetBox) error {
	f, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer f.Close()
	img, err := jpeg.Decode(f)
	if err != nil {
		return err
	}
	rgba := image.NewRGBA(img.Bounds())
	draw.Draw(rgba, rgba.Bounds(), img, img.Bounds().Min, draw.Src)
	b := rgba.Bounds()
	cyan := color.RGBA{0, 255, 255, 255}
	for _, d := range boxes {
		x1 := int(d.X1 * float32(b.Dx()))
		y1 := int(d.Y1 * float32(b.Dy()))
		x2 := int(d.X2 * float32(b.Dx()))
		y2 := int(d.Y2 * float32(b.Dy()))
		for x := x1; x <= x2; x++ {
			setPx(rgba, x, y1, cyan)
			setPx(rgba, x, y2, cyan)
		}
		for y := y1; y <= y2; y++ {
			setPx(rgba, x1, y, cyan)
			setPx(rgba, x2, y, cyan)
		}
	}
	out, err := os.Create(dstPath)
	if err != nil {
		return err
	}
	defer out.Close()
	if len(dstPath) > 4 && dstPath[len(dstPath)-4:] == ".png" {
		return png.Encode(out, rgba)
	}
	return jpeg.Encode(out, rgba, &jpeg.Options{Quality: 90})
}

func setPx(img *image.RGBA, x, y int, c color.RGBA) {
	if x < 0 || y < 0 || x >= img.Rect.Dx() || y >= img.Rect.Dy() {
		return
	}
	img.SetRGBA(x, y, c)
}
