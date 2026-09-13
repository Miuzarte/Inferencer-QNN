// Package yolo 实现 yolo26n QNN 模型的前处理与后处理，
// 逻辑与 androidApp/QnnYoloModel.kt 保持一致。
package yolo

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"sort"
)

const (
	ModelSize   = 640
	NumClasses  = 80
	OutFeatures = 84 // 4 (xywh) + 80 classes
)

// DetBox 归一化检测框（相对原图，0..1）。
type DetBox struct {
	X1, Y1, X2, Y2, Score float32
	Class                 int
	ClassName             string
}

// CocoNames 与 androidApp/CocoClasses.kt 一致。
var CocoNames = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
	"giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
	"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
	"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
	"toothbrush",
}

// Decode 解码图像（JPEG/PNG 等），返回图像与原始尺寸。
func Decode(data []byte) (img image.Image, srcW, srcH int, err error) {
	img, _, err = image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("jpeg decode: %w", err)
	}
	b := img.Bounds()
	return img, b.Dx(), b.Dy(), nil
}

// Letterbox 缩放并黑边填充到 640x640，返回 RGBA 图与映射参数。
func Letterbox(img image.Image, srcW, srcH int) (rgba *image.RGBA, scale float32, padX, padY int) {
	scale = float32(math.Min(float64(ModelSize)/float64(srcW), float64(ModelSize)/float64(srcH)))
	resizedW := int(math.Round(float64(srcW) * float64(scale)))
	resizedH := int(math.Round(float64(srcH) * float64(scale)))
	padX = (ModelSize - resizedW) / 2
	padY = (ModelSize - resizedH) / 2

	var resized *image.RGBA
	if resizedW == srcW && resizedH == srcH {
		resized = image.NewRGBA(image.Rect(0, 0, srcW, srcH))
		draw.Draw(resized, resized.Bounds(), img, img.Bounds().Min, draw.Src)
	} else {
		resized = resizeBilinear(img, resizedW, resizedH)
	}

	rgba = image.NewRGBA(image.Rect(0, 0, ModelSize, ModelSize))
	draw.Draw(rgba, image.Rect(padX, padY, padX+resizedW, padY+resizedH), resized, resized.Bounds().Min, draw.Src)
	return rgba, scale, padX, padY
}

// ToFloat 把 640x640 RGBA 转成 NHWC float32（值 0..1）。
func ToFloat(rgba *image.RGBA) []float32 {
	input := make([]float32, ModelSize*ModelSize*3)
	invStd := float32(1.0 / 255.0)
	for y := 0; y < ModelSize; y++ {
		row := y * rgba.Stride
		for x := 0; x < ModelSize; x++ {
			p := row + x*4
			o := (y*ModelSize + x) * 3
			input[o] = float32(rgba.Pix[p]) * invStd
			input[o+1] = float32(rgba.Pix[p+1]) * invStd
			input[o+2] = float32(rgba.Pix[p+2]) * invStd
		}
	}
	return input
}

// Preprocess 解码图像并做 letterbox 640x640，返回 NHWC float32（值 0..1）。
// 同时返回缩放参数，供 Postprocess 把结果映射回原图。
func Preprocess(data []byte) (input []float32, scale float32, padX, padY int, srcW, srcH int, err error) {
	img, srcW, srcH, err := Decode(data)
	if err != nil {
		return nil, 0, 0, 0, 0, 0, err
	}
	rgba, scale, padX, padY := Letterbox(img, srcW, srcH)
	return ToFloat(rgba), scale, padX, padY, srcW, srcH, nil
}

// quint16LUT 把 0..255 的 RGB 字节映射为 QNN 输入 quint16（scale=1/65536, offset=0）。
var quint16LUT [256]uint16

func init() {
	const invScale = float32(1.0 / 65536.0)
	for b := 0; b < 256; b++ {
		q := int32(float32(b)/255.0/invScale + 0.5)
		if q < 0 {
			q = 0
		} else if q > 65535 {
			q = 65535
		}
		quint16LUT[b] = uint16(q)
	}
}

// RGBToQuint16 把 RGB 字节流按 LUT 量化写入 QNN 输入缓冲（uint16 LE）。
func RGBToQuint16(rgb []byte, out []byte) {
	for i, v := range rgb {
		q := quint16LUT[v]
		out[i*2] = byte(q)
		out[i*2+1] = byte(q >> 8)
	}
}

// Candidate 是 NMS 前的候选框（640 尺度 letterbox 像素坐标）。
type Candidate struct {
	X1, Y1, X2, Y2, Score float32
	Class                 int
}

// Argmax 解析 [1,84,8400] 输出（feature-major，列 = anchor），
// 返回超过 conf 的候选框（640 尺度坐标，未做 NMS）。
func Argmax(out []float32, conf float32) []Candidate {
	const anchors = 8400
	if len(out) < OutFeatures*anchors {
		return nil
	}
	w := anchors
	numClasses := OutFeatures - 4
	var cands []Candidate
	for i := 0; i < w; i++ {
		cls := 0
		best := float32(math.Inf(-1))
		for c := 0; c < numClasses; c++ {
			score := out[(c+4)*w+i]
			if score > best {
				best = score
				cls = c
			}
		}
		if best < conf || cls >= len(CocoNames) {
			continue
		}
		cx := out[i]
		cy := out[w+i]
		bw := out[2*w+i]
		bh := out[3*w+i]
		cands = append(cands, Candidate{
			X1: cx - bw/2, Y1: cy - bh/2,
			X2: cx + bw/2, Y2: cy + bh/2,
			Score: best, Class: cls,
		})
	}
	return cands
}

// ArgmaxU16 直接在 QNN 原始 quint16 输出（uint16 LE）上做 argmax。
// 利用 offset=0、scale 恒正的性质，跳过全量反量化，只对候选框坐标乘 scale。
func ArgmaxU16(out []byte, conf, scale float32) []Candidate {
	const anchors = 8400
	if len(out) < OutFeatures*anchors*2 {
		return nil
	}
	w := anchors
	numClasses := OutFeatures - 4
	thresh := uint16(conf/scale) + 1 // best*scale < conf 等价于 best < conf/scale
	var cands []Candidate
	for i := 0; i < w; i++ {
		cls := 0
		best := uint16(0)
		for c := 0; c < numClasses; c++ {
			o := ((c+4)*w + i) * 2
			s := uint16(out[o]) | uint16(out[o+1])<<8
			if s > best {
				best = s
				cls = c
			}
		}
		if best < thresh || cls >= len(CocoNames) {
			continue
		}
		cx := float32(uint16(out[i*2])|uint16(out[i*2+1])<<8) * scale
		cy := float32(uint16(out[(w+i)*2])|uint16(out[(w+i)*2+1])<<8) * scale
		bw := float32(uint16(out[(2*w+i)*2])|uint16(out[(2*w+i)*2+1])<<8) * scale
		bh := float32(uint16(out[(3*w+i)*2])|uint16(out[(3*w+i)*2+1])<<8) * scale
		cands = append(cands, Candidate{
			X1: cx - bw/2, Y1: cy - bh/2,
			X2: cx + bw/2, Y2: cy + bh/2,
			Score: float32(best) * scale, Class: cls,
		})
	}
	return cands
}

// ArgmaxU16Class 只对指定类别求最大; classID < 0 或越界时退化为全类别扫描。
//
// person-only 时只读该类别的分数行 (8400 个 uint16 = 16.8KB), 不像 ArgmaxU16
// 那样把 84 行全扫一遍 (1.4MB, 且是跨行跨步访问); 坐标只在分数过阈值时才读。
func ArgmaxU16Class(out []byte, conf, scale float32, classID int) []Candidate {
	if classID < 0 || classID >= NumClasses {
		return ArgmaxU16(out, conf, scale)
	}
	const anchors = 8400
	if len(out) < OutFeatures*anchors*2 {
		return nil
	}
	w := anchors
	thresh := uint16(conf/scale) + 1 // best*scale < conf 等价于 best < conf/scale
	row := (classID + 4) * w
	var cands []Candidate
	for i := 0; i < w; i++ {
		o := (row + i) * 2
		best := uint16(out[o]) | uint16(out[o+1])<<8
		if best < thresh {
			continue
		}
		cx := float32(uint16(out[i*2])|uint16(out[i*2+1])<<8) * scale
		cy := float32(uint16(out[(w+i)*2])|uint16(out[(w+i)*2+1])<<8) * scale
		bw := float32(uint16(out[(2*w+i)*2])|uint16(out[(2*w+i)*2+1])<<8) * scale
		bh := float32(uint16(out[(3*w+i)*2])|uint16(out[(3*w+i)*2+1])<<8) * scale
		cands = append(cands, Candidate{
			X1: cx - bw/2, Y1: cy - bh/2,
			X2: cx + bw/2, Y2: cy + bh/2,
			Score: float32(best) * scale, Class: classID,
		})
	}
	return cands
}

// NMS 按分数降序做全局 NMS。
func NMS(cands []Candidate, iou float32) []Candidate {
	sort.Slice(cands, func(i, j int) bool { return cands[i].Score > cands[j].Score })
	var picked []Candidate
	for _, a := range cands {
		keep := true
		for _, b := range picked {
			interX := float32(math.Min(float64(a.X2), float64(b.X2)) - math.Max(float64(a.X1), float64(b.X1)))
			interY := float32(math.Min(float64(a.Y2), float64(b.Y2)) - math.Max(float64(a.Y1), float64(b.Y1)))
			inter := float32(math.Max(0, float64(interX))) * float32(math.Max(0, float64(interY)))
			union := (a.X2-a.X1)*(a.Y2-a.Y1) + (b.X2-b.X1)*(b.Y2-b.Y1) - inter
			if union > 0 && inter/union > iou {
				keep = false
				break
			}
		}
		if keep {
			picked = append(picked, a)
		}
	}
	return picked
}

// Normalize 把 640 尺度候选框映射回原图归一化坐标（0..1）。
func Normalize(cands []Candidate, scale float32, padX, padY, srcW, srcH int) []DetBox {
	xNorm := func(v float32) float32 {
		// 直连 QNN 的输出坐标已是 640 尺度像素（letterbox 图坐标）
		return clampf((v-float32(padX))/scale/float32(srcW), 0, 1)
	}
	yNorm := func(v float32) float32 {
		return clampf((v-float32(padY))/scale/float32(srcH), 0, 1)
	}
	boxes := make([]DetBox, 0, len(cands))
	for _, d := range cands {
		boxes = append(boxes, DetBox{
			X1: xNorm(d.X1), Y1: yNorm(d.Y1),
			X2: xNorm(d.X2), Y2: yNorm(d.Y2),
			Score: d.Score, Class: d.Class, ClassName: CocoNames[d.Class],
		})
	}
	return boxes
}

// Postprocess 解析 [1,84,8400] 输出（feature-major，列 = anchor），
// 返回相对原图归一化的检测框（全局 NMS，iou=0.7，与 Kotlin 一致）。
func Postprocess(out []float32, scale float32, padX, padY, srcW, srcH int, conf float32) []DetBox {
	return Normalize(NMS(Argmax(out, conf), 0.7), scale, padX, padY, srcW, srcH)
}

func clampf(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// resizeBilinear 双线性缩放。
func resizeBilinear(src image.Image, dstW, dstH int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	sb := src.Bounds()
	srcW, srcH := sb.Dx(), sb.Dy()
	draw.Draw(dst, dst.Bounds(), image.NewUniform(color.RGBA{0, 0, 0, 255}), image.Point{}, draw.Src)
	for y := 0; y < dstH; y++ {
		sy := (float64(y) + 0.5) * float64(srcH) / float64(dstH)
		y0 := int(math.Floor(sy))
		fy := sy - float64(y0)
		y0 = clampInt(y0, 0, srcH-1)
		y1 := clampInt(y0+1, 0, srcH-1)
		for x := 0; x < dstW; x++ {
			sx := (float64(x) + 0.5) * float64(srcW) / float64(dstW)
			x0 := int(math.Floor(sx))
			fx := sx - float64(x0)
			x0 = clampInt(x0, 0, srcW-1)
			x1 := clampInt(x0+1, 0, srcW-1)
			c00 := src.At(sb.Min.X+x0, sb.Min.Y+y0)
			c10 := src.At(sb.Min.X+x1, sb.Min.Y+y0)
			c01 := src.At(sb.Min.X+x0, sb.Min.Y+y1)
			c11 := src.At(sb.Min.X+x1, sb.Min.Y+y1)
			for _, ch := range []int{0, 1, 2, 3} {
				v00, v10, v01, v11 := chanVal(c00, ch), chanVal(c10, ch), chanVal(c01, ch), chanVal(c11, ch)
				v0 := v00*(1-fx) + v10*fx
				v1 := v01*(1-fx) + v11*fx
				v := v0*(1-fy) + v1*fy
				dst.Pix[dst.PixOffset(x, y)+ch] = uint8(clampInt(int(v+0.5), 0, 255))
			}
		}
	}
	return dst
}

func chanVal(c color.Color, ch int) float64 {
	r, g, b, a := c.RGBA()
	v := []uint32{r, g, b, a}[ch]
	return float64(v) / 257.0
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
