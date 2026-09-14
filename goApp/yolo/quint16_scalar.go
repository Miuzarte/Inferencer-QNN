//go:build !arm64 || !cgo

package yolo

// quint16LUT 把 0..255 的 RGB 字节映射为 QNN 输入 quint16（scale=1/65536, offset=0）。
// 只被标量实现使用；arm64+cgo 下走 NEON，不查表。
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

// expandQ16 是 arm64 之外的退路（PC 上的 infer/htpbench，或 CGO_ENABLED=0 的构建）。
func expandQ16(rgb, out []byte) {
	for i, v := range rgb {
		q := quint16LUT[v]
		out[i*2] = byte(q)
		out[i*2+1] = byte(q >> 8)
	}
}
