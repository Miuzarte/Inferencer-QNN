// RGB -> quint16 展开。
//
// QNN 的输入张量是 quint16（scale=1/65536, offset=0）的 NHWC RGB，而 JPEG 解出来是
// 每通道 1 字节，所以每帧都要把 1.2MB 展成 2.4MB。这一步在实时链路里是手机 CPU 的
// 大头（占 cpu_ms 的一半），但它的信息量是零：量化值只由 8 位像素决定。具体地
//
//	LUT[b] = round(b/255*65536) = b*257 + (128 <= b <= 254 ? 1 : 0)
//
// （b=255 时 b*257 = 65535 正好顶到 16 位上限，那个 +1 被 clamp 掉了），而 b*257 在
// 小端内存里就是 [b, b] —— 纯字节复制。所以 arm64 上用 NEON 的 vzip 做主体、只给
// 128..254 补那 1，比逐字节查表快 15 倍，且输出逐位相同。
//
// 平台实现见 quint16_neon.go（arm64 + cgo）与 quint16_scalar.go（其余情况）。
package yolo

// RGBToQuint16 把 RGB 字节流展开成 QNN 输入缓冲（quint16 小端），
// out 至少要 len(rgb)*2 字节；不足时只展开装得下的那部分（不做越界写）。
func RGBToQuint16(rgb, out []byte) {
	n := len(rgb)
	if n > len(out)/2 {
		n = len(out) / 2
	}
	if n <= 0 {
		return
	}
	expandQ16(rgb[:n], out)
}
