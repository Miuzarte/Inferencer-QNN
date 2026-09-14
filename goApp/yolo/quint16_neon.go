//go:build arm64 && cgo

package yolo

/*
#cgo CFLAGS: -O2
#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

// expandQ16NEON 把 RGB 字节流展开成 quint16 小端 (推导见 quint16.go)。
//
// 主体: 每个输入字节 b 展开成 [b, b], 即 vzip1/vzip2 把向量和自身交错。
// 修正: LUT[b] = b*257 + 1 只发生在 128 <= b <= 254, 给低位字节补 1 即可;
//       b == 255 必须排除, 否则低位字节 255+1 会进位破坏高位字节。
static inline void expandQ16NEON(const uint8_t *rgb, uint8_t *out, size_t n) {
	size_t i = 0;
	const uint8x16_t zero = vdupq_n_u8(0);
	const uint8x16_t one = vdupq_n_u8(1);
	const uint8x16_t c128 = vdupq_n_u8(128);
	const uint8x16_t c255 = vdupq_n_u8(255);

	for (; i + 16 <= n; i += 16) {
		uint8x16_t v = vld1q_u8(rgb + i);
		uint8x16_t need = vandq_u8(vcgeq_u8(v, c128), vmvnq_u8(vceqq_u8(v, c255)));
		uint8x16_t inc1 = vandq_u8(vzip1q_u8(need, zero), one);
		uint8x16_t inc2 = vandq_u8(vzip2q_u8(need, zero), one);
		vst1q_u8(out + i * 2, vaddq_u8(vzip1q_u8(v, v), inc1));
		vst1q_u8(out + i * 2 + 16, vaddq_u8(vzip2q_u8(v, v), inc2));
	}
	for (; i < n; i++) {
		uint16_t q = (uint16_t)((uint16_t)rgb[i] * 257u);
		if (rgb[i] >= 128 && rgb[i] != 255) {
			q++;
		}
		out[i * 2] = (uint8_t)q;
		out[i * 2 + 1] = (uint8_t)(q >> 8);
	}
}
*/
import "C"

import "unsafe"

// expandQ16 用 NEON 展开。这条路径只在真机 (android/arm64) 上生效。
func expandQ16(rgb, out []byte) {
	C.expandQ16NEON(
		(*C.uint8_t)(unsafe.Pointer(&rgb[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		C.size_t(len(rgb)),
	)
}
