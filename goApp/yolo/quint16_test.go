package yolo

import (
	"bytes"
	"math/rand"
	"testing"
)

// referenceQ16LUT 独立重算参考表，不复用实现里的 quint16LUT —— 否则两边一起错就测不出来。
func referenceQ16LUT() [256]uint16 {
	var l [256]uint16
	const invScale = float32(1.0 / 65536.0)
	for b := range 256 {
		q := int32(float32(b)/255.0/invScale + 0.5)
		if q < 0 {
			q = 0
		} else if q > 65535 {
			q = 65535
		}
		l[b] = uint16(q)
	}
	return l
}

func referenceQ16(rgb, out []byte) {
	l := referenceQ16LUT()
	for i, v := range rgb {
		q := l[v]
		out[i*2] = byte(q)
		out[i*2+1] = byte(q >> 8)
	}
}

// TestQ16MatchesReference 用覆盖全部 256 个取值的输入比对当前实现与参考表。
// 长度取遍 NEON 主循环边界 (16 的倍数前后) 与真实帧长 640*640*3。
func TestQ16MatchesReference(t *testing.T) {
	sizes := []int{1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 255, 1000, 640*640*3 - 1, 640 * 640 * 3}
	rnd := rand.New(rand.NewSource(7))
	for _, n := range sizes {
		rgb := make([]byte, n)
		for i := range rgb {
			rgb[i] = byte(i % 256) // 轮转保证 0..255 每个值都出现
		}
		for i := 0; i < n/4; i++ {
			rgb[rnd.Intn(n)] = byte(rnd.Intn(256))
		}

		want := make([]byte, n*2)
		referenceQ16(rgb, want)
		got := make([]byte, n*2)
		RGBToQuint16(rgb, got)

		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("n=%d: 第 %d 字节不同 got=%#x want=%#x (输入字节 %d)",
						n, i, got[i], want[i], rgb[i/2])
				}
			}
		}
	}
}

// TestQ16EdgeCases 单独钉住两个关键取值: b=128 (第一个需要 +1 的) 与 b=255
// (唯一一个绝不能 +1 的, 否则低位字节进位会污染高位字节)。
func TestQ16EdgeCases(t *testing.T) {
	rgb := []byte{0, 1, 127, 128, 129, 254, 255}
	got := make([]byte, len(rgb)*2)
	want := make([]byte, len(rgb)*2)
	referenceQ16(rgb, want)
	RGBToQuint16(rgb, got)
	if !bytes.Equal(got, want) {
		t.Fatalf("got  %v\nwant %v", got, want)
	}

	// 硬编码几个值的期望结果, 不依赖任何实现
	checks := []struct {
		b    byte
		lo   byte
		hi   byte
		want uint16
	}{
		{0, got[0], got[1], 0},
		{127, got[4], got[5], 32639},   // 127*257
		{128, got[6], got[7], 32897},   // 128*257+1
		{255, got[12], got[13], 65535}, // 255*257 (那个 +1 被 clamp)
	}
	for _, c := range checks {
		if v := uint16(c.lo) | uint16(c.hi)<<8; v != c.want {
			t.Errorf("b=%d: got %d want %d", c.b, v, c.want)
		}
	}
}

// TestQ16ShortOut 确认 out 装不下时只展开装得下的部分, 且不越界写。
func TestQ16ShortOut(t *testing.T) {
	rgb := make([]byte, 100)
	for i := range rgb {
		rgb[i] = byte(i * 2)
	}

	const guard = 0xAA
	backing := make([]byte, 64)
	for i := range backing {
		backing[i] = guard
	}
	out := backing[:50] // 只装得下 25 个像素

	RGBToQuint16(rgb, out)

	for i := 50; i < len(backing); i++ {
		if backing[i] != guard {
			t.Fatalf("越界写: backing[%d]=%#x", i, backing[i])
		}
	}
	want := make([]byte, 50)
	referenceQ16(rgb[:25], want)
	if !bytes.Equal(out, want) {
		t.Fatalf("截断结果不符:\ngot  %v\nwant %v", out, want)
	}
}
