// Package tjpeg 用 libjpeg-turbo 把 JPEG 直接解码为 RGB。
// 仅用于 Termux 等安装了 libjpeg-turbo 的环境（cgo 构建）。
package tjpeg

/*
#cgo LDFLAGS: -lturbojpeg
#include <turbojpeg.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// FlagsFast 是实时解码标志: FASTDCT (快速 IDCT) + FASTUPSAMPLE (快速色度上采样)。
// 代价是像素值有轻微差异, 对检测框的影响远小于量化误差; 需要严格一致时传 0。
const FlagsFast = int(C.TJFLAG_FASTDCT | C.TJFLAG_FASTUPSAMPLE)

// DecodeJPEG 把 JPEG 解码为 RGB（每像素 3 字节），返回尺寸。
func DecodeJPEG(data []byte) (rgb []byte, w, h int, err error) {
	return DecodeJPEGFlags(data, 0)
}

// DecodeJPEGFlags 同 DecodeJPEG, 但可以传 libjpeg-turbo 的 flags (如 FlagsFast)。
func DecodeJPEGFlags(data []byte, flags int) (rgb []byte, w, h int, err error) {
	if len(data) == 0 {
		return nil, 0, 0, fmt.Errorf("empty jpeg data")
	}
	handle := C.tjInitDecompress()
	if handle == nil {
		return nil, 0, 0, fmt.Errorf("tjInitDecompress failed")
	}
	defer C.tjDestroy(handle)

	var cw, ch, subsamp, colorspace C.int
	rc := C.tjDecompressHeader3(handle,
		(*C.uchar)(unsafe.Pointer(&data[0])), C.ulong(len(data)),
		&cw, &ch, &subsamp, &colorspace)
	if rc != 0 {
		return nil, 0, 0, fmt.Errorf("tjDecompressHeader3: %s", C.GoString(C.tjGetErrorStr2(handle)))
	}
	w, h = int(cw), int(ch)
	if w <= 0 || h <= 0 {
		return nil, 0, 0, fmt.Errorf("bad jpeg size %dx%d", w, h)
	}
	rgb = make([]byte, w*h*3)
	rc = C.tjDecompress2(handle,
		(*C.uchar)(unsafe.Pointer(&data[0])), C.ulong(len(data)),
		(*C.uchar)(unsafe.Pointer(&rgb[0])), cw, 0, ch, C.TJPF_RGB, C.int(flags))
	if rc != 0 {
		return nil, 0, 0, fmt.Errorf("tjDecompress2: %s", C.GoString(C.tjGetErrorStr2(handle)))
	}
	return rgb, w, h, nil
}
