// Package qnn 通过 cgo 直连 QNN HTP SDK（libQnnHtp.so / libQnnSystem.so）。
package qnn

/*
#cgo CFLAGS: -I../qnn-headers -I../qnn-headers/QNN
#cgo LDFLAGS: -ldl
#include <stdlib.h>
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Session 封装一个 QNN HTP 会话（backend + device + context + graph）。
type Session struct {
	c *C.qnn_session_t
}

// Create 创建会话。libDir 为包含 QNN 库的目录，arch 为 HTP 架构（如 73）。
func Create(libDir string, arch int, verbose bool) (*Session, error) {
	cdir := C.CString(libDir)
	defer C.free(unsafe.Pointer(cdir))
	cverb := 0
	if verbose {
		cverb = 1
	}
	c := C.qnn_create(cdir, C.int(arch), C.int(cverb))
	if c == nil {
		return nil, fmt.Errorf("qnn_create failed: %s", C.GoString(C.qnn_last_error()))
	}
	return &Session{c: c}, nil
}

// LoadBinary 加载 context binary（官方 ONNX 提取的 ep_cache_context）。
func (s *Session) LoadBinary(bin []byte) error {
	if s == nil || s.c == nil {
		return fmt.Errorf("session closed")
	}
	var ptr unsafe.Pointer
	if len(bin) > 0 {
		ptr = unsafe.Pointer(&bin[0])
	}
	rc := C.qnn_load_binary(s.c, ptr, C.uint64_t(len(bin)))
	if rc != 0 {
		return fmt.Errorf("qnn_load_binary: %s", C.GoString(C.qnn_last_error()))
	}
	return nil
}

// IOInfo 返回输入/输出 tensor 信息与量化参数（当前固定单输入单输出）。
func (s *Session) IOInfo() (inName string, inDims []uint32, outName string, outDims []uint32,
	inDtype int, inScale float32, inOffset int32,
	outDtype int, outScale float32, outOffset int32, err error) {
	if s == nil || s.c == nil {
		return "", nil, "", nil, 0, 0, 0, 0, 0, 0, fmt.Errorf("session closed")
	}
	const nameCap = 128
	inNameBuf := make([]C.char, nameCap)
	outNameBuf := make([]C.char, nameCap)
	inDimsBuf := make([]C.uint32_t, 8)
	outDimsBuf := make([]C.uint32_t, 8)
	var inRank, outRank, outCount C.uint32_t
	var cinDtype, coutDtype C.int
	var cinScale, coutScale C.float
	var cinOffset, coutOffset C.int32_t
	n := C.qnn_io_info(s.c, &inNameBuf[0], &inDimsBuf[0], &inRank,
		&outNameBuf[0], &outDimsBuf[0], &outRank, &outCount,
		&cinDtype, &cinScale, &cinOffset, &coutDtype, &coutScale, &coutOffset)
	if n == 0 || outCount == 0 {
		return "", nil, "", nil, 0, 0, 0, 0, 0, 0, fmt.Errorf("qnn_io_info: no tensors")
	}
	inName = C.GoString(&inNameBuf[0])
	outName = C.GoString(&outNameBuf[0])
	inDims = make([]uint32, inRank)
	for i := range inDims {
		inDims[i] = uint32(inDimsBuf[i])
	}
	outDims = make([]uint32, outRank)
	for i := range outDims {
		outDims[i] = uint32(outDimsBuf[i])
	}
	return inName, inDims, outName, outDims,
		int(cinDtype), float32(cinScale), int32(cinOffset),
		int(coutDtype), float32(coutScale), int32(coutOffset), nil
}

// Execute 执行推理。input/output 是原始字节缓冲（float32 布局，按 IOInfo 的 dims）。
func (s *Session) Execute(input, output []byte) (ms float64, err error) {
	if s == nil || s.c == nil {
		return 0, fmt.Errorf("session closed")
	}
	if len(input) == 0 || len(output) == 0 {
		return 0, fmt.Errorf("empty input/output buffer")
	}
	var cms C.double
	rc := C.qnn_execute(s.c, unsafe.Pointer(&input[0]), unsafe.Pointer(&output[0]), &cms)
	if rc != 0 {
		return 0, fmt.Errorf("qnn_execute: %s", C.GoString(C.qnn_last_error()))
	}
	return float64(cms), nil
}

// Close 释放会话。
func (s *Session) Close() {
	if s != nil && s.c != nil {
		C.qnn_destroy(s.c)
		s.c = nil
	}
}
