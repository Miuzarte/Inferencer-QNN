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
	"errors"
	"fmt"
	"unsafe"

	"Inferencer/qnn/perf"
)

// ErrPerfUnsupported 表示后端/设备不提供 HTP 性能基础设施。
// 调用方应把它当作"降级继续跑", 而不是启动失败。
var ErrPerfUnsupported = errors.New("HTP performance infrastructure not supported")

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

// PerfInit 请求 HTP 性能基础设施并创建 power config id。
//
// 必须在 Create 之后、LoadBinary 之前调用 (与 QIDK 的
// deviceCreate -> perf -> contextCreate 顺序一致)。
// 返回 ErrPerfUnsupported 表示设备/后端不支持, 应降级继续。
func (s *Session) PerfInit() error {
	if s == nil || s.c == nil {
		return fmt.Errorf("session closed")
	}
	switch rc := C.qnn_perf_init(s.c); rc {
	case 0:
		return nil
	case -2:
		return fmt.Errorf("%w: %s", ErrPerfUnsupported, C.GoString(C.qnn_last_error()))
	default:
		return fmt.Errorf("qnn_perf_init: %s", C.GoString(C.qnn_last_error()))
	}
}

// PerfStatus 返回性能基础设施状态。ok 为 false 表示未初始化或不可用。
func (s *Session) PerfStatus() (infraType int, powerConfigID uint32, ok bool) {
	if s == nil || s.c == nil {
		return 0, 0, false
	}
	var it C.int
	var pid C.uint
	if C.qnn_perf_status(s.c, &it, &pid) == 0 {
		return 0, 0, false
	}
	return int(it), uint32(pid), true
}

// PerfApply 下发性能档位, 可反复调用换档。
//
// degraded 为 true 表示整组被拒后退化为只下发 DCVS_V3 (例如本 HTP 版本不支持
// RPC_POLLING_TIME)。cfg 没有任何配置项时是空操作 (用于 default 基线档)。
func (s *Session) PerfApply(cfg perf.Config) (degraded bool, err error) {
	if s == nil || s.c == nil {
		return false, fmt.Errorf("session closed")
	}
	if !cfg.HasAny() {
		return false, nil
	}
	c := C.qnn_power_cfg{
		power_mode:          C.int(cfg.PowerMode),
		dcvs_enable:         C.int(cfg.DcvsEnable),
		sleep_latency:       C.int(cfg.SleepLatency),
		sleep_disable:       C.int(cfg.SleepDisable),
		bus_vc_min:          C.int(cfg.BusMin),
		bus_vc_target:       C.int(cfg.BusTarget),
		bus_vc_max:          C.int(cfg.BusMax),
		core_vc_min:         C.int(cfg.CoreMin),
		core_vc_target:      C.int(cfg.CoreTarget),
		core_vc_max:         C.int(cfg.CoreMax),
		rpc_control_latency: C.int(cfg.RpcControlLatency),
		rpc_polling_time:    C.int(cfg.RpcPollingTime),
	}
	switch rc := C.qnn_perf_apply(s.c, &c); rc {
	case 0:
		return false, nil
	case 1:
		return true, nil
	case -2:
		return false, fmt.Errorf("%w: %s", ErrPerfUnsupported, C.GoString(C.qnn_last_error()))
	default:
		return false, fmt.Errorf("qnn_perf_apply: %s", C.GoString(C.qnn_last_error()))
	}
}

// PlatformInfo 查询平台信息 (HTP 架构/SOC/VTCM/核数)。
// 返回的 PlatformInfo.Valid 为 false 表示该后端不提供此信息, 不算错误。
func (s *Session) PlatformInfo() (perf.PlatformInfo, error) {
	if s == nil || s.c == nil {
		return perf.PlatformInfo{}, fmt.Errorf("session closed")
	}
	var ci C.qnn_plat_info
	if rc := C.qnn_platform_info(s.c, &ci); rc != 0 {
		return perf.PlatformInfo{}, fmt.Errorf("qnn_platform_info: %s", C.GoString(C.qnn_last_error()))
	}
	return perf.PlatformInfo{
		Valid:      ci.valid != 0,
		Arch:       int(ci.arch),
		SocModel:   int(ci.soc_model),
		VtcmMB:     int(ci.vtcm_mb),
		SignedPD:   ci.signed_pd != 0,
		Dlbc:       ci.dlbc != 0,
		DevType:    int(ci.dev_type),
		NumDevices: int(ci.num_devices),
		NumCores:   int(ci.num_cores),
	}, nil
}
