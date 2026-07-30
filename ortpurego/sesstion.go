package ort

import (
	"fmt"
	"unsafe"
)

type Session struct {
	handle      SessionHandle
	engine      *Engine
	InputNames  []string
	OutputNames []string
}

type SessionOptions struct {
	handle SessionOptionsHandle
	engine *Engine
}

func (e *Engine) NewSessionOptions() (*SessionOptions, error) {
	var h SessionOptionsHandle
	status := e.funcs.createSessionOptions(&h)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}
	return &SessionOptions{handle: h, engine: e}, nil
}

// SetIntraOpNumThreads 设置线程数
func (o *SessionOptions) SetIntraOpNumThreads(num int32) error {
	return o.engine.checkStatus(o.engine.funcs.setIntraOpNumThreads(o.handle, num))
}

// SetCpuMemArena 设置内存池策略
//
//	false: 禁用内存池，推理速度稍慢，但 Destroy 后立即归还内存给 OS ，解决内存滞留问题
//	true: 启用内存池，推理速度最快，但 Destroy 后内存会被缓存以供复用（默认）
func (o *SessionOptions) SetCpuMemArena(useArena bool) error {
	if useArena {
		return o.engine.checkStatus(o.engine.funcs.enableCpuMemArena(o.handle))
	}
	return o.engine.checkStatus(o.engine.funcs.disableCpuMemArena(o.handle))
}

// AddSessionConfigEntry 添加会话配置项
func (o *SessionOptions) AddSessionConfigEntry(key, value string) error {
	kPtr, err := stringToCString(key)
	if err != nil {
		return fmt.Errorf("failed to prepare config key: %w", err)
	}
	vPtr, err := stringToCString(value)
	if err != nil {
		return fmt.Errorf("failed to prepare config value: %w", err)
	}
	status := o.engine.funcs.addSessionConfigEntry(o.handle, kPtr, vPtr)
	return o.engine.checkStatus(status)
}

// EnableCUDA 启用 CUDA
func (o *SessionOptions) EnableCUDA() error {
	var cudaOpts CUDAProviderOptionsV2Handle
	status := o.engine.funcs.createCUDAProviderOptions(&cudaOpts)
	if err := o.engine.checkStatus(status); err != nil {
		return fmt.Errorf("failed to create CUDA provider options: %w", err)
	}
	defer o.engine.funcs.releaseCUDAProviderOptions(cudaOpts)

	status = o.engine.funcs.appendExecutionProvider_CUDA_V2(o.handle, cudaOpts)
	return o.engine.checkStatus(status)
}

// RegisterExecutionProviderLibrary 注册 EP 插件库
func (e *Engine) RegisterExecutionProviderLibrary(providerName string, libPath string) error {
	namePtr, err := stringToCString(providerName)
	if err != nil {
		return fmt.Errorf("failed to prepare provider name CString: %w", err)
	}
	pathPtr, err := stringToPathPtr(libPath)
	if err != nil {
		return fmt.Errorf("failed to prepare library path CStringW: %w", err)
	}
	status := e.funcs.registerExecutionProviderLibrary(e.envHandle, namePtr, pathPtr)
	return e.checkStatus(status)
}

// GetEpDevices 获取所有 EP 设备
func (e *Engine) GetEpDevices() ([]uintptr, error) {
	var devicesPtr *uintptr
	var numDevices uintptr
	status := e.funcs.getEpDevices(e.envHandle, &devicesPtr, &numDevices)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}

	devices := make([]uintptr, numDevices)
	for i := uintptr(0); i < numDevices; i++ {
		ptr := (*uintptr)(unsafe.Pointer(uintptr(unsafe.Pointer(devicesPtr)) + i*unsafe.Sizeof(uintptr(0))))
		devices[i] = *ptr
	}
	return devices, nil
}

// GetEpDeviceName 获取 EP 设备名称
func (e *Engine) GetEpDeviceName(device uintptr) (string, error) {
	namePtr := e.funcs.epDeviceEpName(device)
	if namePtr == nil {
		return "", fmt.Errorf("EpDevice_EpName returned null namePtr")
	}
	return cStringToString(namePtr), nil
}

// buildKVArrays 将 map[string]string 构建为 C 数组
func buildKVArrays(opts map[string]string) (**byte, **byte, uintptr, error) {
	numKeys := len(opts)
	if numKeys == 0 {
		return nil, nil, 0, nil
	}
	keys := make([]*byte, numKeys)
	vals := make([]*byte, numKeys)
	i := 0
	for k, v := range opts {
		kPtr, err := stringToCString(k)
		if err != nil {
			return nil, nil, 0, err
		}
		vPtr, err := stringToCString(v)
		if err != nil {
			return nil, nil, 0, err
		}
		keys[i], vals[i] = kPtr, vPtr
		i++
	}
	return &keys[0], &vals[0], uintptr(numKeys), nil
}

// AppendExecutionProviderV2 使用 V2 API 挂载 EP 设备
func (o *SessionOptions) AppendExecutionProviderV2(devices []uintptr, opts map[string]string) error {
	var devPtr *uintptr
	if len(devices) > 0 {
		devPtr = &devices[0]
	}
	pKeys, pVals, nKeys, err := buildKVArrays(opts)
	if err != nil {
		return err
	}
	status := o.engine.funcs.sessionOptionsAppendExecutionProviderV2(
		o.handle, o.engine.envHandle, devPtr, uintptr(len(devices)), pKeys, pVals, nKeys,
	)
	return o.engine.checkStatus(status)
}

// EnableTensorRT 启用 TensorRT
// 注：V1 by-name API 对 NVIDIA 定制 ORT 无效，请使用 RegisterExecutionProviderLibrary + AppendExecutionProviderV2
func (o *SessionOptions) EnableTensorRT(opts map[string]string) error {
	providerNamePtr, err := stringToCString("NvTensorRTRTXExecutionProvider")
	if err != nil {
		return fmt.Errorf("failed to prepare TensorRT provider name CString: %w", err)
	}
	pKeys, pVals, nKeys, err := buildKVArrays(opts)
	if err != nil {
		return err
	}
	status := o.engine.funcs.sessionOptionsAppendExecutionProvider(o.handle, providerNamePtr, pKeys, pVals, nKeys)
	return o.engine.checkStatus(status)
}

func (o *SessionOptions) Destroy() {
	if o.handle != 0 {
		o.engine.funcs.releaseSessionOptions(o.handle)
		o.handle = 0
	}
}

// NewSession 创建会话
//
// # Params:
//
//	modelPath: 模型路径
//	opts: Session 配置项
func (e *Engine) NewSession(modelPath string, opts *SessionOptions) (*Session, error) {
	var optHandle SessionOptionsHandle
	if opts != nil {
		optHandle = opts.handle
	}

	pathPtr, err := stringToPathPtr(modelPath)
	if err != nil {
		return nil, err
	}

	var h SessionHandle
	status := e.funcs.createSession(e.envHandle, pathPtr, optHandle, &h)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}

	s := &Session{
		handle: h,
		engine: e,
	}

	if err := s.initMetadata(); err != nil {
		s.Destroy()
		return nil, err
	}

	return s, nil
}

func (s *Session) initMetadata() error {
	// input
	inputCount, err := s.getInputCount()
	if err != nil {
		return err
	}
	s.InputNames = make([]string, inputCount)
	for i := 0; i < inputCount; i++ {
		name, err := s.getInputName(i)
		if err != nil {
			return err
		}
		s.InputNames[i] = name
	}

	// output
	outputCount, err := s.getOutputCount()
	if err != nil {
		return err
	}
	s.OutputNames = make([]string, outputCount)
	for i := 0; i < outputCount; i++ {
		name, err := s.getOutputName(i)
		if err != nil {
			return err
		}
		s.OutputNames[i] = name
	}

	return nil
}

func (s *Session) getInputCount() (int, error) {
	var count uintptr
	status := s.engine.funcs.sessionGetInputCount(s.handle, &count)
	return int(count), s.engine.checkStatus(status)
}

func (s *Session) getOutputCount() (int, error) {
	var count uintptr
	status := s.engine.funcs.sessionGetOutputCount(s.handle, &count)
	return int(count), s.engine.checkStatus(status)
}

func (s *Session) getInputName(index int) (string, error) {
	var allocator AllocatorHandle
	status := s.engine.funcs.getAllocatorWithDefaultOptions(&allocator)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	var namePtr *byte
	status = s.engine.funcs.sessionGetInputName(s.handle, uintptr(index), allocator, &namePtr)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	name := cStringToString(namePtr)
	// 释放内存
	s.engine.funcs.allocatorFree(allocator, unsafe.Pointer(namePtr))

	return name, nil
}

func (s *Session) getOutputName(index int) (string, error) {
	var allocator AllocatorHandle
	status := s.engine.funcs.getAllocatorWithDefaultOptions(&allocator)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	var namePtr *byte
	status = s.engine.funcs.sessionGetOutputName(s.handle, uintptr(index), allocator, &namePtr)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	name := cStringToString(namePtr)
	s.engine.funcs.allocatorFree(allocator, unsafe.Pointer(namePtr))

	return name, nil
}

func (s *Session) Destroy() {
	if s.handle != 0 {
		s.engine.funcs.releaseSession(s.handle)
		s.handle = 0
	}
}

// Run 执行推理
func (s *Session) Run(inputs map[string]*Value) (map[string]*Value, error) {
	inputCount := len(inputs)
	outputCount := len(s.OutputNames)

	// input
	inputNamePtrs := make([]unsafe.Pointer, inputCount)
	inputHandles := make([]ValueHandle, inputCount)
	i := 0
	for name, val := range inputs {
		cName, err := stringToCString(name)
		if err != nil {
			return nil, err
		}
		inputNamePtrs[i] = unsafe.Pointer(cName)
		inputHandles[i] = val.handle
		i++
	}

	// output
	outputNamePtrs := make([]unsafe.Pointer, outputCount)
	outputHandles := make([]ValueHandle, outputCount)
	for i, name := range s.OutputNames {
		cName, err := stringToCString(name)
		if err != nil {
			return nil, err
		}
		outputNamePtrs[i] = unsafe.Pointer(cName)
	}

	// 调用底层执行推理
	status := s.engine.funcs.run(
		s.handle,
		0,
		&inputNamePtrs[0],
		&inputHandles[0],
		uintptr(inputCount),
		&outputNamePtrs[0],
		uintptr(outputCount),
		&outputHandles[0],
	)

	if err := s.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to run session: %w", err)
	}

	results := make(map[string]*Value, outputCount)
	for i := 0; i < outputCount; i++ {
		results[s.OutputNames[i]] = &Value{
			handle: outputHandles[i],
			engine: s.engine,
		}
	}

	return results, nil
}
