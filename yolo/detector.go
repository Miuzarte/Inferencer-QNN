package yolo

import (
	"fmt"
	"image"
	"math"
	"sync"

	"Inferencer/ortpurego"
)

const (
	InputW     = 640
	InputH     = 640
	InputCh    = 3
	InputName  = "images"
	OutputName = "output0"
	OutputNum  = 300
	OutputDim  = 6
	PlaneSize  = InputW * InputH
)

var cocoNames = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
	"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
	"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

type Detection struct {
	X1         float64 `json:"x1"`
	Y1         float64 `json:"y1"`
	X2         float64 `json:"x2"`
	Y2         float64 `json:"y2"`
	Score      float64 `json:"score"`
	Class      int     `json:"class"`
	ClassName  string  `json:"class_name"`
}

type Detector struct {
	session     *ort.Session
	confThresh  float64
	inputData   []float32
	mu          sync.Mutex
}

type Config struct {
	ModelPath   string
	OrtLibPath  string
	QnnLibPath  string
	ConfThresh  float64
}

func DefaultConfig() Config {
	return Config{
		ModelPath:   "/root/.local/lib/yolo26n_qnn_v81.onnx",
		OrtLibPath:  "/root/.local/lib/onnxruntime/lib/libonnxruntime.so",
		QnnLibPath:  "/root/.local/lib/onnxruntime-qnn/libonnxruntime_providers_qnn.so",
		ConfThresh:  0.45,
	}
}

func New(cfg Config) (*Detector, error) {
	engine, err := ort.NewEngine(cfg.OrtLibPath)
	if err != nil {
		return nil, fmt.Errorf("init ORT engine: %w", err)
	}
	fmt.Printf("ORT version: %s\n", engine.GetVersion())

	if err := engine.RegisterExecutionProviderLibrary("QNNExecutionProvider", cfg.QnnLibPath); err != nil {
		return nil, fmt.Errorf("register QNN EP: %w", err)
	}
	fmt.Println("QNN EP registered")

	devices, err := engine.GetEpDevices()
	if err != nil {
		return nil, fmt.Errorf("get EP devices: %w", err)
	}

	var qnnDevice uintptr
	for _, d := range devices {
		name, err := engine.GetEpDeviceName(d)
		if err != nil {
			continue
		}
		fmt.Printf("  EP device: %s\n", name)
		if name == "QNNExecutionProvider" {
			qnnDevice = d
		}
	}

	if qnnDevice == 0 {
		fmt.Println("WARNING: QNNExecutionProvider not found, running on CPU")
	}

	opts, err := engine.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer func() {
		if err != nil {
			opts.Destroy()
		}
	}()

	if qnnDevice != 0 {
		if err := opts.AppendExecutionProviderV2([]uintptr{qnnDevice}, map[string]string{
			"backend_type":                 "htp",
			"htp_arch":                     "81",
			"enable_htp_fp16_precision":    "1",
			"enable_htp_shared_memory_allocator": "1",
		}); err != nil {
			return nil, fmt.Errorf("append QNN EP: %w", err)
		}
		fmt.Println("QNN EP appended (HTP v81)")
	}

	if err := opts.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1"); err != nil {
		fmt.Printf("WARNING: disable_cpu_ep_fallback failed: %v\n", err)
	}

	session, err := engine.NewSession(cfg.ModelPath, opts)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	fmt.Printf("Model loaded: %s\n", cfg.ModelPath)
	for i, name := range session.InputNames {
		fmt.Printf("  Input %d: %s\n", i, name)
	}
	for i, name := range session.OutputNames {
		fmt.Printf("  Output %d: %s\n", i, name)
	}

	return &Detector{
		session:    session,
		confThresh: cfg.ConfThresh,
		inputData:  make([]float32, InputCh*PlaneSize),
	}, nil
}

func (d *Detector) preprocess(img image.Image) []float32 {
	rgba, ok := img.(*image.RGBA)
	if !ok || rgba.Bounds().Dx() != InputW || rgba.Bounds().Dy() != InputH {
		rgba = image.NewRGBA(image.Rect(0, 0, InputW, InputH))
		for y := 0; y < InputH; y++ {
			for x := 0; x < InputW; x++ {
				r, g, b, _ := img.At(x, y).RGBA()
				off := (y*InputW + x) * 4
				rgba.Pix[off] = uint8(r >> 8)
				rgba.Pix[off+1] = uint8(g >> 8)
				rgba.Pix[off+2] = uint8(b >> 8)
				rgba.Pix[off+3] = 255
			}
		}
	}

	clear(d.inputData)
	for y := 0; y < InputH; y++ {
		for x := 0; x < InputW; x++ {
			off := (y*InputW + x) * 4
			idx := y*InputW + x
			d.inputData[idx] = float32(rgba.Pix[off]) / 255.0
			d.inputData[PlaneSize+idx] = float32(rgba.Pix[off+1]) / 255.0
			d.inputData[2*PlaneSize+idx] = float32(rgba.Pix[off+2]) / 255.0
		}
	}
	return d.inputData
}

func (d *Detector) postprocess(output []float32) []Detection {
	detections := make([]Detection, 0, 16)
	for i := 0; i < OutputNum; i++ {
		off := i * OutputDim
		x1 := float64(output[off+0])
		y1 := float64(output[off+1])
		x2 := float64(output[off+2])
		y2 := float64(output[off+3])
		score := float64(output[off+4])
		classID := int(output[off+5])

		if score < d.confThresh || classID < 0 || classID >= len(cocoNames) {
			continue
		}

		detections = append(detections, Detection{
			X1:        clamp01(x1 / InputW),
			Y1:        clamp01(y1 / InputH),
			X2:        clamp01(x2 / InputW),
			Y2:        clamp01(y2 / InputH),
			Score:     math.Round(score*10000) / 10000,
			Class:     classID,
			ClassName: cocoNames[classID],
		})
	}
	return detections
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return math.Round(v*10000) / 10000
}

func (d *Detector) Detect(img image.Image) ([]Detection, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	inputData := d.preprocess(img)
	tensor, err := ort.NewTensor([]int64{1, InputCh, InputW, InputH}, inputData)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outputs, err := d.session.Run(map[string]*ort.Value{InputName: tensor})
	if err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	outputTensor, ok := outputs[OutputName]
	if !ok {
		return nil, fmt.Errorf("output %q not found in results", OutputName)
	}

	outputData, err := ort.GetTensorData[float32](outputTensor)
	if err != nil {
		return nil, fmt.Errorf("get output data: %w", err)
	}

	return d.postprocess(outputData), nil
}

func (d *Detector) Close() {
	if d.session != nil {
		d.session.Destroy()
	}
}
