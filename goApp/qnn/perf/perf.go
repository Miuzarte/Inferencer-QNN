// Package perf 定义 QNN HTP 性能档位 (power config) 的参数表。
//
// 本包刻意不 import "C": 档位表要能在没有 NDK / QNN 头文件的机器上直接
// go test, 而 qnn 包带 cgo, 测试时无法这样跑。
//
// 结构体字段的字面值直接对应 QNN 2.46 的枚举 (换 SDK 时必须核对):
//
//	QnnHtpPerfInfrastructure_PowerMode_t    qnn-headers/QNN/HTP/QnnHtpPerfInfrastructure.h:290
//	QnnHtpPerfInfrastructure_VoltageCorner_t 同上 :174
//	QnnHtpPerfInfrastructure_PowerConfigOption_t 同上 :376
//
// 档位取值与厂商实现一致:
//   - Qualcomm QIDK VisionSolution1 (inference.cpp:69-238) 的 setPerfConfig
//   - onnxruntime-qnn (qnn_htp_power_config_manager.cc) 的 htp_performance_mode
//   - ai-engine-direct-helper (QnnInferenceEngine.cpp:93-146) 的 boostPerformance
//
// 一个已知的语义差异: burst 在 ORT/QNN sample 里是 sleepDisable=0, 在 QIDK 里
// 是 sleepDisable=1 (HTP 完全不睡)。两者都保留, 用来隔离"休眠唤醒延迟"这一项。
package perf

import (
	"fmt"
	"strings"
)

// DCVS 功率模式 (QnnHtpPerfInfrastructure_PowerMode_t)。
const (
	POWER_MODE_ADJUST_UP_DOWN         = 0x1
	POWER_MODE_ADJUST_ONLY_UP         = 0x2
	POWER_MODE_POWER_SAVER            = 0x4
	POWER_MODE_POWER_SAVER_AGGRESSIVE = 0x8
	POWER_MODE_PERFORMANCE            = 0x10
	POWER_MODE_DUTY_CYCLE             = 0x20
)

// 电压角 (QnnHtpPerfInfrastructure_VoltageCorner_t)。
const (
	VCORNER_DISABLE    = 0x10
	VCORNER_MIN        = 0x20
	VCORNER_SVS2       = 0x30
	VCORNER_SVS        = 0x40
	VCORNER_SVS_PLUS   = 0x50
	VCORNER_NOM        = 0x60
	VCORNER_NOM_PLUS   = 0x70
	VCORNER_TURBO      = 0x80
	VCORNER_TURBO_L2   = 0x92
	VCORNER_TURBO_PLUS = 0x90
	VCORNER_MAX        = 0xA0
)

// Unset 用于 RpcControlLatency / RpcPollingTime, 表示"不下发这一项"。
// 与 0 区分: 0 是显式下发 0 (关闭该项)。
const Unset = -1

// 档位里用到的字面值 (ORT 的 qnn_def.h:162-170)。
const (
	sleepLatencyMin    = 40 // 允许的最短 HTP 休眠等待
	sleepLatencyLow    = 100
	sleepLatencyMedium = 1000
	rpcLatencyDefault  = 100  // QIDK 所有档位统一 100us
	rpcPollingMax      = 9999 // 头文件上限 QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME
)

// Config 是一组 HTP power config 参数。
type Config struct {
	Name string

	// PowerMode 为 0 表示不下发 DCVS_V3 配置项 (即完全不碰 DCVS/电压角)。
	PowerMode    int
	DcvsEnable   int
	SleepLatency int // us
	SleepDisable int // 1 = 不让 HTP 进入休眠
	BusMin       int
	BusTarget    int
	BusMax       int
	CoreMin      int
	CoreTarget   int
	CoreMax      int

	// Unset = 不下发该项; 0 = 显式下发 0 (关闭); >0 = 微秒。
	RpcControlLatency int
	RpcPollingTime    int
}

// HasDcvs 判断本档位是否需要下发 DCVS_V3 配置项。
func (c Config) HasDcvs() bool { return c.PowerMode != 0 }

// HasAny 判断本档位是否需要下发任何配置项。
// default 档且没有任何 -rpc-* 覆盖时为 false, 此时完全不调 setPowerConfig。
func (c Config) HasAny() bool {
	return c.HasDcvs() || c.RpcControlLatency != Unset || c.RpcPollingTime != Unset
}

// profileOrder 是展示档位的固定顺序, default 必须排在最前 (它是基线)。
var profileOrder = []string{
	"default",
	"burst",
	"burst_nosleep",
	"sustained_high_performance",
	"high_performance",
	"balanced",
	"power_saver",
}

// aliases 兼容 ORT 与 QIDK 两套命名。
var aliases = map[string]string{
	"sustained_high_perf": "sustained_high_performance",
	"high_perf":           "high_performance",
}

var profiles = map[string]Config{
	// default: 不下发任何配置, 保持 QNN 默认行为 (与不传 -htp-perf 等价)。
	// 两个 rpc 项必须显式写 Unset: Go 零值是 0, 而 0 在这里的意思是"显式下发 0"。
	"default": {Name: "default", RpcControlLatency: Unset, RpcPollingTime: Unset},

	// burst: 最低延迟档, 允许的最短休眠等待 + 最高电压角 + RPC 满 polling。
	"burst": {
		Name:              "burst",
		PowerMode:         POWER_MODE_PERFORMANCE,
		DcvsEnable:        0,
		SleepLatency:      sleepLatencyMin,
		SleepDisable:      0,
		BusMin:            VCORNER_MAX,
		BusTarget:         VCORNER_MAX,
		BusMax:            VCORNER_MAX,
		CoreMin:           VCORNER_MAX,
		CoreTarget:        VCORNER_MAX,
		CoreMax:           VCORNER_MAX,
		RpcControlLatency: rpcLatencyDefault,
		RpcPollingTime:    rpcPollingMax,
	},

	// burst_nosleep: 同 burst, 但直接禁止 HTP 休眠 (QIDK 的做法)。
	"burst_nosleep": {
		Name:              "burst_nosleep",
		PowerMode:         POWER_MODE_PERFORMANCE,
		DcvsEnable:        0,
		SleepLatency:      sleepLatencyMin,
		SleepDisable:      1,
		BusMin:            VCORNER_MAX,
		BusTarget:         VCORNER_MAX,
		BusMax:            VCORNER_MAX,
		CoreMin:           VCORNER_MAX,
		CoreTarget:        VCORNER_MAX,
		CoreMax:           VCORNER_MAX,
		RpcControlLatency: rpcLatencyDefault,
		RpcPollingTime:    rpcPollingMax,
	},

	// sustained_high_performance: 长局档, 比 burst 保守 (TURBO 而非 MAX, 100us 停等)。
	"sustained_high_performance": {
		Name:              "sustained_high_performance",
		PowerMode:         POWER_MODE_PERFORMANCE,
		DcvsEnable:        0,
		SleepLatency:      sleepLatencyLow,
		SleepDisable:      0,
		BusMin:            VCORNER_TURBO,
		BusTarget:         VCORNER_TURBO,
		BusMax:            VCORNER_TURBO,
		CoreMin:           VCORNER_TURBO,
		CoreTarget:        VCORNER_TURBO,
		CoreMax:           VCORNER_TURBO,
		RpcControlLatency: rpcLatencyDefault,
		RpcPollingTime:    rpcPollingMax,
	},

	// high_performance: 与 sustained_high_performance 参数完全相同 (ORT 与 QIDK 都如此),
	// 只是名字不同; 换小米 17 (v81) 时若要拉开差距再改。
	"high_performance": {
		Name:              "high_performance",
		PowerMode:         POWER_MODE_PERFORMANCE,
		DcvsEnable:        0,
		SleepLatency:      sleepLatencyLow,
		SleepDisable:      0,
		BusMin:            VCORNER_TURBO,
		BusTarget:         VCORNER_TURBO,
		BusMax:            VCORNER_TURBO,
		CoreMin:           VCORNER_TURBO,
		CoreTarget:        VCORNER_TURBO,
		CoreMax:           VCORNER_TURBO,
		RpcControlLatency: rpcLatencyDefault,
		RpcPollingTime:    rpcPollingMax,
	},

	// balanced: 开 DCVS 让固件自己调, 中等电压角, 不 polling。
	"balanced": {
		Name:              "balanced",
		PowerMode:         POWER_MODE_PERFORMANCE,
		DcvsEnable:        1,
		SleepLatency:      sleepLatencyMedium,
		SleepDisable:      0,
		BusMin:            VCORNER_NOM_PLUS,
		BusTarget:         VCORNER_NOM_PLUS,
		BusMax:            VCORNER_NOM_PLUS,
		CoreMin:           VCORNER_NOM_PLUS,
		CoreTarget:        VCORNER_NOM_PLUS,
		CoreMax:           VCORNER_NOM_PLUS,
		RpcControlLatency: rpcLatencyDefault,
		RpcPollingTime:    0,
	},

	// power_saver: 省电对照档, 只用来确认"投票确实生效了"(延迟应比 default 更差)。
	"power_saver": {
		Name:              "power_saver",
		PowerMode:         POWER_MODE_PERFORMANCE,
		DcvsEnable:        1,
		SleepLatency:      sleepLatencyMedium,
		SleepDisable:      0,
		BusMin:            VCORNER_SVS,
		BusTarget:         VCORNER_SVS,
		BusMax:            VCORNER_SVS,
		CoreMin:           VCORNER_SVS,
		CoreTarget:        VCORNER_SVS,
		CoreMax:           VCORNER_SVS,
		RpcControlLatency: rpcLatencyDefault,
		RpcPollingTime:    0,
	},
}

// Names 按固定顺序返回所有档位名 (含 default)。
func Names() []string {
	out := make([]string, len(profileOrder))
	copy(out, profileOrder)
	return out
}

// NamesString 返回逗号分隔的档位名, 用于日志与错误信息。
func NamesString() string { return strings.Join(profileOrder, ",") }

// Lookup 按名字取档位配置 (支持别名, 大小写不敏感)。
func Lookup(name string) (Config, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	if canonical, ok := aliases[key]; ok {
		key = canonical
	}
	cfg, ok := profiles[key]
	if !ok {
		return Config{}, fmt.Errorf("unknown htp-perf profile %q, available: %s", name, NamesString())
	}
	return cfg, nil
}

// IsDefault 判断是否为不投票的基线档。
func IsDefault(name string) bool {
	key := strings.ToLower(strings.TrimSpace(name))
	if canonical, ok := aliases[key]; ok {
		key = canonical
	}
	return key == "default"
}

// PlatformInfo 是 QnnDevice_getPlatformInfo 的 HTP 摘要 (QnnHtpDevice.h:127-145)。
//
// 放在本包而不是 qnn 包, 是为了让读它的代码不必面对 cgo; 它同时也是判断
// "要不要走重新 preparation (改 VTCM / 核数)" 的依据, 与档位表同属一套东西。
type PlatformInfo struct {
	Valid      bool
	Arch       int // HTP 架构 (68/69/73/75/79/81/85/89)
	SocModel   int
	VtcmMB     int // 该设备可用的 VTCM 总量 (MB)
	SignedPD   bool
	Dlbc       bool
	DevType    int // QNN_HTP_DEVICE_TYPE_ON_CHIP=0 / OFF_CHIP=1
	NumDevices int
	NumCores   int
}

// String 用于日志。
func (p PlatformInfo) String() string {
	if !p.Valid {
		return "unavailable"
	}
	return fmt.Sprintf("arch=%d soc=%d vtcm=%dMB signedPd=%t dlbc=%t devType=%d devices=%d cores=%d",
		p.Arch, p.SocModel, p.VtcmMB, p.SignedPD, p.Dlbc, p.DevType, p.NumDevices, p.NumCores)
}
