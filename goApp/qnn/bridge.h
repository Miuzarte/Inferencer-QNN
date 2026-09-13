#ifndef QNN_BRIDGE_H
#define QNN_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qnn_session qnn_session_t;

// 创建 QNN HTP 会话。
// lib_dir: 包含 libQnnHtp.so/libcdsprpc.so 等库的目录（会设置 ADSP_LIBRARY_PATH/LD_LIBRARY_PATH）
// arch: HTP 架构（73/75/81 等），0 表示不指定
// verbose: 1 输出详细日志
qnn_session_t* qnn_create(const char* lib_dir, int arch, int verbose);

// 加载 context binary（从官方 ONNX 提取的 ep_cache_context）。
// 返回 0 成功，非 0 失败。
int qnn_load_binary(qnn_session_t* s, const void* bin, uint64_t size);

// 输入/输出信息查询。
// 返回输入数量；out_count 返回输出数量。
// in_name/out_name 至少 QNN_MAX_NAME(64) 字节。
uint32_t qnn_io_info(const qnn_session_t* s,
                     char* in_name, uint32_t* in_dims, uint32_t* in_rank,
                     char* out_name, uint32_t* out_dims, uint32_t* out_rank,
                     uint32_t* out_count,
                     int* in_dtype, float* in_scale, int32_t* in_offset,
                     int* out_dtype, float* out_scale, int32_t* out_offset);

// 执行推理。input/output 由调用方分配（大小可通过 qnn_io_info 的 dims 计算）。
// ms_out 返回本机推理耗时（毫秒，可选）。
// 返回 0 成功，非 0 失败。
int qnn_execute(qnn_session_t* s, const void* input, void* output, double* ms_out);

// HTP 性能档位参数 (字段值直接对应 QnnHtpPerfInfrastructure_* 枚举)。
// power_mode 为 0 表示不下发 DCVS_V3 配置项。
// rpc_control_latency / rpc_polling_time 为负表示不下发该项 (0 = 显式下发 0)。
typedef struct {
  int power_mode;
  int dcvs_enable;
  int sleep_latency;
  int sleep_disable;
  int bus_vc_min;
  int bus_vc_target;
  int bus_vc_max;
  int core_vc_min;
  int core_vc_target;
  int core_vc_max;
  int rpc_control_latency;
  int rpc_polling_time;
} qnn_power_cfg;

// 请求 HTP 性能基础设施 (QnnDevice_getInfrastructure) 并创建 power config id。
// 幂等: 已初始化时直接返回 0。
// 返回 0 成功; -2 = 设备/后端不支持性能基础设施; -1 = 其它错误 (见 qnn_last_error)。
int qnn_perf_init(qnn_session_t* s);

// 查询性能基础设施状态。返回 1 = 可用, 0 = 不可用。
// infra_type 返回 QnnHtpDevice_InfrastructureType_t, power_config_id 返回配置 id。
int qnn_perf_status(const qnn_session_t* s, int* infra_type, unsigned* power_config_id);

// 下发性能档位 (可反复调用换档)。cfg 为 NULL 时不做任何事并返回 0。
// 整组被拒时自动退化为"只下发 DCVS_V3"重试, 两次的返回码都写进 qnn_last_error。
// 返回 0 成功; 1 = 已应用但降级 (部分配置项被拒); -2 = 未初始化或不支持; -1 = 失败。
int qnn_perf_apply(qnn_session_t* s, const qnn_power_cfg* cfg);

// 平台信息 (QnnDevice_getPlatformInfo)。valid=0 表示查询失败, 其余字段无意义。
typedef struct {
  int valid;
  int arch;
  int soc_model;
  int vtcm_mb;
  int signed_pd;
  int dlbc;
  int dev_type;
  int num_devices;
  int num_cores;
} qnn_plat_info;

// 查询平台信息。返回 0 成功 (不论 valid), 非 0 表示 API 本身不可用。
int qnn_platform_info(qnn_session_t* s, qnn_plat_info* out);

// 销毁会话并释放所有资源。
void qnn_destroy(qnn_session_t* s);

// 返回最后一次错误的描述（线程局部）。
const char* qnn_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // QNN_BRIDGE_H
