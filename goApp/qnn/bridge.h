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

// 销毁会话并释放所有资源。
void qnn_destroy(qnn_session_t* s);

// 返回最后一次错误的描述（线程局部）。
const char* qnn_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // QNN_BRIDGE_H
