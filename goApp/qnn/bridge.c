#include "bridge.h"

#include <dlfcn.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "QNN/QnnBackend.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnTensor.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/System/QnnSystemContext.h"
#include "QNN/System/QnnSystemInterface.h"

#define QNN_CHECK(expr, msg)                     \
  do {                                           \
    Qnn_ErrorHandle_t _e_ = (expr);              \
    if (_e_ != QNN_SUCCESS) {                    \
      qnn_set_error("%s: error %d", msg, (int)_e_); \
      return -1;                                 \
    }                                            \
  } while (0)

typedef Qnn_ErrorHandle_t (*get_providers_fn)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*sys_get_providers_fn)(const QnnSystemInterface_t***, uint32_t*);

typedef struct {
  uint32_t id;
  char* name;
  uint32_t rank;
  uint32_t* dims;
  Qnn_DataType_t dataType;
  Qnn_TensorDataFormat_t dataFormat;
  Qnn_QuantizeParams_t quantize;
  uint64_t bytes;
} tensor_meta;

struct qnn_session {
  void* backend_lib;
  void* system_lib;
  QNN_INTERFACE_VER_TYPE iface;
  QNN_SYSTEM_INTERFACE_VER_TYPE sys_iface;
  Qnn_BackendHandle_t backend;
  Qnn_DeviceHandle_t device;
  Qnn_ContextHandle_t context;
  Qnn_GraphHandle_t graph;
  Qnn_LogHandle_t log;
  char* graph_name;
  tensor_meta* in;
  tensor_meta* out;
  uint32_t num_in;
  uint32_t num_out;
  int verbose;
};

static __thread char g_err[512];
static char g_env_buf[2][4096];
static int g_env_idx = 0;

static void qnn_set_error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
}

const char* qnn_last_error(void) { return g_err; }

static void qnn_log_cb(const char* fmt, QnnLog_Level_t level, uint64_t ts, va_list args) {
  (void)ts;
  fprintf(stderr, "[QNN:%d @%llu] %s\n", (int)level, (unsigned long long)ts, fmt);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
}

static size_t dtype_size(Qnn_DataType_t t) {
  switch (t) {
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      return 1;
  }
}

static void tensor_meta_free(tensor_meta* m) {
  if (!m) return;
  if (m->name) free(m->name);
  if (m->dims) free(m->dims);
  free(m);
}

// 从 Qnn_Tensor_t（binary info 中）深拷贝元数据
static tensor_meta* tensor_meta_copy(const Qnn_Tensor_t* t) {
  tensor_meta* m = calloc(1, sizeof(*m));
  if (!m) return NULL;
  if (!t) {
    qnn_set_error("tensor_meta_copy: null tensor");
    free(m);
    return NULL;
  }
  const char* name = NULL;
  const uint32_t* dims = NULL;
  if (t->version == QNN_TENSOR_VERSION_2) {
    m->id = t->v2.id;
    name = t->v2.name;
    dims = t->v2.dimensions;
    m->rank = t->v2.rank;
    m->dataType = t->v2.dataType;
    m->dataFormat = t->v2.dataFormat;
  } else {
    m->id = t->v1.id;
    name = t->v1.name;
    dims = t->v1.dimensions;
    m->rank = t->v1.rank;
    m->dataType = t->v1.dataType;
    m->dataFormat = t->v1.dataFormat;
  }
  m->quantize = t->version == QNN_TENSOR_VERSION_2 ? t->v2.quantizeParams : t->v1.quantizeParams;
  if (name) m->name = strdup(name);
  if (m->rank > 0 && dims) {
    m->dims = malloc(sizeof(uint32_t) * m->rank);
    if (!m->dims) goto fail;
    memcpy(m->dims, dims, sizeof(uint32_t) * m->rank);
  }
  uint64_t n = 1;
  for (uint32_t i = 0; i < m->rank; i++) n *= m->dims ? m->dims[i] : 0;
  m->bytes = n * dtype_size(m->dataType);
  return m;
fail:
  tensor_meta_free(m);
  return NULL;
}

static void qnn_log(qnn_session_t* s, const char* fmt, ...) {
  if (!s->verbose) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
}

qnn_session_t* qnn_create(const char* lib_dir, int arch, int verbose) {
  qnn_session_t* s = calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->verbose = verbose;

  if (lib_dir && *lib_dir) {
    // Android linker 默认命名空间忽略 LD_LIBRARY_PATH，这里仅作辅助。
    int idx = g_env_idx++ % 2;
    snprintf(g_env_buf[idx], sizeof(g_env_buf[idx]), "ADSP_LIBRARY_PATH=%s", lib_dir);
    putenv(g_env_buf[idx]);
    idx = g_env_idx++ % 2;
    snprintf(g_env_buf[idx], sizeof(g_env_buf[idx]), "LD_LIBRARY_PATH=%s", lib_dir);
    putenv(g_env_buf[idx]);
    qnn_log(s, "env: ADSP_LIBRARY_PATH=%s", lib_dir);
  }

  // 预加载 vendor 依赖链（fastrpc），使后续 dlopen 时依赖已存在于全局命名空间。
  // Android linker 不通过 LD_LIBRARY_PATH 搜索 vendor 库，必须逐个先加载。
  static const char* deps[] = {
      "libvmmem.so",
      "vendor.qti.hardware.dsp@1.0.so",
      "libcdsprpc.so",
  };
  for (size_t i = 0; i < sizeof(deps) / sizeof(deps[0]); i++) {
    void* h = dlopen(deps[i], RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
      qnn_set_error("dlopen %s: %s", deps[i], dlerror());
      goto fail;
    }
    qnn_log(s, "preloaded %s", deps[i]);
  }

  s->backend_lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
  if (!s->backend_lib) {
    qnn_set_error("dlopen libQnnHtp.so: %s", dlerror());
    goto fail;
  }
  s->system_lib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
  if (!s->system_lib) {
    qnn_set_error("dlopen libQnnSystem.so: %s", dlerror());
    goto fail;
  }

  get_providers_fn getp = (get_providers_fn)dlsym(s->backend_lib, "QnnInterface_getProviders");
  if (!getp) {
    qnn_set_error("dlsym QnnInterface_getProviders: %s", dlerror());
    goto fail;
  }
  const QnnInterface_t** providers = NULL;
  uint32_t num = 0;
  if (getp(&providers, &num) != QNN_SUCCESS || !providers || num == 0) {
    qnn_set_error("QnnInterface_getProviders failed (num=%u)", num);
    goto fail;
  }
  int found = 0;
  for (uint32_t i = 0; i < num; i++) {
    if (providers[i] &&
        providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
      s->iface = providers[i]->QNN_INTERFACE_VER_NAME;
      qnn_log(s, "interface v%u.%u provider=%s",
              providers[i]->apiVersion.coreApiVersion.major,
              providers[i]->apiVersion.coreApiVersion.minor,
              providers[i]->providerName ? providers[i]->providerName : "?");
      found = 1;
      break;
    }
  }
  if (!found) {
    qnn_set_error("no compatible QNN interface (need v%u.%u)", QNN_API_VERSION_MAJOR,
                  QNN_API_VERSION_MINOR);
    goto fail;
  }

  sys_get_providers_fn getsys =
      (sys_get_providers_fn)dlsym(s->system_lib, "QnnSystemInterface_getProviders");
  if (!getsys) {
    qnn_set_error("dlsym QnnSystemInterface_getProviders: %s", dlerror());
    goto fail;
  }
  const QnnSystemInterface_t** sysproviders = NULL;
  uint32_t sysnum = 0;
  if (getsys(&sysproviders, &sysnum) != QNN_SUCCESS || !sysproviders || sysnum == 0) {
    qnn_set_error("QnnSystemInterface_getProviders failed (num=%u)", sysnum);
    goto fail;
  }
  int sysfound = 0;
  for (uint32_t i = 0; i < sysnum; i++) {
    if (sysproviders[i] &&
        sysproviders[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        sysproviders[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      s->sys_iface = sysproviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      sysfound = 1;
      break;
    }
  }
  if (!sysfound) {
    qnn_set_error("no compatible QNN System interface");
    goto fail;
  }

  if (s->iface.logCreate) {
    if (s->iface.logCreate(qnn_log_cb, QNN_LOG_LEVEL_WARN, &s->log) != QNN_SUCCESS) {
      s->log = NULL;
    }
  }

  if (s->iface.backendCreate(s->log, NULL, &s->backend) != QNN_SUCCESS) {
    qnn_set_error("QnnBackend_create failed");
    goto fail;
  }

  QnnHtpDevice_CustomConfig_t htp;
  memset(&htp, 0, sizeof(htp));
  QnnDevice_Config_t dev_cfg;
  memset(&dev_cfg, 0, sizeof(dev_cfg));
  const QnnDevice_Config_t* dev_cfgs[] = {NULL, NULL};
  if (arch > 0) {
    htp.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
    htp.arch.deviceId = 0;
    htp.arch.arch = (QnnHtpDevice_Arch_t)arch;
    dev_cfg.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    dev_cfg.customConfig = &htp;
    dev_cfgs[0] = &dev_cfg;
  }
  if (s->iface.deviceCreate(s->log, dev_cfgs, &s->device) != QNN_SUCCESS) {
    qnn_set_error("QnnDevice_create failed (arch=%d)", arch);
    goto fail;
  }
  return s;

fail:
  qnn_destroy(s);
  return NULL;
}

int qnn_load_binary(qnn_session_t* s, const void* bin, uint64_t size) {
  if (!s || !bin || size == 0) {
    qnn_set_error("invalid args to qnn_load_binary");
    return -1;
  }
  QnnSystemContext_Handle_t sys_ctx = NULL;
  const QnnSystemContext_BinaryInfo_t* binfo = NULL;
  Qnn_ContextBinarySize_t binfo_size = 0;
  QNN_CHECK(s->sys_iface.systemContextCreate(&sys_ctx), "systemContextCreate");
  QNN_CHECK(s->sys_iface.systemContextGetBinaryInfo(sys_ctx, (void*)bin, size, &binfo, &binfo_size),
            "systemContextGetBinaryInfo");
  const QnnSystemContext_GraphInfo_t* graphs = NULL;
  uint32_t num_graphs = 0;
  if (binfo) {
    if (binfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
      graphs = binfo->contextBinaryInfoV3.graphs;
      num_graphs = binfo->contextBinaryInfoV3.numGraphs;
    } else if (binfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
      graphs = binfo->contextBinaryInfoV2.graphs;
      num_graphs = binfo->contextBinaryInfoV2.numGraphs;
    } else {
      graphs = binfo->contextBinaryInfoV1.graphs;
      num_graphs = binfo->contextBinaryInfoV1.numGraphs;
    }
  }
  if (!graphs || num_graphs == 0) {
    qnn_set_error("binary info has no graphs");
    s->sys_iface.systemContextFree(sys_ctx);
    return -1;
  }

  const QnnSystemContext_GraphInfo_t* gi = &graphs[0];
  const Qnn_Tensor_t* gins = NULL;
  const Qnn_Tensor_t* gouts = NULL;
  uint32_t nin = 0, nout = 0;
  const char* gname = NULL;
  if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
    gname = gi->graphInfoV3.graphName;
    gins = gi->graphInfoV3.graphInputs;
    gouts = gi->graphInfoV3.graphOutputs;
    nin = gi->graphInfoV3.numGraphInputs;
    nout = gi->graphInfoV3.numGraphOutputs;
  } else if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
    gname = gi->graphInfoV2.graphName;
    gins = gi->graphInfoV2.graphInputs;
    gouts = gi->graphInfoV2.graphOutputs;
    nin = gi->graphInfoV2.numGraphInputs;
    nout = gi->graphInfoV2.numGraphOutputs;
  } else {
    gname = gi->graphInfoV1.graphName;
    gins = gi->graphInfoV1.graphInputs;
    gouts = gi->graphInfoV1.graphOutputs;
    nin = gi->graphInfoV1.numGraphInputs;
    nout = gi->graphInfoV1.numGraphOutputs;
  }
  if (!gname || nin == 0 || nout == 0 || !gins || !gouts) {
    qnn_set_error("graph info incomplete");
    s->sys_iface.systemContextFree(sys_ctx);
    return -1;
  }

  // 深拷贝元数据（binary info 内存随后释放）
  s->graph_name = strdup(gname);
  s->in = calloc(nin, sizeof(tensor_meta));
  s->out = calloc(nout, sizeof(tensor_meta));
  s->num_in = nin;
  s->num_out = nout;
  for (uint32_t i = 0; i < nin; i++) {
    tensor_meta* cp = tensor_meta_copy(&gins[i]);
    if (!cp || !cp->name) {
      qnn_set_error("failed to copy input metadata %u", i);
      if (cp) tensor_meta_free(cp);
      s->sys_iface.systemContextFree(sys_ctx);
      return -1;
    }
    s->in[i] = *cp;
    free(cp);
  }
  for (uint32_t i = 0; i < nout; i++) {
    tensor_meta* cp = tensor_meta_copy(&gouts[i]);
    if (!cp || !cp->name) {
      qnn_set_error("failed to copy output metadata %u", i);
      if (cp) tensor_meta_free(cp);
      s->sys_iface.systemContextFree(sys_ctx);
      return -1;
    }
    s->out[i] = *cp;
    free(cp);
  }
  s->sys_iface.systemContextFree(sys_ctx);
  sys_ctx = NULL;

  qnn_log(s, "graph=%s in=%u out=%u", s->graph_name, nin, nout);
  for (uint32_t i = 0; i < nin; i++) {
    qnn_log(s, "  in[%u] %s rank=%u bytes=%llu", i, s->in[i].name, s->in[i].rank,
            (unsigned long long)s->in[i].bytes);
    if (s->in[i].quantize.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      qnn_log(s, "       quant=SCALE_OFFSET scale=%f offset=%d",
              s->in[i].quantize.scaleOffsetEncoding.scale,
              s->in[i].quantize.scaleOffsetEncoding.offset);
    }
  }
  for (uint32_t i = 0; i < nout; i++) {
    qnn_log(s, "  out[%u] %s rank=%u bytes=%llu", i, s->out[i].name, s->out[i].rank,
            (unsigned long long)s->out[i].bytes);
    if (s->out[i].quantize.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      qnn_log(s, "       quant=SCALE_OFFSET scale=%f offset=%d",
              s->out[i].quantize.scaleOffsetEncoding.scale,
              s->out[i].quantize.scaleOffsetEncoding.offset);
    }
  }

  QNN_CHECK(s->iface.contextCreateFromBinary(s->backend, s->device, NULL, bin, size, &s->context,
                                             NULL),
            "contextCreateFromBinary");
  QNN_CHECK(s->iface.graphRetrieve(s->context, s->graph_name, &s->graph), "graphRetrieve");
  return 0;
}

uint32_t qnn_io_info(const qnn_session_t* s,
                     char* in_name, uint32_t* in_dims, uint32_t* in_rank,
                     char* out_name, uint32_t* out_dims, uint32_t* out_rank,
                     uint32_t* out_count,
                     int* in_dtype, float* in_scale, int32_t* in_offset,
                     int* out_dtype, float* out_scale, int32_t* out_offset) {
  if (!s || !s->in || !s->out) return 0;
  if (in_name && s->in[0].name) strcpy(in_name, s->in[0].name);
  if (in_dims && in_rank) {
    *in_rank = s->in[0].rank;
    memcpy(in_dims, s->in[0].dims, sizeof(uint32_t) * s->in[0].rank);
  }
  if (out_name && s->out[0].name) strcpy(out_name, s->out[0].name);
  if (out_dims && out_rank) {
    *out_rank = s->out[0].rank;
    memcpy(out_dims, s->out[0].dims, sizeof(uint32_t) * s->out[0].rank);
  }
  if (out_count) *out_count = s->num_out;
  if (in_dtype) *in_dtype = (int)s->in[0].dataType;
  if (in_scale && s->in[0].quantize.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    *in_scale = s->in[0].quantize.scaleOffsetEncoding.scale;
  if (in_offset && s->in[0].quantize.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    *in_offset = s->in[0].quantize.scaleOffsetEncoding.offset;
  if (out_dtype) *out_dtype = (int)s->out[0].dataType;
  if (out_scale && s->out[0].quantize.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    *out_scale = s->out[0].quantize.scaleOffsetEncoding.scale;
  if (out_offset && s->out[0].quantize.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    *out_offset = s->out[0].quantize.scaleOffsetEncoding.offset;
  return s->num_in;
}

int qnn_execute(qnn_session_t* s, const void* input, void* output, double* ms_out) {
  if (!s || !s->graph || !s->in || !s->out || !input || !output || s->num_in != 1 ||
      s->num_out != 1) {
    qnn_set_error("invalid args to qnn_execute");
    return -1;
  }
  Qnn_Tensor_t* ins = calloc(s->num_in, sizeof(Qnn_Tensor_t));
  Qnn_Tensor_t* outs = calloc(s->num_out, sizeof(Qnn_Tensor_t));
  if (!ins || !outs) {
    qnn_set_error("OOM in qnn_execute");
    free(ins);
    free(outs);
    return -1;
  }
  Qnn_Tensor_t* in = &ins[0];
  memset(in, 0, sizeof(*in));
  in->version = QNN_TENSOR_VERSION_2;
  Qnn_TensorV2_t in_v2 = QNN_TENSOR_V2_INIT;
  in->v2 = in_v2;
  in->v2.id = s->in[0].id;
  in->v2.name = s->in[0].name;
  in->v2.type = QNN_TENSOR_TYPE_APP_READ;
  in->v2.dataFormat = s->in[0].dataFormat;
  in->v2.dataType = s->in[0].dataType;
  in->v2.rank = s->in[0].rank;
  in->v2.dimensions = s->in[0].dims;
  in->v2.quantizeParams = s->in[0].quantize;
  in->v2.memType = QNN_TENSORMEMTYPE_RAW;
  in->v2.clientBuf.data = (void*)input;
  in->v2.clientBuf.dataSize = s->in[0].bytes;

  Qnn_Tensor_t* out = &outs[0];
  memset(out, 0, sizeof(*out));
  out->version = QNN_TENSOR_VERSION_2;
  Qnn_TensorV2_t out_v2 = QNN_TENSOR_V2_INIT;
  out->v2 = out_v2;
  out->v2.id = s->out[0].id;
  out->v2.name = s->out[0].name;
  out->v2.type = QNN_TENSOR_TYPE_APP_WRITE;
  out->v2.dataFormat = s->out[0].dataFormat;
  out->v2.dataType = s->out[0].dataType;
  out->v2.rank = s->out[0].rank;
  out->v2.dimensions = s->out[0].dims;
  out->v2.quantizeParams = s->out[0].quantize;
  out->v2.memType = QNN_TENSORMEMTYPE_RAW;
  out->v2.clientBuf.data = output;
  out->v2.clientBuf.dataSize = s->out[0].bytes;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  Qnn_ErrorHandle_t rc = s->iface.graphExecute(s->graph, ins, s->num_in, outs, s->num_out, NULL,
                                               NULL);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  free(ins);
  free(outs);
  if (rc != QNN_SUCCESS) {
    qnn_set_error("graphExecute failed: error %d", (int)rc);
    return -1;
  }
  if (ms_out) {
    *ms_out = ((double)(t1.tv_sec - t0.tv_sec) * 1000.0) +
              ((double)(t1.tv_nsec - t0.tv_nsec) / 1000000.0);
  }
  return 0;
}

void qnn_destroy(qnn_session_t* s) {
  if (!s) return;
  // 不调用 contextFree/deviceFree/backendFree：Termux 环境下
  // contextFree 会触发 SIGABRT（QNN 内部线程清理问题）。
  // 句柄随进程退出由 OS 回收；常驻进程结束时直接 os.Exit 即可。
  if (s->graph_name) free(s->graph_name);
  for (uint32_t i = 0; i < s->num_in; i++) tensor_meta_free(&s->in[i]);
  for (uint32_t i = 0; i < s->num_out; i++) tensor_meta_free(&s->out[i]);
  if (s->in) free(s->in);
  if (s->out) free(s->out);
  // 不 dlclose，原因同上。
  free(s);
}
