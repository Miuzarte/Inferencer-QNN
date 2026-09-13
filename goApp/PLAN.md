# Termux + Go 运行 YOLO26n（QNN HTP）方案

> 2026-08-02 可行性验证完成。目标：在 Termux 里用 Go 跑官方
> `yolo26n_v73_qnn.onnx`（HTP v73 / 骁龙 8 Gen 2），并接入 GoCVStreamer
> 远程推理链路，为最终"开销最小化"形态打基础。

## 0. 结论

**可行**，但有一个前提条件和一个路线选择：

1. **前提**：Android linker 默认命名空间不允许 dlopen `/vendor/lib64`（即使 root
   也一样），所以必须把 QNN 依赖的 vendor 库复制到 Termux 自己的目录
   （`$PREFIX/lib`），并通过 `LD_LIBRARY_PATH` / `ADSP_LIBRARY_PATH` 指向那里。
   这个条件今天已在设备上实测通过。
2. **路线**：Android 版 `libonnxruntime.so` 的公开 C API 只导出
   `OrtSessionOptionsAppendExecutionProvider_CPU`，Go 无法通过 ORT C API 启用
   QNN EP（readelf 已验证）。因此不绕 ORT，**直接用 QNN SDK 的 C API**
   （`QnnInterface_getProviders`）加载官方 ONNX 里内嵌的 EPContext 二进制。

## 1. 已验证事实（2026-08-02，小米 13 / HTP v73）

### 1.1 androidApp 成功链路（前情）

- 根因：App 必须声明 `<uses-native-library android:name="libcdsprpc.so"/>`，
  flutterApp 是靠 LiteRT 传递声明才正常；详见 `androidApp/PLAN.md`。
- ORT 1.26.0 + qnn-runtime 2.46.0 + `ADSP_LIBRARY_PATH` 指向自带 Skel 的目录，
  session 创建成功，`outputDims=[1,84,8400]`。

### 1.2 Termux 环境

- Termux 0.118.0，`com.termux`（uid u0_a391），clang 21.1.8 / gcc / pkg 已装；
  **Go 未安装**（`usr/bin/go` 及常见路径均无，待 `pkg install golang`）。
- sshd 运行在 192.168.1.103:8022，SSH 会话内可连续执行命令；
  长任务用 `tmux`/`nohup` 保持（Termux 进程被系统回收时需
  `termux-wake-lock`）。

### 1.3 链接命名空间实测（关键）

设备 `linker.config.pb` 的 default namespace：

```
ld_library_paths=""
default_library_paths="/system/lib64:/system_ext/lib64"
permitted_paths=...（不含 /vendor/lib64）
```

实测（Termux 用户，非 root）：

```
dlopen("libcdsprpc.so")              = OK   # 复制到 $PREFIX/lib 后
dlopen("/vendor/lib64/libcdsprpc.so") = 拒绝 # 绝对路径仍被命名空间挡住
dlopen("libQnnHtp.so")               = OK   # 依赖链（cdsprpc/dsp-hal/vmmem）全部可解析
```

即：**不需要 root、不需要改 sepolicy**，普通 Termux 进程即可加载 QNN 栈，
只要库文件落在 `/data` 下（`$PREFIX/lib`）并设置 `LD_LIBRARY_PATH`。

### 1.4 已复制到 `$PREFIX/lib` 的库

| 库 | 来源 | 作用 |
|---|---|---|
| `libcdsprpc.so` | `/vendor/lib64` | fastrpc 客户端（QNN 必需） |
| `vendor.qti.hardware.dsp@1.0.so` | `/vendor/lib64` | DSP HAL 客户端 |
| `libvmmem.so` | `/vendor/lib64` | 内存分配 |
| `libQnnHtp.so` / `libQnnSystem.so` / `libQnnHtpPrepare.so` | qnn-runtime AAR（androidApp APK） | HTP 后端与 system 库 |
| `libQnnHtpV73Skel.so` / `libQnnHtpV73Stub.so` | qnn-runtime AAR | HTP v73 DSP 侧库 |

`libhidlbase/libutils/libcutils/libhardware/libdmabufheap/liblog/libc++`
在 `/system/lib64`，默认命名空间可访问，无需复制。

## 2. 技术路线

### 路线 A：ORT C API —— 不可行

- readelf：Android `libonnxruntime.so`（1.26.0）动态符号里只有
  `OrtSessionOptionsAppendExecutionProvider_CPU`，没有 QNN 的 append 函数，
  也没有通用 `OrtSessionOptionsAppendExecutionProvider`。
- `onnxruntime_go` 等 Go binding 走同一 C API，同样加不了 QNN EP。

### 路线 B：直连 QNN SDK C API —— 推荐

```text
libQnnHtp.so 导出 QnnInterface_getProviders
libQnnSystem.so 导出 QnnSystemInterface_getProviders
```

执行流程：

1. 从官方 `yolo26n_v73_qnn.onnx` 提取 EPContext 二进制：
   - 模型内有一个 `EPContext` 节点，`embed_mode=1`，属性 `ep_cache_context`
     就是 QNN context binary（ORT 内部 `LoadCachedQnnContextFromBuffer` 用的就是它）。
   - Go 用 protobuf 解析 ONNX 的 `ModelProto/GraphProto/NodeProto/AttributeProto`
     即可（模型小，解析成本可忽略）；也可先在 PC 端一次性提取出
     `ctx_v73.bin` 随仓库分发。
2. cgo `dlopen("libQnnHtp.so")` → `QnnInterface_getProviders` → 取 HTP provider
   API vtable（Backend/Device/Context/Graph/Tensor/Execution/Log）。
3. `QnnBackend_initialize`（backend 配置带 libpath）→
   `QnnDevice_create`（HTP：burst、arch v73；Termux 里读 `/sys/devices/soc0/sku`
   没权限问题可忽略，root 可读）→ `QnnContext_createFromBinary` 加载 context
   binary → 取 graph 句柄。
4. 输入输出：CPU tensor，float32 `[1,640,640,3]`（NHWC，与官方导出一致）；
   `QnnGraph_execute`。
5. 后处理在 Go 里做：argmax + 阈值 + NMS + 归一化坐标（逻辑可从
   `androidApp/QnnYoloModel.kt` 直接移植）。
6. 预处理在 Go 里做：JPEG decode（`image/jpeg`）→ letterbox 640 → RGB float。

优点：无 ORT 依赖、无 JVM，二进制小、延迟可控，最贴近"开销最小化"目标。

风险：需要 QNN SDK 头文件（`QnnInterface.h`、`QnnHtp.h`、`QnnContext.h`、
`QnnGraph.h`、`QnnTensor.h`、`QnnExecution.h`、`QnnBackend.h`、`QnnDevice.h`、
`QnnSystem.h`），Qualcomm 官网下载要账号；可以从开源镜像（sherpa-onnx /
executorch / 其他 QNN 项目）取对应 v2.46 的头文件，或按接口版本手写结构体。

### 路线 C：root 域运行 —— 不需要

实测 root 同样受命名空间限制（`/vendor/lib64` 不在 permitted_paths），
仍要复制库；而普通 Termux 用户已经能加载。root 只对读 `sku` 有帮助，
但 QNN 设备创建在 Termux 下不需要它（androidApp 里 sku EACCES 也不是根因）。
因此默认走非 root，保持 SELinux Enforcing。

## 3. 分阶段实施计划

### P0 环境与资产

- [x] `pkg install golang`（Go 1.26.5 android/arm64）。
- [x] 拉取 QNN SDK 2.46 头文件到 `goApp/qnn-headers/`（来源：官方软件中心
      `https://softwarecenter.qualcomm.com/api/download/software/sdks/
      Qualcomm_AI_Runtime_Community/All/2.46.0.260424/v2.46.0.260424.zip`，
      接口版本 QNN_API_VERSION 2.35）。
- [x] 下载官方模型 `yolo26n_v73_qnn.onnx`（可复用 androidApp/flutterApp 缓存，
     或直接下载 release v0.6.6）。
- [x] 写 `cmd/extract_ctx`：解析 ONNX 提取 `ep_cache_context` → `ctx_v73.bin`
      （md5 `2842e800ce918f0d985ece0a22fade61`，与 onnx 库提取一致）。
- [x] 下载 `yolo26n_v81_qnn.onnx`（release v0.6.6）提取 `ctx_v81.bin`
      （md5 `2507f0d90e3859e9b48e6412a1a50e3c`）。onnx 不入库，models/ 只
      保留 ctx binary。
- [x] 固化库部署脚本 `scripts/setup_libs.sh`：从 `/vendor/lib64` 和 androidApp
      APK 复制依赖到 `$PREFIX/lib`（当前已手动完成一次）。

### P1 单张图片推理

- [x] `qnn/cgo` 封装：provider 加载、backend/device/context/graph 初始化、
      execute、释放。
- [x] `cmd/infer`：输入一张 JPEG，输出 detections（坐标归一化 + 类别 + 分数）。
- [x] 对照验证：Ultralytics bus.jpg，5 个检测框（4 person + 1 bus）分数/位置合理，
      （允许浮点级微小差异）。

**P1 完成情况（2026-08-02）**：

- [x] Termux 非 root、SELinux Enforcing 下 QNN HTP 直连推理跑通：
  `QnnInterface_getProviders` v2.35 → backend/device(v73)/context/graph 全部成功；
- [x] 关键坑 1（vendor 库）：Android linker 忽略 `LD_LIBRARY_PATH`，必须在
  `dlopen("libQnnHtp.so")` 前按依赖顺序预加载
  `libvmmem.so → vendor.qti.hardware.dsp@1.0.so → libcdsprpc.so`；
- [x] 关键坑 2（量化 I/O）：官方模型的 QNN graph I/O 是 **quint16**
  （`QNN_DATATYPE_UFIXED_POINT_16=1046`），输入
  `scale=1/65536 offset=0`，输出 `scale≈0.010455 offset=0`；
  输入 float(0..1)→`round(v/scale)-offset` 量化，输出
  `(q+offset)*scale` 反量化；
- [x] 关键坑 3（tensor id）：execute 必须带 binaryInfo 里 backend 分配的
  **tensor id**（重建 v2 tensor 时 id=0 → `graphExecute` 6004）；
- [x] 关键坑 4（坐标尺度）：直连 QNN 输出坐标是 **640 尺度像素**（letterbox 图
  坐标），归一化用 `(v-pad)/scale/srcW`，**不要**像 ORT 路径那样先 `*640`；
- [x] 关键坑 5（销毁）：Termux 下 `QnnContext_free` 触发 SIGABRT（内部线程清理），
  当前策略是销毁时跳过 QNN free 调用，进程退出由 OS 回收；
- [x] 性能：bus.jpg 单帧推理 **~19-23ms**，与 androidApp（20+ms）持平；
- [x] 二进制：`go build ./cmd/infer` 单文件 ~4.5MB，无 JVM/ORT 依赖。

复现命令（Termux）：

```bash
export ADSP_LIBRARY_PATH=$PREFIX/lib
./infer -image bus.jpg -ctx models/ctx_v73.bin -out boxes.png
```

### P2 接入 GoCVStreamer

- [x] WebSocket 客户端（协议与 flutterApp/androidApp 一致：
      `[4B frame_id LE][JPEG]` → 回 `{"frame_id":N,"detections":[...],"inference_ms":M}`）。
- [x] 单线程串行推理 + 跳帧防堆积（WS 读→推理→回传串行，天然背压跳帧）。
- [x] 统计：推理延迟、帧率（每 5s 打印）；PC 端 wstest 收到
      4 个 person 检测（class=0 过滤）+ inference_ms 16-21ms。

**P2 完成情况（2026-08-02）**：

- [x] `cmd/streamer` 编译通过（Termux，Go 1.26.5 + github.com/coder/websocket，
  `GOPROXY=https://goproxy.cn`）；
- [x] 与临时 WS 服务端（模拟 GoCVStreamer 协议）端到端验证：
  PC 发 `[4B id][JPEG]` → Termux 推理 → PC 收到
  `{"frame_id":N,"detections":[4x person],...}`；
- [x] 固定只回传 person（class=0，`-class` 可调）；
- [x] 断线自动重连（3s 间隔）；
- [x] 注意：真实 GoCVStreamer 发的是 640×640 流帧；streamer 端不写死
  640，用 Preprocess 返回的实际 scale/pad/src 尺寸，兼容任意输入帧。
- [ ] PC 端青色框 + assist 接入待用户在场验证（当前 PC 无交互桌面，
  DXGI 捕获不可用；用 wstest 验证了协议链路）。

### P3 打磨

- [x] v73/v81 模型切换：`ctx_v73.bin` / `ctx_v81.bin` 均已提取；运行时按
      `-arch` 选择对应 Skel/Stub（v81 需 8 Elite 设备上的 `libQnnHtpV81*`）。
- [ ] `termux-wake-lock` 保活、崩溃恢复、日志分级。
- [ ] 性能优化：复用输入/输出 buffer、免拷贝、必要时 cgo 里做
      YUV→RGB 或直接喂 Camera2（可选，远程源为主）。

## 4. 验证清单（最终）

- [ ] SSH 会话内 `go run ./cmd/infer -image test.jpg` 输出 boxes；
- [ ] 同帧对比 androidApp：类别/坐标/分数一致；
- [ ] GoCVStreamer 回传青色框 + 统计（PC 端）；
- [ ] 延迟记录：目标单帧推理 ≤ 20ms（与 androidApp 同模型对比）；
- [ ] 验证完立即停推理（NPU 温控，别长时间跑）。

## 5. 风险与备选

- **QNN 头文件获取**：最大外部依赖。备选：反编译/对照 executorch 的
  `QnnInterface` 定义；或退回"Go 包一层 C 桥 + 复用 ORT JNI"（不推荐，重）。
- **`QnnContext_createFromBinary`**：QNN 2.20+ 支持；若 2.46 接口里没有，
  改用 `QnnSystemContext` 解析 binary → `QnnContext_create`（老流程）。
- **EPContext 提取失败**：备选在 PC 用 Python `onnx` 库提取一次，
  生成 `ctx_v73.bin` 提交到仓库（3MB 内可接受）。
- **Termux 进程被杀**：前台 `termux-wake-lock`；SSH 会话断开用 tmux 保持。
- **温控**：与 androidApp 相同，验证后立即 stop。

## 6. P4: HTP 执行延迟调优 (2026-09-13 完成)

### 结论

`execute` 从 **14.8ms 降到 5.5ms (-63%)**: 之前的 14.8ms **不是算子本身的开销**,
而是"从来没有给 HTP 投过性能票", DSP 一直跑在 DCVS 默认(低频)档上。加上
`QnnDevice_getInfrastructure` → `createPowerConfigId` → `setPowerConfig`
之后就下来了, 手机端入口是 `cmd/streamer -htp-perf <档位>`。

### 为什么是它

当时的账本 (小米13, 30fps 源, 640x640 JPEG ~50KB, 串行 + `-affinity 7`):

```
latency 29.1ms = net 8.0 (链路+PC, 已到底) + cpu 6.3 (手机 CPU, 已压过) + execute 14.8
```

`qnn/bridge.c` 当时只设了 `QNN_HTP_DEVICE_CONFIG_OPTION_ARCH`,
性能档 / DVFS / fastrpc 延迟参数一个都没下发。

### 已核对的事实 (qnn-headers = QNN 2.46, API 2.35)

**调用链** (`QnnDevice.h:354` 声明, 实现在 `QnnInterface.h:565` 的 vtable):

```
deviceCreate -> iface.deviceGetInfrastructure(&infra)
             -> (QnnHtpDevice_Infrastructure_t*)infra->perfInfra.createPowerConfigId(0, 0, &id)
             -> perfInfra.setPowerConfig(id, NULL 结尾的配置数组)
```

- 头文件签名写作 `const QnnDevice_Infrastructure_t*` (= `_t**`), 是 SDK 的 const
  写法怪癖; 厂商实现一律 `QnnDevice_Infrastructure_t infra = NULL; iface.deviceGetInfrastructure(&infra);`
- `dcvsV3Config.contextId` 填 `createPowerConfigId` 返回的 id (不是 0) — QIDK /
  onnxruntime-qnn / ai-engine-direct-helper 三家实现完全一致
- `setPowerConfig` 可反复调用换档 (QIDK 就是这么在运行时切 profile 的)
- 投票是**进程级且不可撤销**: 一旦投过票, "不下发" 的 default 就回不去
- `RPC_POLLING_TIME` 仅 V69+, 作用于整个进程, 上限 9999us (`QnnHtpPerfInfrastructure.h:381`)
- 电压角 MAX=0xA0 / TURBO=0x80 / NOM_PLUS=0x70 / SVS=0x40; sleepLatency 40(最短)/100/1000/2000us

**厂商参考实现** (语义一致):
- `B:\Git\qidk\...\YoloNas\app\src\main\cpp\src\inference.cpp:69-238` (Qualcomm 自己的档位表)
- `B:\Git\onnxruntime-qnn\...\qnn_htp_power_config_manager.cc` + `qnn_def.h:162-170`
- `B:\Git\ai-engine-direct-helper\src\QnnInferenceEngine.cpp:57-181, 2120-2152`

**明确排掉、不要浪费时间的方向**:
- Context 的 `INIT_ACCELERATION` 只加速**反序列化/加载** (`QnnHtpContext.h:175-180`),
  **与每帧延迟无关**; `USE_EXTENDED_UDMA` 仅 v81+; `SKIP_VALIDATION_ON_BINARY_SECTION`
  只对 LoRA 生效 → 原计划里的"试 context 选项"作废
- Device 的 `SOC`/`SECUREPD`/`SIGNEDPD` (`QnnHtpDevice.h:71-77`) 不是延迟旋钮
- `DDR_PERF_MODE` 要 V81 + LLM + RPC polling 同时满足; `HMX_V2`/`CENG`/
  `ADAPTIVE_POLLING_TIME` 在 v73 上无意义 → 留给换小米 17 (v81) 时再开

### 工具与测量纪律

- `cmd/htpbench`: 单进程内**轮询**多档位 (default -> burst -> ... -> default -> ...),
  按 `-interval 33ms` 复现真实占空比。背靠背测会让 DSP 一直醒着, 掩盖休眠唤醒成本。
  纯 default 跑法 (`-profiles default -rounds N`) 能拿到完整时间序列看热衰减。
- **只看平均值**: 单帧噪声 ±3ms
- **成对交替**: 同一配置跨时段能差 3ms (手机 CPU 段 5.7~10.6ms, worker 核 0.86~1.6GHz)
- 口径: `htpbench` 的 execute = `graphExecute` 的 wall time, 与 PC 的
  `stream_inference_ms` 同义

### 实测 (小米13, HTP v73, arch=73 soc=43 vtcm=8MB, 非 root)

`htpbench` 单进程轮询 (3 轮 x 120 帧, 33ms 占空比):

| 档位 | execute avg | vs default |
|---|---|---|
| `default` (不下发) | 15.81ms | — |
| `burst` | **5.80ms** | **-63.3%** |
| `burst_nosleep` | 5.73ms | -63.7% |
| `sustained_high_performance` | 6.04ms | -61.8% |
| `balanced` | 7.34ms | -53.6% |
| `power_saver` | 11.22ms | -29.1% |

- 单调性 (`power_saver` 比 default 还慢) 证明这是真实频率效应, 不是噪声
- 4 个全新进程复测一致: default 15.59/15.52ms, burst 5.67ms
- 背靠背 burst 5.07ms vs 33ms 下 5.67ms → 每帧约 0.6ms 休眠唤醒成本,
  远小于不投票损失的 ~10ms (后者是频率差)
- `sleepDisable` 几乎无差别 (burst 5.80 vs burst_nosleep 5.73), 收益全在电压角

端到端 (真实 `cmd/streamer` + 协议喂帧器 30fps/122KB 帧, 3 对交替, 每腿 42s):

| leg | 档位 | execute | decode | quant | read | 首窗->末窗 |
|---|---|---|---|---|---|---|
| A1/A2/A3 | `default` | 14.81/14.71/14.79 | 4.8/4.5/4.8 | 3.2/3.0/3.2 | 10.0/10.7/10.2 | 稳定 |
| B1/B2/B3 | `burst` | **5.47/5.47/5.47** | 3.4/3.3/3.3 | 2.2/2.2/2.2 | 22.1/22.2/22.2 | 稳定 |

- `execute` 14.77±0.04 -> 5.47±0.00, 三对完全一致
- `read` 10.2 -> 22.2ms: 手机端余量翻倍
- CPU 段也跟着快了 (decode 4.7->3.3, quant 3.1->2.2): 不投票时线程长时间阻塞在
  fastrpc 上, CPU 频率被压下去; 投票把整机状态带起来后 CPU 段也受益

长局热稳定 (每档连续 4 分钟 = 8 个 30s 温度采样 / 52 个 5s execute 窗口, 30fps):

| 档位 | execute 逐窗范围 | 均值 | 首窗 -> 末窗 | fps | read |
|---|---|---|---|---|---|
| `default` | 14.03 ~ 14.94 | 14.72 | 14.75 -> 14.65 | 30 | 10.2 |
| `burst` | 5.43 ~ 5.51 | **5.47** | 5.48 -> 5.48 | 30 | 22.4 |
| `sustained_high_performance` | 5.79 ~ 5.87 | 5.84 | 5.86 -> 5.83 | 30 | 22.1 |

**没有任何热衰减**: burst 连续 4 分钟稳定在 5.47±0.03, fps 恒定 30, 无一帧丢失。
温度采样 (`/sys/class/thermal/thermal_zone*/temp`, 30s 一次) 也说明这份负载很轻:
DSP (`nspss-*`) 从 46-47C 升到 49-50C (+3C), CPU (`cpuss-*`/`cpu-1-*`) 50-51C,
`battery` 41-42C, `quiet_therm` 44C, `ddr` 46-48C — 离任何热关都不近。
真机长局 (>=30min) 的功耗/温度代价仍未量化, 但 4 分钟看不到拐点。

### 旋钮隔离 (哪个参数真正在起作用)

同一进程内轮换 (`-rounds 3 -iters 120 -interval 33ms`, burst 基准 5.80ms):

| 配置 | execute | 结论 |
|---|---|---|
| `burst` (rpcPolling=9999, rpcLatency=100) | 5.79 ~ 5.80 | 基准 |
| `burst -rpc-polling-time 0` | **6.24** | polling 值 **0.44ms** (V69+ 才有的参数, 确实有用) |
| `burst -rpc-control-latency 0` | 5.78 | `RPC_CONTROL_LATENCY` 在本机 **无效果** (噪声内) |
| `burst -rpc-control-latency 0 -rpc-polling-time 0` | 6.27 | 与只去掉 polling 一致 |
| `burst_nosleep` (两者都 0) | 6.30 | `sleepDisable` 仍无差别 |

即: **收益几乎全在 DCVS_V3 的电压角 + polling**, 其余旋钮可以不动。

### 数值结果不变 (无检测回归)

`htpbench -hash-output` (默认开) 对输出张量做 FNV-1a 64: `default` / `burst` /
`sustained_high_performance` / `burst_nosleep` 在同一输入下哈希**完全相同**
(`9be3fc3889d96fc7`), 且同档位跨轮次也一致 → 投票只改频率/电压, 不改变数值结果,
不存在精度回归风险 (也不需要重跑检测正确性对比)。

### 最终默认

`scripts/run_device.sh` 的默认档位改成 **`burst`** (实测最快且 4 分钟无衰减);
`STREAM_HTP_PERF=default` 退回不投票, `sustained_high_performance` 是保守档
(只慢 0.37ms, 用 TURBO 角而不是 MAX 角, 长局若发现耗电/发热明显就换它)。

### 方法论事故 (必须记住)

1. **小米13 上 `pgrep -x` / `pkill -x` 一个进程都匹配不到**: `/proc/<pid>/comm`
   不可读, procps 静默返回空。第一次跑 A/B 时 `pkill -x streamer` 从未生效,
   6 条腿全部并发、6 个 QNN 会话抢同一个 HTP, 测出 `execute` 65ms 的**完全相反的
   结论**。杀进程必须 `pgrep -f '^\./streamer'` + `kill -9`, 或直接记 `$!`;
   每条腿跑之前断言残留进程数为 0。`scripts/build_android.ps1` 的 push 前置检查
   已按此改。
2. **`build_android.ps1` 设置的 `GOOS`/`GOARCH`/`CGO_ENABLED` 是进程级的且不还原**,
   同一条命令里接着跑 `go test` 会尝试执行 android 产物并报
   `%1 is not a valid Win32 application`。
3. **`cmd/htpbench` 必须在结尾 `os.Exit`**: Termux 下正常 `return` 会让 QNN/厂商库的
   atexit 清理触发 SIGABRT (`signal arrived during cgo execution`), 与
   `cmd/streamer` 同一个坑。

### 下一步 (按收益排序, 都还没做)

1. **PC metrics 上的精确 latency 复测**: 本次端到端用的是协议喂帧器 (wstest), 只拿到
   手机侧分段; 有了 `burst` 之后应重跑一次带真游戏的 A/B, 用
   `stream_latency_avg_ms` / `stream_network_avg_ms` 把 29.1ms 的新值钉下来。
   注意 PC 端 streamer 在游戏非前台时会降到 `FpsIdle=2` (硬编码在 `main.go:474`),
   喂不出 30fps, 必须游戏在前台。
2. **长局 (>=30min) 热稳定**: 4 分钟没衰减, 但真实对局更长, 且 `burst` 把 DSP 钉在
   MAX 电压角, 功耗/温度代价还没量化 (手机电池/pa 温度可读, `/sys/class/thermal`)。
   若长局衰减明显, 退到 `sustained_high_performance` (TURBO 角, 只慢 0.24ms)。
3. **张量 I/O 零拷贝 (Phase 2)**: 现在 tensor 是 `QNN_TENSORMEMTYPE_RAW` + 普通
   Go 堆缓冲, 每次 execute 都要把 2.4MB 输入 + 1.4MB 输出经 fastrpc 往返。可行做法是
   `libcdsprpc.so` 的 `rpcmem_alloc/free/to_fd` 分配 dma-buf -> `mmap` ->
   `QnnMem_register(QNN_HTP_MEM_SHARED_BUFFER, fd, offset)` 拿 `Qnn_MemHandle_t`
   -> tensor 改 `QNN_TENSORMEMTYPE_MEMHANDLE`。**先探测再实现**。
   注意现在 `execute` 已经只有 5.5ms, 这条的收益上限也变小了。
4. **重生成 ctx binary (Phase 3)**: 改 `VTCM_SIZE`/`NUM_HVX_THREADS`/`NUM_CORES`
   必须重走 preparation。SDK 已经齐了 (`tmp/qairt-2.46.0.260424.zip` 里有
   `lib/aarch64-android/libQnnHtpPrepare.so`、`bin/x86_64-windows-msvc/qairt-converter`
   与 `qnn-context-binary-generator`), 唯一外部依赖是重新下载
   `yolo26n_v73_qnn.onnx` (仓库不存 onnx)。
5. **换小米 17 (v81) —— 环境已就绪 (2026-09-13)**: 见 §7。

### 环境事实 (别丢)

- **两台机器都可用, `run.sh` 一字不差, 差异只在 `~/streamer/device.env`**:

  | | 小米13 (备用) | 小米17 (主力) |
  |---|---|---|
  | Termux ssh | `192.168.1.103:8022`, 用户 `u0_a441` | `192.168.1.102:8022`, 用户 `u0_a352` |
  | `device.env` | 无需 (默认 v73) | `STREAM_ARCH=81` / `STREAM_CTX=models/ctx_v81.bin` |
  | 平台 | `arch=73 soc=43 vtcm=8MB` | `arch=81 soc=87 vtcm=8MB` |
  | `execute` 不投票 -> burst | 15.8 -> 5.47ms | 16.6 -> **3.32ms** |
  | adb 序列号 | `9b4a818` | (未连) |
  | `run.sh` 版本 | 旧版 (默认已经写死 v73, 功能等价; 等 13 开机后 `scp scripts/run_device.sh 192.168.1.103:~/streamer/run.sh` 对齐) | 与仓库 md5 一致 |

  两边目录都是 `~/streamer/{streamer, htpbench, run.sh, device.env, models/ctx_v*.bin, bus.jpg}`
- 部署: `scripts/build_android.ps1 [-Cmd ./cmd/xxx -Out tmp/xxx -RemoteName xxx] [-Push]`;
  **push 前必须先杀掉运行中的同名进程** (覆盖运行中的可执行文件会 ETXTBSY 静默失败),
  脚本内置的前置检查用 `pgrep -f '^\./name'`
- 库部署: `bash setup_libs.sh -a <73|81> [-q <QNN库目录>]` (递归解析 vendor 依赖,
  见 §7 的两个反向坑)
- PC 跑法: `.\streamer.exe -autodisplay -game=r6s -mhub-addr 127.0.0.1:9000 -noyolo`
  (metrics 在 `:8080/metrics`); 手机: `cd ~/streamer && ./run.sh`
  (档位用 `STREAM_HTP_PERF=burst ./run.sh`)
- 没有游戏在前台时, 用 `tmp/wstest` (仓库外, 见 `.gitignore` 的 `tmp/`) 作喂帧器:
  `go build -o tmp/wstest.exe ./tmp/wstest` 然后
  `tmp/wstest.exe -addr :9090 -img tmp/bus.jpg -fps 30 -quality 80`, 手机照常 `run.sh`

## 7. 小米17 (HTP v81 / Android 16) 环境落地 (2026-09-13)

`192.168.1.102:8022`, 同样的 `~/streamer` 布局, 同样的 `run.sh`。做完的验证:
真机 `run.sh` 跑通 (30fps / 122KB 帧): `execute 3.32ms`、`decode 2.0`、`quant 2.9`、
`read 25.3`、fps 29.6~29.8、`dropped=0`、`pinned worker thread mask=128` (CPU7)。

档位表 (同一套 `htpbench`, 33ms 占空比):

| 档位 | 小米13 (v73) | 小米17 (v81) |
|---|---|---|
| `default` | 15.81ms | 16.57ms |
| `burst` | 5.80ms | **3.53ms** |
| `burst_nosleep` | 5.73ms | 3.54ms |
| `sustained_high_performance` | 6.04ms | 4.05ms |
| `balanced` | 7.34ms | 4.96ms |
| `power_saver` | 11.22ms | 7.65ms |

规律完全一致 (单调、`power_saver` 仍慢于 default、7 个档位输出哈希相同
`01c5745fa9a9bcdd`), 只是绝对值更低。

### 与 Android 13 不同的两个坑 (都已修进 `setup_libs.sh`)

1. **漏复制**: Android 16 的 `/vendor/lib64/libcdsprpc.so` 除了
   `vendor.qti.hardware.dsp@1.0.so`, 还 NEEDED 一个 AIDL NDK 变体
   `vendor.qti.hardware.dsp-V1-ndk.so`。手工清单必漏, 症状是
   `dlopen libcdsprpc.so: dlopen failed: library "vendor.qti.hardware.dsp-V1-ndk.so" not found`。
2. **多复制 (更阴)**: `/system/lib64` 里已有的 soname **绝对不能**复制到
   `$PREFIX/lib`。`LD_LIBRARY_PATH` 优先, vendor 那份 `libc++.so` 会遮蔽系统版本,
   然后炸在**别的**库上:
   `CANNOT LINK EXECUTABLE: cannot locate symbol "_ZNSt3__113__hash_memoryEPKvm" referenced by "/system/lib64/liblog.so"`。
   `setup_libs.sh` 现在对 `/system/lib64` 已有的 soname 既不复制也不递归, 并对残留的
   遮蔽副本打印 `rm` 建议。

另外: 真实设备上 `-arch` 会被 QNN 忽略 (日志明说 `Specified config ARCH, ignoring on
real target`), 实际架构由 ctx binary 决定; 所以二进制**不用为换机重编**, 同一个
arm64 产物两台通用。

v81 上还没试的旋钮 (head 文件里有, 留给下一轮): `DDR_PERF_MODE` (要 V81 + polling +
官方说仅 LLM 场景)、`ADAPTIVE_POLLING_TIME`、`HMX_V2`/`CENG` (这代才有意义)。

