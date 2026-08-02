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
