# Inferencer — 骁龙 NPU 推理完整进度

## 目标

在 Xiaomi 17 (Snapdragon 8 Elite Gen 5 / v81, Android 16) 的 HTP NPU 上运行 YOLO26n 推理，连接 GoCVStreamer WebSocket 做实时目标检测。

最终形态：**Android App 一键启动** → YOLO 推理 → JSON 回传 GoCVStreamer。

---

## 架构演进

| 阶段 | 方案 | 结果 |
|------|------|------|
| Phase 1 | Kotlin + Maven ORT AAR | ❌ Maven ORT 不含 QNN EP，也不支持动态插件 EP |
| Phase 2 | Go + CGo (Termux proot) | ❌ proot 无法访问 /dev |
| Phase 3 | Go + purego (Termux 原生) | ❌ KSU root 下 QNN HTP 初始化失败 (FastRPC dev -1) |
| Phase 4 | Android App + exec Go binary (su) | 二进制跑起来了，但 QNN HTP 初始化仍然失败 |

---

## 当前项目结构

```
B:\Git\Inferencer\
├── go.mod                    # module Inferencer, go 1.26.5
├── main.go                   # WebSocket 客户端入口 (flag + yolo.New)
├── build.sh                  # 设备端编译脚本 (Termux 原生)
├── process.md                # ← 这个文件
├── inferencer                # Termux 编译的 Go ARM64 二进制 (6.5 MB)
├── .gitignore                # inferencer, *.exe, build/
│
├── ortpurego/                # onnxruntime_purego 副本 (纯 Go FFI)
│   ├── onnxruntime.go        # 引擎初始化 + API 表绑定
│   ├── api.go                # ORT API 结构体定义
│   ├── sesstion.go           # Session 管理 + AppendExecutionProvider V1/V2
│   ├── value.go              # Tensor 创建/读取
│   └── utils*.go             # 平台工具函数
│
├── yolo/
│   └── detector.go           # YOLO 预处理 (RGB→CHW /255) + 后处理 ([1,300,6])
│
├── androidApp/               # Android Studio 项目
│   ├── app/src/main/
│   │   ├── assets/           # 打包进 APK
│   │   │   ├── inferencer                  (Go ARM64 二进制)
│   │   │   └── yolo26n_qnn_v81.onnx        (HTP v81 编译模型, 3.4 MB)
│   │   ├── jniLibs/arm64-v8a/
│   │   │   ├── libonnxruntime.so             (交叉编译, 21.6 MB)
│   │   │   └── libonnxruntime_providers_qnn.so (Maven AAR, 3.7 MB)
│   │   ├── java/io/github/miuzarte/inferencer/
│   │   │   └── MainActivity.kt             (Compose UI + exec su)
│   │   ├── AndroidManifest.xml             (INTERNET + FOREGROUND_SERVICE)
│   │   └── res/...
│   └── app/build/outputs/apk/debug/app-debug.apk  (编译产物, ~60 MB)
│
├── build/                    # 工作文件 (被 .gitignore 忽略)
│   └── libs/                 # AAR 提取产物 (~45 MB)
│       ├── ort/jni/arm64-v8a/libonnxruntime.so    (Maven 版, 28.6 MB)
│       └── qnn/jni/arm64-v8a/libonnxruntime_providers_qnn.so (3.7 MB)
│
└── CROSS_BUILD.md            # ORT 交叉编译方案文档
```

**依赖项目：**

```
B:\Git\onnxruntime\                    # ORT v1.28.0 源码 (非 git, tarball 解压)
├── .venv\                             # uv Python 虚拟环境 (Python 3.13.9)
└── build\android_arm64\Release\
    └── libonnxruntime.so              # 交叉编译产物 (21.6 MB, Bionic ARM64)

B:\Git\onnxruntime-qnn\                # ORT + QNN EP 源码 v2.6.0 (非 git)
B:\Git\go-vision\_weights\yolo26_weights\
    └── yolo26n_qnn_v81.onnx           # YOLO26n HTP v81 模型 (3.4 MB)

B:\Software\AndroidSDK\                # Android SDK
├── ndk\29.0.14206865\                 # NDK r29
└── platforms\android-37.0\            # API 37
```

---

## Go 程序详解

### main.go

WebSocket 客户端连接 GoCVStreamer：
- 收 binary `[4B frameID LE][JPEG]`
- YOLO 推理
- 回 JSON `{"frame_id":N,"detections":[{class_id,confidence,x1,y1,x2,y2}]}`
- 坐标归一化 [0,1] / 640

**Flags:**
```
--host     默认: 无 (必填)
--model    默认: /data/data/com.termux/files/home/.local/lib/yolo26n_qnn_v81.onnx
--ort-lib  默认: $nativeLibraryDir/libonnxruntime.so (App) / Termux 自定义
--qnn-lib  默认: $nativeLibraryDir/libonnxruntime_providers_qnn.so
--qnn-htp  默认: /vendor/lib64/libQnnHtp.so
--conf     默认: 0.45
--retry    默认: 0 (无限重连)
```

### yolo/detector.go — QNN EP 初始化

```go
// 1. 注册 QNN EP 插件库
engine.RegisterExecutionProviderLibrary("QNNExecutionProvider", qnnLibPath)

// 2. V2 设备发现
devices := engine.GetEpDevices()
// → 找到 QNNExecutionProvider 设备

// 3. V2 优先 → V1 by-name 回退
opts := SessionOptions{
    "backend_path": "/vendor/lib64/libQnnHtp.so",
    "htp_arch": "81",
    "enable_htp_fp16_precision": "1",
    "enable_htp_shared_memory_allocator": "1",
}
if !AppendExecutionProviderV2(device, opts) {
    opts["backend_type"] = "htp"
    AppendExecutionProvider("QNNExecutionProvider", opts)  // V1 fallback
}

// 4. 成功后禁用 CPU 回退
AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1")
```

### 模型信息

- 文件: `yolo26n_qnn_v81.onnx` (3.4 MB, INT8+W8A16)
- 输入: `"images"` [1, 3, 640, 640] RGB /255.0
- 输出: `"output0"` [1, 300, 6] (x1, y1, x2, y2, score, class_id)
- NMS 已内建（不需后处理 NMS）
- 嵌入 EPContext node: `QNNExecutionProvider_QNNExecutionProvider_7160...`

---

## ORT 交叉编译

### 编译命令 (B:\Git\onnxruntime)

```powershell
# 1. 创建 venv
uv venv --python 3.13 B:\Git\onnxruntime\.venv
uv pip install --python .venv\Scripts\python.exe flatbuffers numpy packaging protobuf

# 2. 交叉编译 (约 27 分钟, 2598 targets)
$env:PATH = "B:\Software\ninja;$env:PATH"
python tools\ci_build\build.py `
  --config Release `
  --android `
  --android_sdk_path B:\Software\AndroidSDK `
  --android_ndk_path B:\Software\AndroidSDK\ndk\29.0.14206865 `
  --android_abi arm64-v8a `
  --android_api 27 `
  --build_shared_lib `
  --use_xnnpack `
  --cmake_generator Ninja `
  --build_dir build\android_arm64 `
  --parallel `
  --cmake_extra_defines onnxruntime_USE_EP_API_ADAPTERS=ON `
  --skip_tests `
  --skip_submodule_sync
```

**关键标志:**
- `onnxruntime_USE_EP_API_ADAPTERS=ON` ← 动态插件 EP 支持（不是 `ENABLE_DYNAMIC_PLUGIN_EP`）
- `use_nnapi` ← 不传（默认 OFF），避免链接 `libandroid.so`
- 无 NNAPI 的构建只链接 `liblog.so`（Bionic 日志）

**依赖 (Maven 版 vs 交叉编译版):**

| 版本 | 依赖 | 大小 |
|------|------|------|
| Maven AAR | libc, libm, libdl, liblog, **libandroid** → libgui ❌ | 28.6 MB |
| 交叉编译 | libc, libm, libdl, liblog ← 干净 | 21.6 MB |

---

## Android App 详解

### 架构

```
MainActivity (Compose UI)
├── extractAssets() → /data/user/0/.../files/inferencer/
│   ├── inferencer                    (assets)
│   └── yolo26n_qnn_v81.onnx          (assets)
├── jniLibs → nativeLibraryDir
│   ├── libonnxruntime.so             (jniLibs, legacy packaging)
│   └── libonnxruntime_providers_qnn.so (jniLibs)
└── startInference()
    └── ProcessBuilder("su", "-c", cmd)
        └── LD_LIBRARY_PATH=...
            ADSP_LIBRARY_PATH=...
            ./inferencer --host ... --model ... --ort-lib ... --qnn-lib ... --qnn-htp ...
```

### build.gradle.kts 关键配置

```kotlin
compileSdk 37 (api 37.0)
minSdk 27
targetSdk 36
packaging { jniLibs { useLegacyPackaging = true } }  // AGP 9 必须开, 否则不提取 .so
androidResources { noCompress += setOf("so", "onnx", "") }  // 不压缩 asset 文件
```

### 关键发现

1. **AGP 9 默认不提取 jniLibs** — 必须设置 `useLegacyPackaging = true`
2. **app data 目录 noexec** — 必须用 `su` 或 `linker64` 执行二进制
3. **su -c 可行** — `ProcessBuilder("su", "-c", cmd)` 能跑 Go 二进制
4. **KSU Manager 需授权 shell** — 否则 su 无权限
5. **不能放 `/data/local/tmp/`** — 用户选择用自己的目录

---

## QNN HTP 初始化失败 — 完整诊断历史

### 错误: "Unable to acquire HTP arch"

这是 QNN SDK (`libQnnHtp.so`) 内部的运行时错误。FastRPC 可以打开 CDSP 默认 domain (3)，但无法创建计算域。

### FastRPC 失败链路

```
ORT QNN EP
  → libQnnHtp.so (dlopen, NEEDED: libc/libm/libdl/liblog)
    → dlopen("libcdsprpc.so")        ← 成功: "libcdsprpc.so loaded"
      → 构造函数 multidsplib_env_init()
        → check_multilib_util()      ← 成功: setenv("FASTRPC_CDSP_REFCNT_<PID>")
      → QNN 后端创建
        → domain_init(CDSP1..CDSP6)  ← 全部失败: dev -1
        → dlclose("libcdsprpc.so")   ← 触发析构 close_dev
          → domain_deinit(所有 11 个域): dev -1
          → fastrpc_apps_user_deinit
    → "Unable to acquire HTP arch"   ← libQnnHtp 返回错误
```

### 设备 DSP 守护进程 (所有存活)

| PID | 进程 | 功能 |
|-----|------|------|
| 2940 | `cdsprpcd` | **CDSP RPC 守护进程** (管理 CDSP 资源) |
| 2939 | `adsprpcd` | ADSP RPC 守护进程 |
| 2130 | `hexlpservice` | Hexagon 低功耗服务 (对应 `IHexlpService/unsigned`) |
| 2046 | `sscrpcd` | Snapdragon Sensor Core RPC |
| 591 | `audioadsprpcd` | 音频 ADSP RPC |
| 2876 | `mediaserver64` | 媒体服务 (打开 `/dev/fastrpc-cdsp` fd 9/11) |

### 已排除的原因

| 原因 | 验证方法 | 结果 |
|------|----------|------|
| SELinux | `setenforce 0` + `dmesg` 检查 AVC | 无 DSP 相关 AVC denied |
| 设备节点权限 | `sudo dd if=/dev/fastrpc-cdsp` | root 可以 open |
| 文件缺失 | `ls /vendor/lib64/libQnnHtp.so` 等 | 全部存在 |
| FastRPC flags | `vendor.fastrpc.process.attrs=0x8/0x68` | 无效果 |
| sepolicy | `sepolicy.rule` 注入 `ksu` + `untrusted_app` | 无效果 |
| QNN 后端 | 尝试 `libQnnDsp.so` 替代 `libQnnHtp.so` | 模型强制 HTP |
| ORT 版本 | 尝试 Maven AAR ORT | libandroid 依赖断裂 |
| ORT 交叉编译 | 尝试自编 ORT | 加载正常，QNN HTP 一样失败 |

### 关键 logcat 日志

```
# 成功: 默认 CDSP domain 打开
fastrpc_apps_user_init done with default domain:3

# 成功: libcdsprpc 加载
multidsplib_env_init: libcdsprpc.so loaded

# 失败: 合计 11 个计算域全 dev -1
close_dev: unloading library libcdsprpc.so
domain_deinit for domain 200000..200005: dev -1
domain_deinit for domain 100000..100005: dev -1

# 结果: QNN HTP 初始化失败
Failed to validate compiled model compatibility for EP QNNExecutionProvider:
Unable to acquire HTP arch.
```

### 根因

设备 CDSP 资源由 `cdsprpcd` 守护进程独家管理。非系统进程（即使 KSU root）无法通过 FastRPC 创建 CDSP 计算域。Binder 层 (vendor.qti.hardware.dsp.IDspService) 需要特定的上下文/签名/包名校验，不仅 SELinux 层面。

---

## KSU 模块 (已创建，未成功)

```
/data/adb/modules/inferencer_sys/
├── module.prop        # id=inferencer_sys, 用于在 KSU Manager 显示
└── sepolicy.rule      # allow ksu/untrusted_app → vendor_qdsp/xdsp_device
```

**撤销方法:** `su -c "rm -rf /data/adb/modules/inferencer_sys" && reboot`

---

## 设备文件位置

```
# Termux 原生 (非 proot)
/data/data/com.termux/files/home/.local/lib/
├── libonnxruntime.so                  (交叉编译版)
├── libonnxruntime_providers_qnn.so    (Maven AAR)
└── yolo26n_qnn_v81.onnx               (已移动到这)

# Android App runtime
/data/user/0/io.github.miuzarte.inferencer/files/inferencer/
├── inferencer                         (assets 提取)
└── yolo26n_qnn_v81.onnx               (assets 提取)

# App native libs (jniLibs)
/data/app/~~XX==/io.github.miuzarte.inferencer-XX==/lib/arm64/
├── libonnxruntime.so                  (交叉编译, jniLibs)
├── libonnxruntime_providers_qnn.so    (Maven AAR, jniLibs)
└── libandroidx.graphics.path.so

# 系统 QNN 库
/vendor/lib64/libQnn{Htp,Dsp,Cpu,Gpu,Hta,Lpai,Ir,System,Saver}*.so
/vendor/lib64/libcdsprpc.so             (依赖链极深, 见下)
/vendor/lib64/libQnnHtpV81Stub.so
/vendor/lib64/libQnnHtpV81CalculatorStub.so
/vendor/lib/rfsa/adsp/libQnnHtpV81.so   (DSP 侧)

# 设备节点
/dev/fastrpc-cdsp          crw-rw-r-- system:system vendor_qdsp_device
/dev/fastrpc-cdsp-secure   crw-r--r-- system:system vendor_xdsp_device
/dev/fastrpc-adsp-secure   crw-r--r-- system:system vendor_xdsp_device
```

### libcdsprpc.so 完整依赖链

```
libcdsprpc.so
  → libhidlbase.so          # HIDL base (Binder IPC)
  → libhardware.so          # Hardware abstraction
  → libutils.so             # Android utils
  → vendor.qti.hardware.dsp@1.0.so    # DSP HIDL client
  → vendor.qti.hardware.dsp-V1-ndk.so # DSP AIDL client
  → libbinder_ndk.so        # Binder NDK
  → liblog.so
  → libc.so
  → libcutils.so            # cutils
  → libdmabufheap.so        # DMA buffer
  → libvmmem.so             # VM memory
  → libc++.so
  → libm.so
  → libdl.so
```

### QNN EP 插件依赖

```
libonnxruntime_providers_qnn.so (3.7 MB)
  → libQnnHtp.so              (2.1 MB, /vendor/lib64/)
  → libQnnSystem.so           (1.7 MB)
  → libQnnHtpPrepare.so       (存在但 access denied)
  → libcdsprpc.so             (596 KB, → 上述 12 个框架依赖)
```

---

## 剩余方案

| 方案 | 概率 | 复杂度 | 说明 |
|------|------|--------|------|
| **CPU EP + 新模型** | 🟢 高 | 🟡 中 | 模型导出时去掉 EPContext node, CPU 推理跑通功能 |
| **Binder 直调** | 🟡 中 | 🔴 高 | 逆向 `vendor.qti.hardware.dsp.IDspService` AIDL, 用 JNI 调用绕过 FastRPC |
| **Zygisk 注入** | 🟡 中 | 🔴 高 | 劫持 app 的 Binder 调用, 伪装 cdsprpcd 的身份 |
| **逆向相机 App** | 🟡 中 | 🔴 高 | 解包 MIUI 相机 APK, 看它如何用 QNN HTP |
| **重编译 ORT with QNN built-in** | 🔴 低 | 🔴 高 | 需要 Qualcomm QNN SDK (非公开), `B:\Git\onnxruntime-qnn` 源码需要 `--qnn_home` |
| **用 XNNPACK 加速** | 🟢 高 | 🟢 低 | ORT 已编译 XNNPACK, CPU 推理可用 ARM NEON 加速 (10-30ms/infer) |

---

## 快速恢复命令

```bash
# 恢复: 删除 KSU 模块
su -c "rm -rf /data/adb/modules/inferencer_sys" && reboot

# 恢复: 重置 FastRPC flags
su -c "setprop vendor.fastrpc.process.attrs 0"

# 编译 Go 程序 (Termux)
cd ~/git/Inferencer && bash build.sh

# 编译 APK (Windows)
cd B:\Git\Inferencer\androidApp && gradlew assembleDebug

# 安装 APK
adb install -r app\build\outputs\apk\debug\app-debug.apk

# 看日志
adb logcat -c && adb logcat -s Inferencer:*

# ORT 交叉编译 (Windows)
# 详见上方编译命令
```

---

## 当前状态

- [x] Go binary 构建 + purego FFI
- [x] ORT v1.28.0 交叉编译 (Bionic ARM64, 无 libandroid 依赖)
- [x] QNN EP 插件从 Maven AAR 提取
- [x] Android App 框架 (Compose UI, exec su, asset 提取, jniLibs)
- [x] QNN EP 注册 + V2 设备发现成功
- [ ] **QNN HTP session 创建 — 卡在 FastRPC CDSP 域初始化**
- [ ] 实际 YOLO 推理测试
- [ ] WebSocket 对接到 GoCVStreamer

---

## 路线 B：官方 Ultralytics Flutter App（2026-07-31 验证成功，当前主线）

> 自定义 androidApp 的 ORT QNN EP 路线已搁置。sherpa-onnx 官方 QNN APK 对照实验证明**小米 ROM 不限制第三方 App 使用 HTP**，问题出在自研 App 的 ORT/QNN 组合。转向官方 yolo-flutter-app 后一次通过。

### 结论（关键差异，之前自定义 App 失败的原因）

| 项 | 自定义 App（失败） | 官方 yolo-flutter-app（成功） |
|---|---|---|
| ORT | 1.26/1.27 | onnxruntime-android-qnn 1.26.0 |
| QNN runtime | 2.40/2.42/2.46/2.48 混用 | qnn-runtime 2.46.0（显式覆盖，含 SM8850 设备表） |
| ADSP_LIBRARY_PATH | 未设置 | nativeLibraryDir + /vendor/lib/rfsa/adsp + /system/lib/rfsa/adsp |
| htp_arch | 显式传 81 | 不传（context binary 自带 arch） |

官方插件 `OrtQnnModel.kt` 在创建 session 前调用 `Os.setenv("ADSP_LIBRARY_PATH", nativeLibraryDir + ";/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp", true)`，让 fastrpc 从 App 自带目录加载 `libQnnHtpV81Skel.so`（17.8MB，qnn-runtime AAR 内置），避免与厂商 CDSP 版本不匹配导致 `QNN_DEVICE_ERROR_INVALID_CONFIG`。

### 验证记录（Xiaomi 17 / SM8850 / HTP v81 / SELinux Enforcing）

1. 官方集成测试（`example/integration_test/qnn_benchmark_test.dart`）：
   `ENABLE_QNN=1 flutter test ... --dart-define=RUN_BENCH=true --dart-define=RUN_QNN=true --dart-define=QNN_ARCH=81`
   - 全部 7 任务（detect/segment/semantic/depth/classify/pose/obb）NPU 跑通，输出校验通过（All tests passed）
   - 模型：官方 v0.6.6 Release `yolo26n_v81_qnn.onnx` 等（运行时下载）
   - 性能（15 次均值）：detect 推理 4.7ms / classify 0.5ms / obb 4.1ms（对比 CPU detect 52.4ms）

2. 正常入口 App：example 新增 QNN NPU 相机页（`example/lib/presentation/screens/qnn_camera_screen.dart`，main.dart 默认入口 `/qnn`），打开即下载官方 v81 模型并跑 NPU 相机推理。logcat 确认：
   `ONNX Runtime QNN session on NPU; inputDims=[1, 640, 640, 3] outputDims=[[1, 84, 8400]]`
   `libQnnHtpV81Skel.so` fastrpc remote_handle64_open 成功（domain 3 / cdsp）

### 本地修改（B:\Git\yolo-flutter-app，未提交）

- `example/android/gradle.properties`：追加 `kotlin.incremental=false`（Windows 跨盘符增量编译崩溃）
- `example/android/app/build.gradle`：Kotlin `jvmTarget = JVM_17`（Flutter 用 AS JBR 21，与 Java 17 target 不一致）
- `example/lib/main.dart` + `example/lib/presentation/screens/qnn_camera_screen.dart`：QNN NPU 相机页默认入口

### 构建命令

```powershell
# 集成测试
$env:ENABLE_QNN="1"; flutter test integration_test/qnn_benchmark_test.dart -d 2cf30e03 --dart-define=RUN_BENCH=true --dart-define=RUN_QNN=true --dart-define=QNN_ARCH=81

# 正常 App（当前已装到设备）
$env:ENABLE_QNN="1"; flutter build apk --debug
adb install -r -t build\app\outputs\flutter-apk\app-debug.apk
```

---

## 路线 B2：最简 Flutter App（flutterApp/，2026-07-31 小米 13 验证成功）

> 目标：照官方插件做一个最简可用的 App（相机 + 远程 WebSocket 两路），先把可用性搞定，
> 为后续在 Termux + Go 里复刻成功方案提供最小参考实现。

### 项目结构（B:\Git\Inferencer\flutterApp\）

```
flutterApp/
├── lib/
│   ├── main.dart                    # MaterialApp 暗色主题
│   ├── config.dart                  # 官方 v0.6.6 QNN 模型 URL（v73/v81）
│   ├── home_screen.dart             # LazyColumn：PreviewBox → 模型下拉 → Tab → 配置面板
│   ├── models/det_box.dart          # 归一化检测框，JSON 协议与 GoCVStreamer 对齐
│   ├── services/remote_stream.dart  # WebSocket：收 [4B frame_id][JPEG]，回传 detections
│   └── widgets/
│       ├── preview_box.dart         # 1:1 预览 + 叠加框 + INF/SRC 指标
│       ├── source_tabs.dart         # Camera / Remote 切换
│       ├── model_selector.dart      # QNN 模型下拉（HTP v73 / v81）
│       ├── camera_panel.dart        # Start/Stop + 切换镜头 + Conf
│       └── remote_panel.dart        # WS URL + Connect + Conf
├── third_party/shared_preferences_android/   # 本地 vendored 副本（2.4.23）
└── android/                         # QNN 依赖 + useLegacyPackaging
```

### 关键点（成功经验）

1. **QNN 依赖与官方一致**（`android/app/build.gradle.kts`）：
   `com.microsoft.onnxruntime:onnxruntime-android-qnn:1.26.0` +
   `com.qualcomm.qti:qnn-runtime:2.46.0`，`useLegacyPackaging = true`。
2. **模型按 HTP 架构选**：v73（骁龙 8 Gen 2+，小米 13）与 v81（骁龙 8 Elite Gen 5，小米 17）
   是不同的 context binary，不能混用；App 内用 Dropdown 切换，方便两台设备对比。
3. **shared_preferences_android 编译坑**：2.4.x 的 `LegacySharedPreferencesPlugin.java` 引用
   已被迁移成 Kotlin 的 `StringListObjectInputStream`，javac 在 Flutter 插件构建里看不到 Kotlin 类
   （flutter/flutter #165850）。解法：vendored 2.4.23 副本，把旧版 Java 实现放回
   `android/src/main/java/.../StringListObjectInputStream.java` 并删除 Kotlin 副本（避免重复类）。
4. **镜头切换**：`YOLOView(controller: ..., lensFacing: LensFacing.front)` +
   `YOLOViewController.switchCamera()`；手机在支架上挡住后置镜头时默认用前置。
5. **模型热切换**：YOLOView 的 `didUpdateWidget` 检测到 modelPath 变化会原地
   `switchModel`，不需要重启相机。

### 验证记录（Xiaomi 13 / fuxi / SD 8 Gen 2 / HTP v73 / 备用机）

- `flutter build apk --debug` 成功，`adb install -r -t` 后启动正常。
- 首次 Start 需联网下载官方 `yolo26n_v73_qnn.onnx`（约 40-60 MB，GitHub 需代理）；
  代理开启后模型加载成功，相机预览 + 检测可用。
- 远程源：WS 收帧 → `YOLO.predict(jpeg)` → 按 `{frame_id, detections:[{x1,y1,x2,y2,score,class,class_name}]}`
  回传，协议与 GoCVStreamer 一致（跳帧防堆积）。

### 待办

- [ ] 小米 17（v81）上同样验证 flutterApp（切 Dropdown 到 v81）
- [ ] 远程源端到端联调 GoCVStreamer
- [ ] 对照插件源码 `OrtQnnModel.kt`，把 ADSP_LIBRARY_PATH / QNN session 配置复刻到 Termux + Go

---

## 路线 B3：GoCVStreamer 服务端（2026-07-31 已实现并冒烟测试通过）

> 对应旧计划 Inferencer-kotlin/PLAN.md 的 Phase B：在 GoCVStreamer 里加 WebSocket 流服务，
> 把 640×640 检测帧推给手机端，接收手机回传的归一化检测 JSON。

### 新增内容（B:\Git\GoCVStreamer）

- `sender/sender.go`（新包）：
  - HTTP `GET /stream` WebSocket 升级（gorilla/websocket，允许任意 Origin）
  - 推流循环：捕获帧 → 中心裁剪（`-streamcrop`，默认 1280，与本地 detector 一致）
    → libyuv 缩放 640×640 → JPEG（`-streamquality`，默认 80）→ 广播
  - 协议：binary `[4B frame_id LE][JPEG]`（与 flutterApp RemoteStream 一致）
  - 接收：text JSON `{"frame_id":N,"detections":[{x1,y1,x2,y2,score,class,class_name}]}`
    → 计数 + `OnResult` 回调（assist 接入点，`Transform` 可还原屏幕坐标）
  - 慢客户端背压：写超时 5s 自动断开
- `capturer/server.go`：新增 `CloneRgba()` / `CloneMat()`（深拷贝，供其他 goroutine 取帧）
- `main.go`：新 flag `-stream :9090`、`-streamfps 30`、`-streamquality 80`、
  `-streamcrop 1280`、`-nosender`（完全禁用推流）
- `http.go`：/metrics 增加 `stream_clients / stream_fps / stream_frames_sent /
  stream_detections / stream_last_count`

### 行为设计

- 有客户端连接时才 `RaiseCeiling(streamfps)`，无人观看时捕获回落到最低帧率，
  因此 `-noyolo` 下不会白白把捕获钉在 30FPS（`-nosender` 可彻底关闭）。
- 与旧计划的差异：推流循环独立于 detector 循环（而不是在 detector 里抓 resizeDst），
  这样 `-noyolo` / 游戏空闲时也能稳定推流，方便先用手机端联调。

### 验证（本机 127.0.0.1）

```powershell
streamer.exe -nogui -noyolo -noopencv -autodisplay -stream :9090 -port :8081
# 连 ws://127.0.0.1:9090/stream
# 收到 binary 59347 B，frameId=7，JPEG 魔数 FF D8 正确
# 回传 {"frame_id":7,"detections":[...]} 后 /metrics 显示 stream_detections=1
```

### 待办

- [x] `OnResult` 接入 detector/assist/gioui（2026-07-31）
- [ ] 手机端 Flutter Remote tab 连 PC 端 ws 端到端验证（同一局域网）
- [ ] 需要时再考虑把推流循环挂进 detector 循环复用 resizeDst（省一次缩放）

### 远程结果接入（2026-07-31 已完成）

- `detector.Engine` 新增远程结果通道：
  - `SetRemoteResults([]yolo26.DetResult)`：手机端回调写入（屏幕坐标系）
  - `SnapshotSources() (local, remote, stats)`：远程结果带 TTL（`-streamttl`，默认 500ms），
    超过未更新自动清理；本地/远程可同时存在
  - `Snapshot()` 保持原签名，返回合并结果（assist/metrics 兼容）
- gioui 绘制：本地 YOLO 绿框 + 远程 NPU 青框（`ColorCyan`）同屏叠加
- assist 多来源策略：**远程新鲜时优先远程，过期/断开自动回退本地**
  （assist 仍按“距屏幕中心最近”选择目标，只是结果来源按上述优先级取）
- `main.go` OnResult：`sender.Transform` 归一化 → 屏幕坐标 → `SetRemoteResults`

### 坐标 Bug 修复（2026-07-31）

- 现象：青色框缩成 1px 点，位于实际位置左上方。
- 根因：`sender.Transform` 把归一化坐标（已除以 640）又乘了 `cropSize/InputSize=2`，
  0..1 只映射到 0..2，导致框宽高≈1px。正确做法是直接乘裁剪区边长 `cropSize`（1280）。
- 诊断手段：`stream status` 每秒日志输出归一化值与转换后屏幕坐标
  （`n_x1/n_y1/n_x2/n_y2` 与 `s_x1/s_y1/s_x2/s_y2`），一眼定位是回传值错还是转换错。

### 推理源重构 + 按全链路延迟选源（2026-07-31 已完成）

- `detector.Source` 接口（类比 capturer.Source）：
  `Snapshot() (results []Result, latency time.Duration, fresh bool)` + `Close() error`
  - `detector.Engine`（本地 YOLO）：`latency` = 最近一帧推理耗时（`stats.Cost`）
  - `detector.RemoteSource`（手机端 NPU）：结果由 `sender.OnResult` 写入，
    `latency` = 帧发出到收到结果的全链路延迟，TTL 内新鲜（`-streamttl`）
- `sender` 记录每帧发送时间（`sentAt[frameID]`，256 条上限/2s 清理），
  收到回传时算 `now - sentAt[frameID]`；`/metrics` 新增 `stream_latency_ms`
- 远程延迟拆分：Flutter 回传 JSON 增加 `inference_ms`（手机端 predict 耗时），
  服务端 `网络延迟 = 全链路 - inference_ms`（clamp ≥ 0）；
  `/metrics` 新增 `stream_inference_ms` / `stream_network_ms`
- assist 策略改为**每 Tick 遍历所有推理源，选全链路延迟最低且新鲜的来源**
  （不再无条件远程优先）：远程 NPU 10ms 推理 + 网络延迟 > 本地 20ms 时会自动用本地
- UI：`detector.Drawer` 聚合绘制所有源（本地绿/远程青）；`-noyolo` 时远程源也能单独显示
- gioui 统计：Detection 单独一行，显示
  `Local: X.Xms`（本地推理延迟）+ `Remote: net X.Xms + inf X.Xms`（远程网络/推理延迟）
- 远程统计与青色框同口径清理：`/metrics` 新增 `stream_fresh`，
  超过 `-streamttl` 未收到结果时延迟/结果数统计清零，gioui 远程部分不再显示
- sender 锁拆分（clientMu/statsMu）：慢客户端写阻塞不再卡住结果处理（OnResult）
- `-streamttl` 现在作用于 RemoteSource 的新鲜度（默认 500ms，0=禁用远程结果）

### 前台游戏门控（2026-07-31 已完成）

- 新增 `gamefocus.go`：每秒取前台窗口 PID（GetForegroundWindow + GetWindowThreadProcessId），
  gopsutil 拿进程名做大小写不敏感匹配
  - `-game=cs2`：匹配 `cs2` / `cs2.exe`
  - `-game=r6s`：匹配 `rainbowsix` / `rainbowsix.exe`
- assist 新增 `SetForegroundAllowed(func() bool)` 门控：非目标游戏前台时完全停用
  （Tick 直接清空目标），gioui 状态显示 `Aim:OFF(FG)`
- 状态变化时打一次 Info 日志（不会每秒刷屏）；未知 game 模式不启用门控

### 验证

```powershell
# 回环冒烟：回传后 300ms 内 /metrics
detection_count=1 stream_latency_ms=9.5
```
