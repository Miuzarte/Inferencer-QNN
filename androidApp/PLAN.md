# androidApp 重构规划（对齐 flutterApp 成功链路）

> 2026-08-02 调研完成。目标：参考 flutterApp 的初始化流程重新实现原生 Kotlin App，
> 与 flutterApp 统一使用包名 `io.github.miuzarte.inferencer`（flutterApp 已从
> `io.github.miuzarte.inferencer_flutter` 改过来，androidApp 本来就是这个名字），
> 两者互相 `adb install -r` 覆盖时保留相机授权。**不处理文件复用**：模型各自
> 下载到自己的目录，和 flutterApp 当作两个不同的 App。

## 0. 背景与约束

- 当前调试设备：小米 13（fuxi / 2211133C / SD 8 Gen 2 / HTP **v73**），
  ADB `192.168.1.103:5555`；用户暂不在设备前，只能靠 adb 验证。
- 设备上现装的是旧包名 `io.github.miuzarte.inferencer_flutter`（debug 签名，
  versionCode=1）；改名后的 flutterApp / androidApp 会作为新应用安装，
  首次需重新授权相机，之后两者同包名互相覆盖不再丢授权。
- 模型不依赖 flutterApp 的缓存，首次启动自行下载到 `filesDir/models/`。
- 最终形态对齐 flutterApp：预览框 + 模型下拉（v73/v81）+ Camera/Remote Tab +
  配置面板（前后镜头切换、Conf、WS URL）；远程回传协议与 GoCVStreamer 一致。

## 1. flutterApp 初始化链路（研究结论，实现必须照抄）

### 1.1 依赖组合（成功关键，禁止自行改版本）

```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.26.0")
implementation("com.qualcomm.qti:qnn-runtime:2.46.0")
packaging { jniLibs { useLegacyPackaging = true } }
```

- `onnxruntime-android-qnn:1.26.0` 是带 QNN EP 的 ORT Android 构建；
- `qnn-runtime:2.46.0` 显式覆盖传递依赖，内含 `libQnnHtpV73Skel.so` 等
  QAIRT Skel（设备表覆盖 SM8850 / 8 Gen 2）；
- `useLegacyPackaging = true` 保证这些 .so 真实落盘到 nativeLibraryDir
  （fastrpc 按文件路径加载，不能留在 APK 内）。

### 1.2 原生 session 初始化（OrtQnnModel.kt 三步）

**Manifest 前置条件（本次调试找到的真正根因）**：

```xml
<application ...>
    <uses-native-library
        android:name="libcdsprpc.so"
        android:required="false" />
</application>
```

- QNN HTP 的 `QnnDevice_create` 需要 dlopen `/vendor/lib64/libcdsprpc.so`
  （fastrpc 客户端，及其依赖 `vendor.qti.hardware.dsp@1.0.so` 等 vendor 库）。
- Android 12+ 的 App linker namespace 默认**搜不到 vendor 分区**；没有上面的声明时，
  QNN 只能找到 APK 自带库，找不到 libcdsprpc → `QNN_DEVICE_ERROR_INVALID_CONFIG`，
  且没有任何 fastrpc/dsp-client 日志。
- flutterApp 能跑是因为 LiteRT 依赖传递声明了该 native library（合并 Manifest 里
  有 `<!-- Qualcomm NPU --> <uses-native-library android:name="libcdsprpc.so" .../>`），
  androidApp 没有 LiteRT，所以必须自己声明。
- 调试手段：strace 对比两个进程，唯一差异是 libcdsprpc.so 的搜索路径
  （flutterApp 多搜了 `/odm/lib64`、`/vendor/lib64`）；`sku` 的 SELinux EACCES
  两边都有，不是根因，`soc_model`/`htp_arch` 也无效。

```kotlin
// 1) 创建 session 前必须设置（进程级，一次即可）
android.system.Os.setenv(
    "ADSP_LIBRARY_PATH",
    context.applicationInfo.nativeLibraryDir + ";/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp",
    true,
)

// 2) QNN EP 选项（与失败路线 A 的差异：backend_path 用库名，加 burst 模式）
OrtSession.SessionOptions().apply {
    addQnn(mapOf(
        "backend_path" to "libQnnHtp.so",
        "htp_performance_mode" to "burst",
    ))
}

// 3) 直接 createSession(文件路径)，context binary 自带 arch，不需要 htp_arch
```

### 1.3 模型

- 官方 release v0.6.6：`yolo26n_v73_qnn.onnx`（8 Gen 2+）/
  `yolo26n_v81_qnn.onnx`（8 Elite Gen 5），运行时下载并缓存；
- **v73/v81 是不同 context binary，不可混用**（旧 androidApp 内置 v81 模型
  跑在 v73 设备上，很可能就是当年 Start 崩溃的原因）；
- 输入 shape：官方 QNN 导出为 NHWC `[1, 640, 640, 3]`（代码要同时兼容
  NCHW，按 `inputShape[3]==3` 判断）；
- 输出：端到端 `[1, 300, 6]`（x1,y1,x2,y2,score,class）或经典 `[1, 84, 8400]`
  （后处理分别走阈值直出 / argmax+NMS）。

### 1.4 预处理

- 相机：CameraX `OUTPUT_IMAGE_FORMAT_RGBA_8888` 单平面快速路径
  （~2-5ms，勿用旧工程的 YUV→JPEG→Bitmap 慢路径）→ letterbox 640×640
  （min-scale + 黑边，与官方 ImageUtils 一致）→ `getPixels` → HWC float /255；
- 远程：JPEG decode → Bitmap → 640×640 letterbox（等尺寸时恒等）→ 同上；
- 结果框：归一化到**输入原图**（相机为旋转校正后的帧，远程为 JPEG 帧），
  叠加层按预览尺寸缩放。

### 1.5 结果与协议

- Flutter 端踩坑：插件 `predictSingleImage` 的 boxes 里 `class` 是 **String 类名**，
  且 `detections[].classIndex` 写死 0 → 必须从原始 boxes 读类别。Kotlin 直接拿
  `Box.index`（Int），无此问题；
- 回传 GoCVStreamer：`{"frame_id":N,"detections":[{x1,y1,x2,y2,score,class,class_name}],"inference_ms":M}`，
  M 为手机端实测推理耗时；跳帧防堆积。

## 2. 现状盘点（androidApp 已有资产）

`B:\Git\Inferencer\androidApp`（git 未跟踪）目前是路线 A 的残留：

- ✅ 可保留：Gradle 骨架（AGP 9.3.1 / Kotlin 2.2.10 / Compose BOM / Gradle 9.5）、
  ORT+QNN 依赖与 `useLegacyPackaging`、`noCompress onnx`、
  `InferenceEngine.kt`（QNN init + WS 客户端 + 端到端/经典后处理 + NMS +
  class_name 输出，已经按官方写法实现）；
- ❌ 要改：
  - `assets/yolo26n_qnn_v81.onnx`（3.4MB v81）→ 删除，改为运行时下载/复用缓存；
  - Manifest 缺 `CAMERA` 权限；
  - UI 只是 host + 日志页，没有预览/相机/模型下拉；
  - versionCode=1 与已装 Flutter 版相同 → 升到 2 更稳妥。

## 3. 分阶段实施计划（不一步到位）

### P0 调研（本次完成）

- 读通 flutterApp 的 `YOLO.loadModel/predict` 调用链、官方插件
  `OrtQnnModel.kt / ObjectDetector.kt / ImageUtils.kt / YOLOPlugin.kt`、
  模型下载缓存逻辑、旧 kotlin 工程 UI 结构。

### P1 骨架 + 包名 + 远程链路（先验证 QNN session 在 v73 上成功）

1. `app/build.gradle.kts`
   - `namespace`/`applicationId` 保持 `io.github.miuzarte.inferencer`（已满足）；
   - `versionCode = 2`、`versionName = "1.0.0"`；
   - 补 CameraX 依赖（core/camera2/lifecycle/view，1.4.1 或官方 1.6.0）；
   - 保留现有 ORT 1.26.0 / qnn-runtime 2.46.0 / okhttp。
2. `AndroidManifest.xml`：加 `<uses-permission android:name="android.permission.CAMERA" />`。
3. 新增 `ModelRepository`：
   - 模型存 `filesDir/models/`（与 flutterApp 的 `app_flutter/` 互不干扰）；
   - 缺失时从官方 release URL 下载（进度回调），URL 与
     `flutterApp/lib/config.dart` 一致；v73/v81 用 Dropdown 切换；
   - 切换即重建 `QnnYoloModel`（close 旧 session → 创建新 session）。
4. 把 `InferenceEngine.kt` 拆成：
   - `QnnYoloModel`（session 生命周期 + `detect(bitmap)`，去掉画框/日志刷屏）；
   - `RemoteStream`（okhttp WS：收 `[4B frameId][JPEG]` → 推理 → 回传含
     `inference_ms`，单线程串行 + 跳帧）。
5. 最小 UI：把旧 `MainActivity` 换成 Compose 骨架
   （参考 `Inferencer-kotlin` 的 `MainScreen/ResultDisplay/DetectionCanvas/RemoteOptions`），
   先只做 Remote Tab 跑通。
6. 验证（用户不在设备前，adb 驱动）：
   - `.\gradlew assembleDebug`；
   - `adb -s 192.168.1.103:5555 install -r -t app-debug.apk`；
   - 启动后 logcat 断言：`ONNX Runtime QNN session on NPU; inputDims=[1, 640, 640, 3]`、
     `libQnnHtpV73Skel.so` fastrpc open 成功；
   - 连 PC streamer 回传，PC 端出现青色框 + 统计。

**P1 完成情况（2026-08-02）**：

- [x] `uses-native-library libcdsprpc.so` 声明后，小米 13（v73）QNN session 创建成功：
  `CreateDevice succeed → QNN SetupBackend succeed → Session successfully initialized`，
  logcat 出现 `ONNX Runtime QNN session on NPU; inputDims=[1, 640, 640, 3]
  outputDims=[1, 84, 8400]`；
- [x] 截图 `androidApp/screenshot/03-qnn-session-ok.png`；
- [ ] 连 GoCVStreamer 回传（青色框 + 统计）待下一步验证；
- [ ] CameraX 前后镜头切换待验证（P2）。

### P2 相机源 + 完整 UI

1. `CameraSource`（CameraX）：
   - `OUTPUT_IMAGE_FORMAT_RGBA_8888` + `KEEP_ONLY_LATEST`；
   - 中心裁剪或 letterbox 到 640×640（与预处理保持一致）；
   - 前后镜头切换（默认**前置**，用户手机在支架上挡住后置）、旋转校正；
   - 预览：PreviewView 或沿用旧工程 displayBitmap 方案，先跑通再优化。
2. 界面对齐 flutterApp：
   - LazyColumn：预览框（1:1 + 检测框叠加 + INF/SRC 统计）→ 模型下拉 →
     TabRow（Camera/Remote）→ 对应配置面板（Start/Stop、切换镜头、Conf；
     WS URL、Connect/Disconnect、Conf）。
3. 验证：adb 截图/`input tap` + logcat，前后镜头各跑一轮。

### P3 打磨

- 模型热切换不重启相机、错误状态与重连提示；
- 可选的 adb 自动化入口（intent extra 指定 source/url/conf，方便用户不在场时验证）；
- 日志分级（推理结果 Trace 级）、崩溃现场日志（`/proc/self/maps` 已内置）；
- 把成功经验整理进 `process.md` 路线 B4，为 Termux + Go 复刻做准备。

## 4. 验证清单（最终）

- [ ] `adb install -r` 覆盖后包名/签名一致、CAMERA 仍为 granted
      （flutterApp ↔ androidApp 同包名互覆盖生效；旧 `_flutter` 包不受影响）；
- [ ] 首次启动自行下载 v73 模型到 `filesDir/models/` 并可复用（不依赖 flutterApp）；
- [ ] 远程源：连 GoCVStreamer → 推理 → 回传（含 `inference_ms`）→ PC 青色框 + 统计；
- [ ] 相机源：预览 + 前置/后置切换 + 检测框；
- [ ] Dropdown 切 v81：在 v73 设备上应给出明确错误提示（不崩溃）；
- [ ] 与 flutterApp 对照推理延迟（同帧率下）。

## 5. 风险与注意

- **签名**：debug 构建必须用同一台机器的默认 `debug.keystore`，`adb install -r`
  才能覆盖 Flutter 版并保留数据；release 签名暂不需要。
- **版本配对**：ORT 1.26.0 + qnn-runtime 2.46.0 是验证过的组合；不要混用
  2.42/2.48（旧失败组合）。
- **ADSP_LIBRARY_PATH**：必须在 createSession 之前设置；进程级一次即可，
  热切换模型无需重设。
- **模型 arch**：v73/v81 不可混用，下拉默认 v73（小米 13）。
- **SELinux**：保持 Enforcing，不涉及 setenforce。真正的坑是 vendor 库链接
  命名空间：**必须声明 `uses-native-library libcdsprpc.so`**（等价于 flutterApp
  经 LiteRT 得到的传递声明）。`/sys/devices/soc0/sku` 读 EACCES 两边都有，可忽略。
- **数据目录**：同名包覆盖后数据目录是同一个，`app_flutter/` 残留属正常；
  本 App 只用 `files/` 自己的模型目录，不读写 flutterApp 的缓存。
