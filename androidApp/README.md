# androidApp — 原生 Kotlin + ORT QNN 版 YOLO26n

参照 flutterApp 的成功链路，用 Kotlin + Compose 重新实现的原生 Android 推理
App：包名 `io.github.miuzarte.inferencer`，支持 v73/v81 模型切换、相机与远程
双源，远程推理结果回传 GoCVStreamer。

## 技术栈

- ONNX Runtime `onnxruntime-android-qnn:1.26.0` + `qnn-runtime:2.46.0`
  （验证过的版本组合，勿混用 2.42/2.48）；
- Kotlin + Compose，CameraX 相机源；
- okhttp WebSocket 远程源。

## 当前状态

- [x] 小米 13（HTP v73）QNN session 创建成功（`CreateDevice` → `SetupBackend`
      → `Session successfully initialized`，outputDims `[1,84,8400]`）；
- [x] P1 骨架、包名、模型运行时下载/缓存（v73/v81 Dropdown 切换）；
- [ ] 连 GoCVStreamer 回传（PC 端青色框 + 统计）待验证；
- [ ] CameraX 前后镜头切换待验证。

## 构建与安装

```powershell
.\gradlew assembleDebug
adb -s 192.168.1.103:5555 install -r -t app\build\outputs\apk\debug\app-debug.apk
```

与 flutterApp 同包名，可互相 `adb install -r` 覆盖并保留相机授权。

## 关键点（踩坑结论）

- Manifest 必须声明 `<uses-native-library android:name="libcdsprpc.so"/>`，
  否则 App linker 命名空间搜不到 vendor 分区，`QnnDevice_create` 失败；
- `ADSP_LIBRARY_PATH` 必须在 createSession 之前设置（进程级一次即可）；
- QNN EP 用 `backend_path=libQnnHtp.so` + `htp_performance_mode=burst`；
- v73/v81 是不同的 context binary，不可混用（v81 模型跑 v73 设备是旧版崩溃的根因）；
- 保持 SELinux Enforcing，不需要 root。

详细调研结论、分阶段计划与验证清单见 [PLAN.md](PLAN.md)。
