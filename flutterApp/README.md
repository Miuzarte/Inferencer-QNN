# flutterApp — YOLO26n QNN 检测（Flutter）

主要参考 Ultralytics 官方 [yolo-flutter-app](https://github.com/ultralytics/yolo-flutter-app)
实现（官方插件以本地源码方式依赖 `../../yolo-flutter-app`，含 SM8850 QNN 支持），
在小米 13（Snapdragon 8 Gen 2 / HTP v73）上验证 QNN 推理与 GoCVStreamer 远程链路。

## 功能

- 主页 LazyColumn：预览框（检测框叠加 + 统计）→ 模型下拉 → TabRow
  （相机源 / 远程源）→ 对应配置面板；
- 模型 Dropdown：`yolo26n_v73_qnn.onnx`（8 Gen 2+）与 `yolo26n_v81_qnn.onnx`
  （8 Elite Gen 5），运行时从官方 release v0.6.6 下载并缓存；
- 相机源：CameraX 前后镜头切换（默认前置）；
- 远程源：WebSocket 对接 GoCVStreamer 的 640×640 流帧，回传
  `{"frame_id":N,"detections":[...],"inference_ms":M}`；
- 结果默认只保留 person（class 0）。

## 运行

```bash
flutter pub get
flutter run -d <device>
```

## 说明

- 包名：`io.github.miuzarte.inferencer`；
- 默认远程地址：`ws://192.168.1.100:9090/stream`（见 `lib/config.dart`）；
- `third_party/shared_preferences_android` 是 vendored 副本，用于绕开
  `shared_preferences_android 2.4.23` 在部分 JDK 下的编译失败
  （`dependency_overrides` 指向它）。
