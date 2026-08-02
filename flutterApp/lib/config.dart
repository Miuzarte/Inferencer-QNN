/// 官方 v0.6.6 Release 的 QNN detect 模型（运行时下载并缓存）。
/// v73: Snapdragon 8 Gen 2+（小米 13 等）；v81: Snapdragon 8 Elite Gen 5（小米 17 等）。
const String kQnnDetectModelUrl =
    'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/yolo26n_v73_qnn.onnx';

/// HTP v81（SM8850 / 小米 17）用的官方模型。
const String kQnnDetectModelUrlV81 =
    'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/yolo26n_v81_qnn.onnx';

/// 默认远程 WebSocket 地址（GoCVStreamer）。
const String kDefaultWsUrl = 'ws://192.168.1.100:9090/stream';
