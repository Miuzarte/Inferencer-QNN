import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/det_box.dart';

/// GoCVStreamer WebSocket 客户端。
///
/// 收：binary `[4B frame_id LE][JPEG]`
/// 发：`{"frame_id":N,"detections":[...],"inference_ms":M}`（M 为手机端推理耗时）
class RemoteStream {
  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _sub;

  bool connected = false;

  /// 收到一帧（已拆出 frameId 与 JPEG 字节）。
  void Function(int frameId, Uint8List jpeg)? onFrame;

  /// 每秒回调一次接收帧率。
  void Function(double fps)? onFps;

  int _frameCount = 0;
  DateTime? _fpsStart;

  Future<void> connect(String url) async {
    disconnect();
    final channel = WebSocketChannel.connect(Uri.parse(url));
    _channel = channel;
    connected = true;
    _frameCount = 0;
    _fpsStart = DateTime.now();
    _sub = channel.stream.listen(
      _onData,
      onDone: () => connected = false,
      onError: (_) => connected = false,
      cancelOnError: true,
    );
  }

  void disconnect() {
    connected = false;
    _sub?.cancel();
    _sub = null;
    _channel?.sink.close();
    _channel = null;
  }

  void _onData(dynamic data) {
    if (data is! List<int>) return;
    final bytes = Uint8List.fromList(data);
    if (bytes.length < 5) return;
    final frameId = ByteData.sublistView(bytes).getUint32(0, Endian.little);
    final jpeg = Uint8List.sublistView(bytes, 4);
    onFrame?.call(frameId, jpeg);

    _frameCount++;
    final start = _fpsStart;
    if (start != null) {
      final elapsed = DateTime.now().difference(start).inMilliseconds;
      if (elapsed >= 1000) {
        onFps?.call(_frameCount * 1000 / elapsed);
        _frameCount = 0;
        _fpsStart = DateTime.now();
      }
    }
  }

  void sendResult(int frameId, List<DetBox> detections,
      {double inferenceMs = 0}) {
    final channel = _channel;
    if (channel == null || !connected) return;
    channel.sink.add(
      jsonEncode({
        'frame_id': frameId,
        'detections': [for (final d in detections) d.toJson()],
        'inference_ms': inferenceMs,
      }),
    );
  }
}
