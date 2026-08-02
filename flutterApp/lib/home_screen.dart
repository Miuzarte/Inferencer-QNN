import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import 'config.dart';
import 'models/det_box.dart';
import 'services/remote_stream.dart';
import 'widgets/camera_panel.dart';
import 'widgets/model_selector.dart';
import 'widgets/preview_box.dart';
import 'widgets/remote_panel.dart';
import 'widgets/source_tabs.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  SourceType _source = SourceType.camera;
  bool _cameraRunning = false;
  bool _remoteConnected = false;
  double _conf = 0.45;
  Uint8List? _remoteFrame;
  List<DetBox> _detections = const [];
  double _inferenceMs = 0;
  double _sourceFps = 0;
  String? _status;
  String _modelUrl = kQnnDetectModelUrl;
  final YOLOViewController _cameraController = YOLOViewController();
  final LensFacing _lensFacing = LensFacing.front;

  final RemoteStream _remote = RemoteStream();
  YOLO? _remoteYolo;
  bool _remoteModelLoaded = false;
  bool _processing = false;

  @override
  void initState() {
    super.initState();
    _remote.onFrame = _handleRemoteFrame;
    _remote.onFps = (fps) {
      if (mounted) setState(() => _sourceFps = fps);
    };
  }

  @override
  void dispose() {
    _remote.disconnect();
    _remoteYolo?.dispose();
    _cameraController.dispose();
    super.dispose();
  }

  void _onModelChanged(String url) {
    if (url == _modelUrl) return;
    setState(() {
      _modelUrl = url;
      _detections = const [];
      _status = '模型切换中…';
    });
    // 远程源下一次收到帧时按新模型重建。
    _remoteYolo?.dispose();
    _remoteYolo = null;
    _remoteModelLoaded = false;
  }

  void _switchSource(SourceType type) {
    if (type == _source) return;
    _stopCamera();
    _disconnectRemote();
    setState(() {
      _source = type;
      _detections = const [];
      _inferenceMs = 0;
      _sourceFps = 0;
      _remoteFrame = null;
    });
  }

  // ---------- 摄像头源 ----------

  Future<void> _startCamera() async {
    final perm = await Permission.camera.request();
    if (!perm.isGranted) {
      setState(() => _status = '相机权限被拒绝');
      return;
    }
    setState(() {
      _cameraRunning = true;
      _status = '加载 QNN 模型…';
    });
  }

  void _stopCamera() {
    setState(() {
      _cameraRunning = false;
      _detections = const [];
      _status = null;
    });
  }

  void _onCameraResults(List<YOLOResult> results) {
    if (!mounted || !_cameraRunning) return;
    setState(() {
      _detections = [for (final r in results) DetBox.fromYolo(r)];
    });
  }

  void _onCameraMetrics(YOLOPerformanceMetrics m) {
    if (!mounted) return;
    setState(() {
      _inferenceMs = m.inferenceMs;
      _sourceFps = m.fps;
    });
  }

  // ---------- 远程源 ----------

  Future<void> _connectRemote(String url) async {
    setState(() {
      _status = '连接中…';
      _remoteFrame = null;
      _detections = const [];
    });
    await _remote.connect(url);
    if (!mounted) return;
    setState(() {
      _remoteConnected = _remote.connected;
      _status = _remote.connected ? '已连接，等待帧…' : '连接失败';
    });
  }

  void _disconnectRemote() {
    _remote.disconnect();
    _remoteYolo?.dispose();
    _remoteYolo = null;
    _remoteModelLoaded = false;
    setState(() {
      _remoteConnected = false;
      _remoteFrame = null;
      _detections = const [];
      _status = null;
    });
  }

  Future<void> _handleRemoteFrame(int frameId, Uint8List jpeg) async {
    if (_processing) return; // 跳帧，避免推理堆积
    _processing = true;
    try {
      _remoteYolo ??= YOLO(modelPath: _modelUrl, task: YOLOTask.detect);
      if (!_remoteModelLoaded) {
        final ok = await _remoteYolo!.loadModel();
        if (!ok) {
          if (mounted) setState(() => _status = 'QNN 模型加载失败');
          return;
        }
        _remoteModelLoaded = true;
        if (mounted) setState(() => _status = 'NPU 就绪');
      }

      final stopwatch = Stopwatch()..start();
      final result = await _remoteYolo!.predict(jpeg, confidenceThreshold: _conf);
      final ms = stopwatch.elapsedMicroseconds / 1000.0;

      // 只用 person（COCO class 0）：插件 detections 的 classIndex 写死为 0，
      // 必须从原始 boxes 里读真实类别。
      final rawList = (result['boxes'] as List?) ?? const [];
      final dets = <DetBox>[
        for (final raw in rawList)
          if (raw is Map && DetBox.classIndexFromBoxMap(raw) == 0)
            DetBox.fromBoxMap(raw),
      ];
      if (!mounted) return;
      setState(() {
        _remoteFrame = jpeg;
        _detections = dets;
        _inferenceMs = ms;
      });
      _remote.sendResult(frameId, dets, inferenceMs: ms);
    } catch (e) {
      if (mounted) setState(() => _status = '推理错误: $e');
    } finally {
      _processing = false;
    }
  }

  // ---------- UI ----------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Inferencer')),
      body: LazyColumn(
        padding: const EdgeInsets.all(12),
        spacing: 12,
        children: [
          PreviewBox(
            detections: _detections,
            inferenceMs: _inferenceMs,
            sourceFps: _sourceFps,
            child: _buildPreviewChild(),
          ),
          ModelSelector(value: _modelUrl, onChanged: _onModelChanged),
          SourceTabs(selected: _source, onChanged: _switchSource),
          if (_source == SourceType.camera)
            CameraPanel(
              running: _cameraRunning,
              conf: _conf,
              status: _status,
              onStartStop: _cameraRunning ? _stopCamera : _startCamera,
              onSwitchCamera: () => _cameraController.switchCamera(),
              onConfChanged: (v) => setState(() => _conf = v),
            )
          else
            RemotePanel(
              connected: _remoteConnected,
              conf: _conf,
              status: _status,
              onConnect: _connectRemote,
              onDisconnect: _disconnectRemote,
              onConfChanged: (v) => setState(() => _conf = v),
            ),
        ],
      ),
    );
  }

  Widget? _buildPreviewChild() {
    if (_source == SourceType.camera && _cameraRunning) {
      return YOLOView(
        modelPath: _modelUrl,
        task: YOLOTask.detect,
        controller: _cameraController,
        lensFacing: _lensFacing,
        onResult: _onCameraResults,
        onPerformanceMetrics: _onCameraMetrics,
        onModelLoad: (path, task) {
          if (mounted) setState(() => _status = 'NPU 就绪');
        },
        onModelError: (error, path, task) {
          if (mounted) setState(() => _status = '模型加载失败: $error');
        },
      );
    }
    if (_source == SourceType.remote && _remoteFrame != null) {
      return Image.memory(_remoteFrame!, fit: BoxFit.fill, gaplessPlayback: true);
    }
    return null;
  }
}

/// 极简 LazyColumn 包装（避免每次手动 ListView 样板）。
class LazyColumn extends StatelessWidget {
  final List<Widget> children;
  final EdgeInsetsGeometry padding;
  final double spacing;

  const LazyColumn({
    super.key,
    required this.children,
    this.padding = EdgeInsets.zero,
    this.spacing = 12,
  });

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      padding: padding,
      itemCount: children.length,
      itemBuilder: (context, index) => Padding(
        padding: EdgeInsets.only(bottom: index == children.length - 1 ? 0 : spacing),
        child: children[index],
      ),
    );
  }
}
