import 'package:ultralytics_yolo/models/yolo_result.dart';

/// 归一化 (0..1) 检测框，与 GoCVStreamer 协议一致。
class DetBox {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double score;
  final int classIndex;
  final String className;

  const DetBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.score,
    required this.classIndex,
    required this.className,
  });

  factory DetBox.fromYolo(YOLOResult r) => DetBox(
        x1: r.normalizedBox.left,
        y1: r.normalizedBox.top,
        x2: r.normalizedBox.right,
        y2: r.normalizedBox.bottom,
        score: r.confidence,
        classIndex: r.classIndex,
        className: r.className,
      );

  /// 从官方 predict 返回的 detection map 解析（YOLOResult.toMap 格式）。
  factory DetBox.fromMap(Map<dynamic, dynamic> m) {
    final box = Map<dynamic, dynamic>.from(m['normalizedBox'] as Map? ?? const {});
    final name = m['className'] as String? ?? '';
    final cls = (m['classIndex'] as num?)?.toInt() ?? cocoIndex(name);
    return DetBox(
      x1: (box['left'] as num?)?.toDouble() ?? 0,
      y1: (box['top'] as num?)?.toDouble() ?? 0,
      x2: (box['right'] as num?)?.toDouble() ?? 0,
      y2: (box['bottom'] as num?)?.toDouble() ?? 0,
      score: (m['confidence'] as num?)?.toDouble() ?? 0,
      classIndex: cls,
      className: name,
    );
  }

  /// 从官方 predict 返回的原始 boxes 元素解析。
  /// 注意：插件 detections 里的 classIndex 写死为 0，必须用 boxes 才能拿到真实类别。
  factory DetBox.fromBoxMap(Map<dynamic, dynamic> m) {
    final cls = classIndexFromBoxMap(m) ?? 0;
    return DetBox(
      x1: _asDouble(m['x1_norm']) ?? 0,
      y1: _asDouble(m['y1_norm']) ?? 0,
      x2: _asDouble(m['x2_norm']) ?? 0,
      y2: _asDouble(m['y2_norm']) ?? 0,
      score: _asDouble(m['confidence']) ?? 0,
      classIndex: cls,
      className: classNameFor(cls),
    );
  }

  /// 从原始 boxes 元素读取类别 ID。
  /// 原生 class 字段是 String 类名（如 "person"），也可能是数字字符串/数字。
  static int? classIndexFromBoxMap(Map<dynamic, dynamic> m) {
    final v = m['class'];
    if (v is num) return v.toInt();
    if (v is String) {
      final trimmed = v.trim();
      final n = int.tryParse(trimmed);
      if (n != null) return n;
      final i = _coco.indexOf(trimmed);
      return i < 0 ? null : i;
    }
    return null;
  }

  static String classNameFor(int classIndex) =>
      (classIndex >= 0 && classIndex < _coco.length) ? _coco[classIndex] : '';

  static double? _asDouble(dynamic v) {
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v.trim());
    return null;
  }

  Map<String, dynamic> toJson() => {
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'score': score,
        'class': classIndex,
        'class_name': className,
      };

  static const List<String> _coco = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
  ];

  static int cocoIndex(String name) {
    final i = _coco.indexOf(name);
    return i < 0 ? -1 : i;
  }
}
