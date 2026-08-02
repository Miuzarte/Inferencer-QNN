import 'package:flutter/material.dart';

import '../models/det_box.dart';
import 'detection_painter.dart';

/// 1:1 预览框：内容（相机/远程帧）+ 检测框叠加 + INF/SRC 状态标签。
class PreviewBox extends StatelessWidget {
  final Widget? child;
  final List<DetBox> detections;
  final double inferenceMs;
  final double sourceFps;

  const PreviewBox({
    super.key,
    this.child,
    this.detections = const [],
    this.inferenceMs = 0,
    this.sourceFps = 0,
  });

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1,
      child: Container(
        color: Colors.black,
        child: Stack(
          fit: StackFit.expand,
          children: [
            ?child,
            if (child != null && detections.isNotEmpty)
              CustomPaint(painter: DetectionPainter(detections)),
            Positioned(
              top: 8,
              left: 8,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _badge('INF: ${inferenceMs.toStringAsFixed(1)}ms'),
                  const SizedBox(height: 2),
                  _badge('SRC: ${sourceFps.toStringAsFixed(1)}fps'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _badge(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
      color: const Color(0xAA000000),
      child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 12)),
    );
  }
}
