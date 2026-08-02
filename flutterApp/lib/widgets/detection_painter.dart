import 'package:flutter/material.dart';

import '../models/det_box.dart';

/// 在预览框上绘制归一化检测框与标签。
class DetectionPainter extends CustomPainter {
  final List<DetBox> detections;

  DetectionPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final border = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.greenAccent;
    final labelBg = Paint()..color = Colors.greenAccent;
    final textStyle = TextStyle(
      color: Colors.black,
      fontSize: 12,
      fontWeight: FontWeight.w600,
    );
    for (final d in detections) {
      final rect = Rect.fromLTRB(
        d.x1 * size.width,
        d.y1 * size.height,
        d.x2 * size.width,
        d.y2 * size.height,
      );
      canvas.drawRect(rect, border);
      final label = '${d.className} ${(d.score * 100).toStringAsFixed(0)}%';
      final tp = TextPainter(
        text: TextSpan(text: label, style: textStyle),
        textDirection: TextDirection.ltr,
      )..layout();
      final top = (rect.top - tp.height).clamp(0.0, size.height - tp.height);
      canvas.drawRect(
        Rect.fromLTWH(rect.left, top, tp.width, tp.height),
        labelBg,
      );
      tp.paint(canvas, Offset(rect.left, top));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) =>
      oldDelegate.detections != detections;
}
