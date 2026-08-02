import 'package:flutter/material.dart';

enum SourceType { camera, remote }

/// 摄像头源 / 远程源切换。
class SourceTabs extends StatelessWidget {
  final SourceType selected;
  final ValueChanged<SourceType> onChanged;

  const SourceTabs({super.key, required this.selected, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: Colors.white10,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          _tab(SourceType.camera, 'Camera'),
          _tab(SourceType.remote, 'Remote'),
        ],
      ),
    );
  }

  Widget _tab(SourceType type, String label) {
    final active = selected == type;
    return Expanded(
      child: Material(
        color: active ? Colors.white : Colors.transparent,
        borderRadius: BorderRadius.circular(8),
        child: InkWell(
          borderRadius: BorderRadius.circular(8),
          onTap: () => onChanged(type),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 10),
            child: Text(
              label,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: active ? Colors.black : Colors.white70,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
