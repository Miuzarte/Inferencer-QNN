import 'package:flutter/material.dart';

class CameraPanel extends StatelessWidget {
  final bool running;
  final double conf;
  final String? status;
  final VoidCallback onStartStop;
  final VoidCallback? onSwitchCamera;
  final ValueChanged<double> onConfChanged;

  const CameraPanel({
    super.key,
    required this.running,
    required this.conf,
    required this.status,
    required this.onStartStop,
    this.onSwitchCamera,
    required this.onConfChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        FilledButton(
          onPressed: onStartStop,
          child: Text(running ? 'Stop' : 'Start'),
        ),
        const SizedBox(height: 4),
        OutlinedButton.icon(
          onPressed: running ? onSwitchCamera : null,
          icon: const Icon(Icons.cameraswitch),
          label: const Text('切换镜头'),
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            const Text('Conf'),
            Expanded(
              child: Slider(
                value: conf,
                min: 0.05,
                max: 0.95,
                onChanged: onConfChanged,
              ),
            ),
            Text(conf.toStringAsFixed(2)),
          ],
        ),
        if (status != null)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(status!, style: const TextStyle(color: Colors.white70, fontSize: 13)),
          ),
      ],
    );
  }
}
