import 'package:flutter/material.dart';

import '../config.dart';

class RemotePanel extends StatefulWidget {
  final bool connected;
  final double conf;
  final String? status;
  final void Function(String url) onConnect;
  final VoidCallback onDisconnect;
  final ValueChanged<double> onConfChanged;

  const RemotePanel({
    super.key,
    required this.connected,
    required this.conf,
    required this.status,
    required this.onConnect,
    required this.onDisconnect,
    required this.onConfChanged,
  });

  @override
  State<RemotePanel> createState() => _RemotePanelState();
}

class _RemotePanelState extends State<RemotePanel> {
  late final TextEditingController _url =
      TextEditingController(text: kDefaultWsUrl);

  @override
  void dispose() {
    _url.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        TextField(
          controller: _url,
          decoration: const InputDecoration(
            labelText: 'WebSocket URL',
            border: OutlineInputBorder(),
            isDense: true,
          ),
        ),
        const SizedBox(height: 8),
        FilledButton(
          onPressed: widget.connected ? widget.onDisconnect : () => widget.onConnect(_url.text.trim()),
          child: Text(widget.connected ? 'Disconnect' : 'Connect'),
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            const Text('Conf'),
            Expanded(
              child: Slider(
                value: widget.conf,
                min: 0.05,
                max: 0.95,
                onChanged: widget.onConfChanged,
              ),
            ),
            Text(widget.conf.toStringAsFixed(2)),
          ],
        ),
        if (widget.status != null)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(widget.status!,
                style: const TextStyle(color: Colors.white70, fontSize: 13)),
          ),
      ],
    );
  }
}
