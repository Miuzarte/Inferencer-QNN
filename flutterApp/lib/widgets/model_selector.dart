import 'package:flutter/material.dart';

import '../config.dart';

class ModelOption {
  const ModelOption(this.label, this.url);

  final String label;
  final String url;
}

/// 官方 v0.6.6 两个 HTP 架构的 QNN detect 模型。
const List<ModelOption> kQnnModelOptions = [
  ModelOption('HTP v73（骁龙 8 Gen 2+）', kQnnDetectModelUrl),
  ModelOption('HTP v81（骁龙 8 Elite Gen 5）', kQnnDetectModelUrlV81),
];

class ModelSelector extends StatelessWidget {
  final String value;
  final ValueChanged<String> onChanged;

  const ModelSelector({
    super.key,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return DropdownButtonFormField<String>(
      initialValue: value,
      decoration: const InputDecoration(
        labelText: 'QNN 模型',
        border: OutlineInputBorder(),
        isDense: true,
      ),
      items: [
        for (final option in kQnnModelOptions)
          DropdownMenuItem(value: option.url, child: Text(option.label)),
      ],
      onChanged: (v) {
        if (v != null) onChanged(v);
      },
    );
  }
}
