import 'package:flutter/material.dart';

import 'home_screen.dart';

void main() {
  runApp(const InferencerApp());
}

class InferencerApp extends StatelessWidget {
  const InferencerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Inferencer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}
