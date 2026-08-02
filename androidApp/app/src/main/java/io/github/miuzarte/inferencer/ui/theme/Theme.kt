package io.github.miuzarte.inferencer.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import top.yukonga.miuix.kmp.theme.ColorSchemeMode
import top.yukonga.miuix.kmp.theme.MiuixTheme
import top.yukonga.miuix.kmp.theme.ThemeController

@Composable
fun InferencerTheme(content: @Composable () -> Unit) {
    val darkTheme = isSystemInDarkTheme()
    val mode = if (darkTheme) ColorSchemeMode.Dark else ColorSchemeMode.Light
    MiuixTheme(
        controller = ThemeController(colorSchemeMode = mode),
        content = content,
    )
}
