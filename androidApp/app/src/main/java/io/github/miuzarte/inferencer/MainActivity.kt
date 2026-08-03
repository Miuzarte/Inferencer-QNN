package io.github.miuzarte.inferencer

import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.navigationevent.OnBackInvokedDefaultInput
import androidx.navigationevent.compose.LocalNavigationEventDispatcherOwner
import androidx.navigationevent.compose.rememberNavigationEventDispatcherOwner
import io.github.miuzarte.inferencer.ui.MainScreen
import io.github.miuzarte.inferencer.ui.theme.InferencerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            // miuix 0.9.3 的 Popup/Dialog 内部使用 NavigationBackHandler，
            // 必须在组合树根部提供 NavigationEventDispatcherOwner，否则打开 Dropdown 会崩溃。
            val owner = rememberNavigationEventDispatcherOwner(parent = null)
            val backInput = remember {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    OnBackInvokedDefaultInput(onBackInvokedDispatcher)
                } else {
                    null
                }
            }
            DisposableEffect(owner, backInput) {
                if (backInput != null) {
                    owner.navigationEventDispatcher.addInput(backInput)
                }
                onDispose {
                    if (backInput != null) {
                        owner.navigationEventDispatcher.removeInput(backInput)
                    }
                }
            }
            CompositionLocalProvider(LocalNavigationEventDispatcherOwner provides owner) {
                InferencerTheme {
                    MainScreen()
                }
            }
        }
    }
}
