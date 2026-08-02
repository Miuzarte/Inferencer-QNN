package io.github.miuzarte.inferencer

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import io.github.miuzarte.inferencer.ui.MainScreen
import io.github.miuzarte.inferencer.ui.theme.InferencerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            InferencerTheme {
                MainScreen()
            }
        }
    }
}
