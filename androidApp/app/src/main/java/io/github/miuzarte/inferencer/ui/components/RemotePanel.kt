package io.github.miuzarte.inferencer.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import top.yukonga.miuix.kmp.basic.Button
import top.yukonga.miuix.kmp.basic.LinearProgressIndicator
import top.yukonga.miuix.kmp.basic.Slider
import top.yukonga.miuix.kmp.basic.Text
import top.yukonga.miuix.kmp.basic.TextField

@Composable
fun RemotePanel(
    url: String,
    onUrlChange: (String) -> Unit,
    connected: Boolean,
    downloading: Boolean,
    progress: Float,
    status: String?,
    conf: Float,
    onConfChange: (Float) -> Unit,
    onConnect: () -> Unit,
    onDisconnect: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        TextField(
            value = url,
            onValueChange = onUrlChange,
            label = "WebSocket URL",
            useLabelAsPlaceholder = true,
            singleLine = true,
            enabled = !connected && !downloading,
            modifier = Modifier.fillMaxWidth(),
        )
        Button(
            onClick = { if (connected) onDisconnect() else onConnect() },
            enabled = !downloading,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(text = if (connected) "Disconnect" else "Connect")
        }
        if (downloading) {
            LinearProgressIndicator(
                progress = progress,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        status?.let {
            Text(text = it, fontSize = 12.sp)
        }
        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
            Text(text = "Conf", fontSize = 12.sp)
            Slider(
                value = conf,
                onValueChange = onConfChange,
                valueRange = 0.05f..0.95f,
                modifier = Modifier.weight(1f),
            )
        }
    }
}
