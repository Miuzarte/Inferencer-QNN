package io.github.miuzarte.inferencer.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.miuzarte.inferencer.ui.components.CameraPanel
import io.github.miuzarte.inferencer.ui.components.ModelSelector
import io.github.miuzarte.inferencer.ui.components.PreviewBox
import io.github.miuzarte.inferencer.ui.components.RemotePanel
import io.github.miuzarte.inferencer.ui.viewmodel.DetectorViewModel
import io.github.miuzarte.inferencer.ui.viewmodel.SourceType
import top.yukonga.miuix.kmp.basic.Scaffold
import top.yukonga.miuix.kmp.basic.SmallTopAppBar
import top.yukonga.miuix.kmp.basic.TabRow

@Composable
fun MainScreen(viewModel: DetectorViewModel = viewModel()) {
    val displayBitmap by viewModel.displayBitmap.collectAsState()
    val detections by viewModel.detections.collectAsState()
    val inferenceMs by viewModel.inferenceMs.collectAsState()
    val sourceFps by viewModel.sourceFps.collectAsState()
    val status by viewModel.status.collectAsState()
    val connected by viewModel.connected.collectAsState()
    val downloading by viewModel.downloading.collectAsState()
    val downloadProgress by viewModel.downloadProgress.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val conf by viewModel.confThreshold.collectAsState()
    val remoteUrl by viewModel.remoteUrl.collectAsState()
    val sourceType by viewModel.sourceType.collectAsState()

    Scaffold(
        topBar = {
            SmallTopAppBar(title = "Inferencer")
        },
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                PreviewBox(
                    bitmap = displayBitmap,
                    detections = detections,
                    inferenceMs = inferenceMs,
                    sourceFps = sourceFps,
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
            }
            item {
                ModelSelector(
                    selected = selectedModel,
                    enabled = !connected && !downloading,
                    onSelected = viewModel::setModel,
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
            }
            item {
                TabRow(
                    tabs = listOf("Camera", "Remote"),
                    selectedTabIndex = if (sourceType == SourceType.Camera) 0 else 1,
                    onTabSelected = { index ->
                        viewModel.selectSource(
                            if (index == 0) SourceType.Camera else SourceType.Remote
                        )
                    },
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
            }
            item {
                when (sourceType) {
                    SourceType.Camera -> CameraPanel()
                    SourceType.Remote -> RemotePanel(
                        url = remoteUrl,
                        onUrlChange = viewModel::setRemoteUrl,
                        connected = connected,
                        downloading = downloading,
                        progress = downloadProgress,
                        status = status,
                        conf = conf,
                        onConfChange = viewModel::setConf,
                        onConnect = viewModel::connect,
                        onDisconnect = viewModel::disconnect,
                    )
                }
            }
        }
    }
}
