package io.github.miuzarte.inferencer.ui.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.miuzarte.inferencer.inference.DetBox
import io.github.miuzarte.inferencer.inference.QnnYoloModel
import io.github.miuzarte.inferencer.model.ModelOption
import io.github.miuzarte.inferencer.model.ModelRepository
import io.github.miuzarte.inferencer.network.RemoteStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class SourceType { Camera, Remote }

class DetectorViewModel(application: Application) : AndroidViewModel(application) {
    private val context = application.applicationContext
    private val tag = "DetectorViewModel"

    private var model: QnnYoloModel? = null
    private var stream: RemoteStream? = null

    val selectedModel = MutableStateFlow(ModelRepository.models.first())
    val remoteUrl = MutableStateFlow("ws://192.168.1.100:9090/stream")
    val confThreshold = MutableStateFlow(0.45f)
    val sourceType = MutableStateFlow(SourceType.Remote)

    private val _status = MutableStateFlow<String?>(null)
    val status: StateFlow<String?> = _status.asStateFlow()

    private val _connected = MutableStateFlow(false)
    val connected: StateFlow<Boolean> = _connected.asStateFlow()

    private val _downloading = MutableStateFlow(false)
    val downloading: StateFlow<Boolean> = _downloading.asStateFlow()

    private val _downloadProgress = MutableStateFlow(0f)
    val downloadProgress: StateFlow<Float> = _downloadProgress.asStateFlow()

    private val _displayBitmap = MutableStateFlow<Bitmap?>(null)
    val displayBitmap: StateFlow<Bitmap?> = _displayBitmap.asStateFlow()

    private val _detections = MutableStateFlow<List<DetBox>>(emptyList())
    val detections: StateFlow<List<DetBox>> = _detections.asStateFlow()

    private val _inferenceMs = MutableStateFlow(0f)
    val inferenceMs: StateFlow<Float> = _inferenceMs.asStateFlow()

    private val _sourceFps = MutableStateFlow(0f)
    val sourceFps: StateFlow<Float> = _sourceFps.asStateFlow()

    fun setModel(option: ModelOption) {
        if (option == selectedModel.value) return
        disconnect()
        selectedModel.value = option
        _status.value = "模型已切换，点击 Connect 生效"
    }

    fun setConf(conf: Float) {
        confThreshold.value = conf
        model?.confThreshold = conf
    }

    fun setRemoteUrl(url: String) {
        remoteUrl.value = url
    }

    fun selectSource(type: SourceType) {
        if (type == sourceType.value) return
        disconnect()
        sourceType.value = type
        clearResults()
    }

    fun connect() {
        if (stream != null || _downloading.value) return
        viewModelScope.launch {
            _downloading.value = true
            _downloadProgress.value = 0f
            _status.value = "加载模型…"
            try {
                val option = selectedModel.value
                val file = ModelRepository.ensureModel(context, option) { p ->
                    _downloadProgress.value = p
                }
                // 与官方插件一致：QNN session 创建在后台线程（flutterApp 的 CameraX
                // analyzer 线程上可成功；主线程上 QNN 的设备创建会 INVALID_CONFIG）。
                val newModel = withContext(Dispatchers.IO) {
                    QnnYoloModel(context, file.absolutePath, confThreshold.value)
                }
                model?.close()
                model = newModel
                _downloading.value = false
                _status.value = "NPU 就绪，连接中…"
                startStream(newModel)
            } catch (e: Exception) {
                Log.e(tag, "model load failed", e)
                _downloading.value = false
                _status.value = "模型加载失败: ${e.message}"
            }
        }
    }

    private fun startStream(m: QnnYoloModel) {
        val s = RemoteStream(
            model = m,
            onOpen = {
                _connected.value = true
                _status.value = "已连接，等待帧…"
            },
            onClose = { msg ->
                _connected.value = false
                stream = null
                _status.value = msg
                clearResults()
            },
            onFrame = { _, bmp ->
                viewModelScope.launch {
                    // 直接投递新解码的 Bitmap：每帧都是新引用，StateFlow 才会通知 UI 更新。
                    // 不能提前 recycle，Image 绘制期间仍持有它（交给 GC 回收）。
                    _displayBitmap.value = bmp
                }
            },
            onDetections = { _detections.value = it },
            onInferenceMs = { _inferenceMs.value = it.toFloat() },
            onFps = { _sourceFps.value = it },
        )
        stream = s
        s.connect(remoteUrl.value)
    }

    fun disconnect() {
        stream?.disconnect()
        stream = null
        _connected.value = false
        _status.value = null
        clearResults()
    }

    private fun clearResults() {
        _detections.value = emptyList()
        _inferenceMs.value = 0f
        _sourceFps.value = 0f
        _displayBitmap.value = null
    }

    override fun onCleared() {
        stream?.disconnect()
        model?.close()
        super.onCleared()
    }
}
