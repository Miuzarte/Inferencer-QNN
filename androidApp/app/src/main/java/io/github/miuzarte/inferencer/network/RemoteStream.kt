package io.github.miuzarte.inferencer.network

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import io.github.miuzarte.inferencer.inference.DetBox
import io.github.miuzarte.inferencer.inference.QnnYoloModel
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * GoCVStreamer WebSocket 客户端。
 * 收 binary `[4B frame_id LE][JPEG]`，回
 * `{"frame_id":N,"detections":[...],"inference_ms":M}`。
 */
class RemoteStream(
    private val model: QnnYoloModel,
    private val onOpen: () -> Unit,
    private val onClose: (String) -> Unit,
    private val onFrame: (frameId: Int, bitmap: Bitmap) -> Unit,
    private val onDetections: (List<DetBox>) -> Unit,
    private val onInferenceMs: (Double) -> Unit,
    private val onFps: (Float) -> Unit,
) {
    companion object {
        private const val TAG = "RemoteStream"
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private val executor = Executors.newSingleThreadExecutor()
    private val processing = AtomicBoolean(false)

    @Volatile
    private var webSocket: WebSocket? = null

    @Volatile
    private var closed = false

    @Volatile
    var connected = false
        private set

    private var frameCount = 0
    private var fpsStartNanos = 0L

    fun connect(url: String) {
        if (connected || closed) return
        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(request, listener)
    }

    fun disconnect() {
        closed = true
        connected = false
        webSocket?.close(1000, "bye")
        webSocket = null
        executor.shutdownNow()
        client.dispatcher.executorService.shutdown()
    }

    private val listener = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            connected = true
            onOpen()
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
            // disconnect() 后 executor 已关闭，OkHttp 回调可能还会带着残留消息过来，
            // 此时直接丢弃，避免 RejectedExecutionException 污染连接状态。
            if (closed || executor.isShutdown) return
            // 跳帧：上一帧还在推理时直接丢弃新帧，避免堆积
            if (processing.compareAndSet(false, true)) {
                executor.execute { handleFrame(bytes.toByteArray()) }
            }
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            webSocket.close(code, reason)
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            connected = false
            onClose("连接已关闭")
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            connected = false
            onClose("连接失败: ${t.message}")
        }
    }

    private fun handleFrame(data: ByteArray) {
        try {
            if (data.size < 5) return
            val frameId = ByteBuffer.wrap(data, 0, 4).order(ByteOrder.LITTLE_ENDIAN).int
            val jpeg = data.copyOfRange(4, data.size)

            val startNanos = System.nanoTime()
            val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
            if (bitmap == null) return
            val dets = model.detect(bitmap)
            val ms = (System.nanoTime() - startNanos) / 1_000_000.0

            val json = JSONObject().apply {
                put("frame_id", frameId)
                put("detections", JSONArray().apply { dets.forEach { put(it.toJson()) } })
                put("inference_ms", Math.round(ms * 100) / 100.0)
            }
            webSocket?.send(json.toString())

            Log.v(TAG, "dets=${dets.size} frame=$frameId infer=%.1fms".format(ms))
            onFrame(frameId, bitmap)
            onDetections(dets)
            onInferenceMs(ms)

            frameCount++
            val now = System.nanoTime()
            if (fpsStartNanos == 0L) fpsStartNanos = now
            val elapsed = (now - fpsStartNanos) / 1e9f
            if (elapsed >= 1f) {
                onFps(frameCount / elapsed)
                frameCount = 0
                fpsStartNanos = now
            }
        } catch (e: Exception) {
            Log.e(TAG, "handleFrame failed", e)
        } finally {
            processing.set(false)
        }
    }
}
