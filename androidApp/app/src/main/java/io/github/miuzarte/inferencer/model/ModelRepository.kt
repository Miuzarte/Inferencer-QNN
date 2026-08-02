package io.github.miuzarte.inferencer.model

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

data class ModelOption(
    val id: String,
    val label: String,
    val url: String,
)

/** 官方 yolo-flutter-app v0.6.6 release 的 QNN context binary，按 HTP arch 区分。 */
object ModelRepository {
    val models = listOf(
        ModelOption(
            id = "v73",
            label = "HTP v73（骁龙 8 Gen 2+）",
            url = "https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/yolo26n_v73_qnn.onnx",
        ),
        ModelOption(
            id = "v81",
            label = "HTP v81（骁龙 8 Elite Gen 5）",
            url = "https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/yolo26n_v81_qnn.onnx",
        ),
    )

    fun modelFile(context: Context, option: ModelOption): File =
        File(context.filesDir, "models/${option.url.substringAfterLast('/')}")

    /** 已存在则直接返回；否则流式下载到 filesDir/models/（带进度回调 0..1）。 */
    suspend fun ensureModel(
        context: Context,
        option: ModelOption,
        onProgress: (Float) -> Unit = {},
    ): File = withContext(Dispatchers.IO) {
        val target = modelFile(context, option)
        if (target.exists() && target.length() > 0) return@withContext target

        target.parentFile?.mkdirs()
        val tmp = File(target.parentFile, "${target.name}.download")
        val client = OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()
        try {
            val request = Request.Builder().url(option.url).build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IOException("HTTP ${response.code} for ${option.url}")
                }
                val total = response.body?.contentLength() ?: -1L
                var received = 0L
                response.body?.byteStream()?.use { input ->
                    tmp.outputStream().buffered().use { sink ->
                        val buf = ByteArray(64 * 1024)
                        while (true) {
                            val n = input.read(buf)
                            if (n < 0) break
                            sink.write(buf, 0, n)
                            received += n
                            if (total > 0) {
                                onProgress((received.toFloat() / total).coerceIn(0f, 0.99f))
                            }
                        }
                    }
                }
            }
            if (target.exists()) target.delete()
            if (!tmp.renameTo(target)) throw IOException("Failed to move $tmp to $target")
            onProgress(1f)
            target
        } catch (e: Exception) {
            tmp.delete()
            throw e
        } finally {
            client.dispatcher.executorService.shutdown()
        }
    }
}
