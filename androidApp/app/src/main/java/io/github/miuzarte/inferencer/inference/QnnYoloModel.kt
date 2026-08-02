package io.github.miuzarte.inferencer.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.system.Os
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * QNN HTP 推理模型，初始化流程与官方 yolo-flutter-app 的 OrtQnnModel.kt 一致：
 * ADSP_LIBRARY_PATH 指向 APK 自带 QAIRT Skel + addQnn(backend_path=libQnnHtp.so, burst)。
 */
class QnnYoloModel(
    private val context: Context,
    modelPath: String,
    confThreshold: Float = 0.45f,
) : AutoCloseable {
    companion object {
        private const val TAG = "QnnYoloModel"
        private const val INPUT_MEAN = 0f
        private const val INPUT_STD = 255f
        private const val MODEL_SIZE = 640
    }

    @Volatile
    var confThreshold: Float = confThreshold

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val inputShape: LongArray
    private val inputUsesNchw: Boolean
    private val outFeatures: Int
    private val outAnchors: Int
    private val endToEnd: Boolean
    private val floatInput: FloatArray
    private val pixels: IntArray

    init {
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        Log.i(
            TAG,
            "QnnYoloModel init: modelPath=$modelPath thread=${Thread.currentThread().name} " +
                "nativeLibDir=$nativeLibDir ortv=${env.getVersion()}",
        )
        // fastrpc 优先加载 APK 自带 QAIRT Skel，避免与 /vendor 版本不匹配
        try {
            Os.setenv(
                "ADSP_LIBRARY_PATH",
                nativeLibDir + ";/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp",
                true,
            )
        } catch (t: Throwable) {
            Log.w(TAG, "Could not set ADSP_LIBRARY_PATH: ${t.message}")
        }
        Log.i(
            TAG,
            "ADSP_LIBRARY_PATH=${System.getenv("ADSP_LIBRARY_PATH")} " +
                "nativeLibDir=$nativeLibDir " +
                "skelV73=${File(nativeLibDir, "libQnnHtpV73Skel.so").exists()}",
        )

        val options = OrtSession.SessionOptions().apply {
            addQnn(
                mapOf(
                    "backend_path" to "libQnnHtp.so",
                    "htp_performance_mode" to "burst",
                ),
            )
            setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
        }
        session = try {
            env.createSession(modelPath, options)
        } catch (e: Throwable) {
            try {
                val maps = File("/proc/self/maps")
                    .readLines()
                    .filter {
                        it.contains("Qnn", ignoreCase = true) ||
                            it.contains("onnx", ignoreCase = true) ||
                            it.contains("cdsprpc", ignoreCase = true)
                    }
                Log.e(TAG, "native maps:\n" + maps.joinToString("\n"))
            } catch (_: Throwable) {
            }
            throw e
        }

        inputName = session.inputNames.first()
        val info = session.inputInfo.getValue(inputName).info as TensorInfo
        inputShape = info.shape
        require(inputShape.size == 4) { "Expected a [1,3,H,W] or [1,H,W,3] input, got ${inputShape.toList()}" }
        val inputIsNhwc = inputShape[3] == 3L && inputShape[1] != 3L
        inputUsesNchw = !inputIsNhwc
        val height = (if (inputIsNhwc) inputShape[1] else inputShape[2]).toInt()
        val width = (if (inputIsNhwc) inputShape[2] else inputShape[3]).toInt()
        require(height == MODEL_SIZE && width == MODEL_SIZE) {
            "Model input must be 640x640, got ${width}x$height"
        }

        val outName = session.outputNames.first()
        val outShape = (session.outputInfo.getValue(outName).info as TensorInfo).shape
        outFeatures = when {
            outShape.size >= 3 -> outShape[1].toInt()
            outShape.size == 2 -> outShape[0].toInt()
            else -> 0
        }
        outAnchors = when {
            outShape.size >= 3 -> outShape[2].toInt()
            outShape.size == 2 -> outShape[1].toInt()
            else -> 0
        }
        endToEnd = outAnchors >= 6 && outAnchors < outFeatures

        floatInput = FloatArray(width * height * 3)
        pixels = IntArray(width * height)

        Log.i(
            TAG,
            "ONNX Runtime QNN session on NPU; inputDims=[1, 640, 640, 3] " +
                "outputDims=${outShape.toList()} endToEnd=$endToEnd",
        )
    }

    /** 对 [bitmap] 做 letterbox 后推理，返回相对原图的归一化框。 */
    fun detect(bitmap: Bitmap): List<DetBox> {
        val srcW = bitmap.width
        val srcH = bitmap.height
        val scale = min(MODEL_SIZE.toFloat() / srcW, MODEL_SIZE.toFloat() / srcH)
        val resizedW = (srcW * scale).roundToInt()
        val resizedH = (srcH * scale).roundToInt()
        val padX = (MODEL_SIZE - resizedW) / 2
        val padY = (MODEL_SIZE - resizedH) / 2

        val scaled = if (resizedW == srcW && resizedH == srcH) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, resizedW, resizedH, true)
        }
        scaled.getPixels(pixels, 0, resizedW, 0, 0, resizedW, resizedH)

        floatInput.fill(0f)
        val invStd = 1f / INPUT_STD
        val plane = MODEL_SIZE * MODEL_SIZE
        if (inputUsesNchw) {
            for (y in 0 until resizedH) {
                for (x in 0 until resizedW) {
                    val p = pixels[y * resizedW + x]
                    val idx = (y + padY) * MODEL_SIZE + (x + padX)
                    floatInput[idx] = (((p shr 16) and 0xFF) - INPUT_MEAN) * invStd
                    floatInput[plane + idx] = (((p shr 8) and 0xFF) - INPUT_MEAN) * invStd
                    floatInput[2 * plane + idx] = ((p and 0xFF) - INPUT_MEAN) * invStd
                }
            }
        } else {
            for (y in 0 until resizedH) {
                for (x in 0 until resizedW) {
                    val p = pixels[y * resizedW + x]
                    val o = ((y + padY) * MODEL_SIZE + (x + padX)) * 3
                    floatInput[o] = (((p shr 16) and 0xFF) - INPUT_MEAN) * invStd
                    floatInput[o + 1] = (((p shr 8) and 0xFF) - INPUT_MEAN) * invStd
                    floatInput[o + 2] = ((p and 0xFF) - INPUT_MEAN) * invStd
                }
            }
        }
        if (scaled !== bitmap) scaled.recycle()

        OnnxTensor.createTensor(env, FloatBuffer.wrap(floatInput), inputShape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { results ->
                val outTensor = results.get(session.outputNames.first()).get() as OnnxTensor
                val out = FloatArray(outTensor.floatBuffer.remaining())
                outTensor.floatBuffer.get(out)
                return postprocess(out, srcW, srcH, scale, padX.toFloat(), padY.toFloat())
            }
        }
    }

    private fun postprocess(
        out: FloatArray,
        srcW: Int,
        srcH: Int,
        scale: Float,
        padX: Float,
        padY: Float,
    ): List<DetBox> {
        val xNorm: (Float) -> Float = { v ->
            ((v * MODEL_SIZE - padX) / scale / srcW).coerceIn(0f, 1f)
        }
        val yNorm: (Float) -> Float = { v ->
            ((v * MODEL_SIZE - padY) / scale / srcH).coerceIn(0f, 1f)
        }
        val result = mutableListOf<DetBox>()

        if (endToEnd) {
            // [rows, 6]：x1, y1, x2, y2, score, class
            for (i in 0 until outFeatures) {
                val off = i * outAnchors
                val score = out[off + 4]
                if (score < confThreshold) continue
                val cls = out[off + 5].toInt()
                if (cls < 0 || cls >= CocoClasses.NAMES.size) continue
                result.add(
                    DetBox(
                        xNorm(out[off]),
                        yNorm(out[off + 1]),
                        xNorm(out[off + 2]),
                        yNorm(out[off + 3]),
                        score,
                        cls,
                        CocoClasses.NAMES[cls],
                    ),
                )
            }
        } else {
            // [features, anchors]：feature-major，列 = anchor
            val w = outAnchors
            val numClasses = (outFeatures - 4).coerceAtLeast(0)
            val candidates = mutableListOf<Candidate>()
            for (i in 0 until w) {
                var cls = 0
                var best = -Float.MAX_VALUE
                for (c in 0 until numClasses) {
                    val score = out[(c + 4) * w + i]
                    if (score > best) {
                        best = score
                        cls = c
                    }
                }
                if (best < confThreshold || cls >= CocoClasses.NAMES.size) continue
                val cx = out[i]
                val cy = out[w + i]
                val bw = out[2 * w + i]
                val bh = out[3 * w + i]
                candidates.add(
                    Candidate(
                        cx - bw / 2f, cy - bh / 2f,
                        cx + bw / 2f, cy + bh / 2f,
                        best, cls,
                    ),
                )
            }
            for (d in nms(candidates)) {
                result.add(
                    DetBox(
                        xNorm(d.x1), yNorm(d.y1),
                        xNorm(d.x2), yNorm(d.y2),
                        d.score, d.cls, CocoClasses.NAMES[d.cls],
                    ),
                )
            }
        }
        return result
    }

    private class Candidate(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val score: Float, val cls: Int,
    )

    /** 与官方 native-lib.cpp 一致：全局 NMS（不按类别分组），按置信度降序。 */
    private fun nms(candidates: List<Candidate>, iouThresh: Float = 0.7f): List<Candidate> {
        val sorted = candidates.sortedByDescending { it.score }
        val picked = mutableListOf<Candidate>()
        for (a in sorted) {
            var keep = true
            for (b in picked) {
                val interX = minOf(a.x2, b.x2) - maxOf(a.x1, b.x1)
                val interY = minOf(a.y2, b.y2) - maxOf(a.y1, b.y1)
                val inter = maxOf(0f, interX) * maxOf(0f, interY)
                val union =
                    (a.x2 - a.x1) * (a.y2 - a.y1) +
                        (b.x2 - b.x1) * (b.y2 - b.y1) -
                        inter
                if (union > 0f && inter / union > iouThresh) {
                    keep = false
                    break
                }
            }
            if (keep) picked.add(a)
        }
        return picked
    }

    override fun close() {
        try {
            session.close()
        } catch (_: Throwable) {
            // best-effort
        }
    }
}
