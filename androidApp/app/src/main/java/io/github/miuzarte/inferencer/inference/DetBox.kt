package io.github.miuzarte.inferencer.inference

import org.json.JSONObject

/** 归一化检测框（0..1，相对推理输入原图），与 GoCVStreamer 协议一致。 */
data class DetBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val classId: Int,
    val className: String,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("x1", round4(x1))
        put("y1", round4(y1))
        put("x2", round4(x2))
        put("y2", round4(y2))
        put("score", round4(score))
        put("class", classId)
        put("class_name", className)
    }

    private fun round4(v: Float): Double = Math.round(v * 10000) / 10000.0
}
