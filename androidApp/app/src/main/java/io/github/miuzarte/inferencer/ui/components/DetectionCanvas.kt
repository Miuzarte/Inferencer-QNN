package io.github.miuzarte.inferencer.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import io.github.miuzarte.inferencer.inference.DetBox

@Composable
fun DetectionCanvas(
    detections: List<DetBox>,
    modifier: Modifier = Modifier,
) {
    val boxColor = Color(0xFF00FF00)
    val bgColor = Color(0xAA000000)

    Canvas(modifier = modifier) {
        val cw = size.width
        val ch = size.height

        for (d in detections) {
            val left = d.x1 * cw
            val top = d.y1 * ch
            val right = d.x2 * cw
            val bottom = d.y2 * ch

            drawRect(
                color = boxColor,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                style = Stroke(width = 2f),
            )

            val label = "${d.className} %.2f".format(d.score)
            val paint = android.graphics.Paint().apply {
                color = android.graphics.Color.WHITE
                textSize = 26f
                isAntiAlias = true
            }
            val fm = paint.fontMetrics
            val textW = paint.measureText(label)
            val textH = fm.descent - fm.ascent

            val labelX = left
            val labelY = (top - textH).coerceAtLeast(0f)

            drawRect(
                color = bgColor,
                topLeft = Offset(labelX, labelY),
                size = Size(textW + 6f, textH + 2f),
            )
            drawContext.canvas.nativeCanvas.drawText(
                label,
                labelX + 3f,
                labelY - fm.ascent + 1f,
                paint,
            )
        }
    }
}
