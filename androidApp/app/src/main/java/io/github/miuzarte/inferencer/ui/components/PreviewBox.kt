package io.github.miuzarte.inferencer.ui.components

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.miuzarte.inferencer.inference.DetBox
import top.yukonga.miuix.kmp.basic.Text

@Composable
fun PreviewBox(
    bitmap: Bitmap?,
    detections: List<DetBox>,
    inferenceMs: Float,
    sourceFps: Float,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .background(Color.DarkGray),
        contentAlignment = Alignment.Center,
    ) {
        if (bitmap != null) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = null,
                contentScale = ContentScale.FillBounds,
                modifier = Modifier.fillMaxSize(),
            )
            DetectionCanvas(
                detections = detections,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            Text(text = "预览", color = Color.White, fontSize = 16.sp)
        }

        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(8.dp),
        ) {
            Text(
                text = "INF: %.1fms".format(inferenceMs),
                color = Color.White,
                fontSize = 12.sp,
                modifier = Modifier
                    .background(Color(0xAA000000))
                    .padding(horizontal = 4.dp, vertical = 2.dp),
            )
            Text(
                text = "SRC: %.1ffps".format(sourceFps),
                color = Color.White,
                fontSize = 12.sp,
                modifier = Modifier
                    .background(Color(0xAA000000))
                    .padding(horizontal = 4.dp, vertical = 2.dp),
            )
        }
    }
}
