package io.github.miuzarte.inferencer.ui.components

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import io.github.miuzarte.inferencer.model.ModelOption
import io.github.miuzarte.inferencer.model.ModelRepository
import top.yukonga.miuix.kmp.preference.OverlayDropdownPreference

@Composable
fun ModelSelector(
    selected: ModelOption,
    enabled: Boolean,
    onSelected: (ModelOption) -> Unit,
    modifier: Modifier = Modifier,
) {
    val models = ModelRepository.models
    val selectedIndex = models.indexOfFirst { it.id == selected.id }.coerceAtLeast(0)
    OverlayDropdownPreference(
        items = models.map { it.label },
        selectedIndex = selectedIndex,
        title = "模型",
        enabled = enabled,
        onSelectedIndexChange = { index -> onSelected(models[index]) },
        modifier = modifier.fillMaxWidth(),
    )
}
