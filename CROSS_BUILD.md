# Inferencer 交叉编译方案

## 架构

```
Termux 原生 (Bionic, root)
├── Go binary (purego 调用 ORT C API)
├── libonnxruntime.so             ← Windows 交叉编译（v1.28.0 + ENABLE_DYNAMIC_PLUGIN_EP=ON）
├── libonnxruntime_providers_qnn.so ← Maven AAR 提取（已是 Bionic ARM64）
├── libQnnHtp.so + QNN 系统库     ← /vendor/lib64/（设备自带）
├── libcdsprpc.so                 ← /vendor/lib64/（设备自带）
└── fastrpc-cdsp                  ← /dev/（Termux root 可访问）
```

## 实施步骤

### 1. 交叉编译 ORT（Windows + NDK）

```powershell
# 克隆 ORT
cd B:\Git\
git clone --depth 1 --branch v1.28.0 https://github.com/microsoft/onnxruntime ort-src
cd ort-src

# 交叉编译 Android ARM64（Bionic）
python tools/ci_build/build.py --update --build --config Release --parallel --android --android_abi arm64-v8a --android_api 27 --android_ndk_path "$env:LOCALAPPDATA\Android\Sdk\ndk\29.0.14206865" --cmake_extra_defines onnxruntime_ENABLE_DYNAMIC_PLUGIN_EP=ON --skip_tests --build_shared_lib
```

产出：`B:\Git\ort-src\build\Android\Release\libonnxruntime.so`（约 20MB，Bionic ARM64）

### 2. 提取 QNN EP 插件（Maven AAR）

```powershell
# 下载 AAR（如未缓存）
# Invoke-WebRequest -Uri "https://repo.maven.apache.org/maven2/com/qualcomm/qti/onnxruntime-android-qnn/2.4.0/onnxruntime-android-qnn-2.4.0.aar" -OutFile "qnn-ep.aar"

# 从本地 Gradle 缓存提取
Expand-Archive "$env:USERPROFILE\.gradle\caches\modules-2\files-2.1\com.qualcomm.qti\onnxruntime-android-qnn\2.4.0\*\onnxruntime-android-qnn-2.4.0.aar" -DestinationPath "qnn-ep\"

# 或直接解压 .aar（.aar 就是 .zip）
Copy-Item "qnn-ep\jni\arm64-v8a\libonnxruntime_providers_qnn.so" -Destination ".\"
```

### 3. 推送文件到设备

```bash
adb push B:\Git\ort-src\build\Android\Release\libonnxruntime.so /sdcard/
adb push libonnxruntime_providers_qnn.so /sdcard/

# Termux 中
sudo cp /sdcard/libonnxruntime.so /data/data/com.termux/files/home/.local/lib/
sudo cp /sdcard/libonnxruntime_providers_qnn.so /data/data/com.termux/files/home/.local/lib/
```

### 4. 代码已就绪

代码已包含：
- ✅ `ortpurego/sesstion.go`: `AppendExecutionProvider` V1 by-name fallback
- ✅ `yolo/detector.go`: V2 优先 → V1 fallback；`disable_cpu_ep_fallback` 仅在 QNN EP 成功挂载后启用
- ✅ `yolo/detector.go` DefaultConfig: 指向 Termux 原生路径
- ✅ `main.go`: `--qnn-htp` flag

无需手动修改，直接 `go build` 即可。

### 5. 编译 & 运行

```bash
# Termux 原生（非 proot）
cd ~/git/Inferencer
go mod tidy
go build -ldflags="-s -w" -o inferencer .

# 运行（必须 root）
sudo LD_LIBRARY_PATH=$HOME/.local/lib:/vendor/lib64 \
  ADSP_LIBRARY_PATH=/vendor/lib/rfsa/adsp \
  ./inferencer --host 192.168.x.x:8080
```

## 风险

| 风险 | 等级 | 对策 |
|------|------|------|
| ORT 交叉编译配置复杂 | 🟡 中 | `--android --android_ndk_path` 已充分测试 |
| Termux root 下 QNN 初始化是否受 Android DSP HAL 拦截 | 🔴 高 | 无解，只能测试 |
| 模型 EPContext node 需要 `ep.context_enable=1` | 🟡 中 | 模型已内嵌，应直接可用 |

## 当前状态

- [ ] Step 1: 交叉编译 ORT
- [ ] Step 2: 提取 QNN EP 插件
- [ ] Step 3: 推送文件到设备
- [x] Step 4: 修改代码（AppendExecutionProvider V1 fallback）
- [ ] Step 5: 编译 & 运行测试
