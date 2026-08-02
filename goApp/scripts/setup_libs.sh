#!/data/data/com.termux/files/usr/bin/bash
# 部署 QNN HTP 运行库到 $PREFIX/lib（Termux 非 root、SELinux Enforcing 可用）。
# 依赖来源：
#   /vendor/lib64          -> fastrpc / DSP HAL / vmmem
#   androidApp APK         -> QNN 2.46 runtime 库
set -euo pipefail

PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
LIBDIR="$PREFIX/lib"
APK="${1:-}"

mkdir -p "$LIBDIR"

echo "==> 复制 vendor 库"
for lib in libcdsprpc.so vendor.qti.hardware.dsp@1.0.so libvmmem.so; do
  if [ -f "$LIBDIR/$lib" ]; then
    echo "skip $lib (exists)"
  else
    cp -v "/vendor/lib64/$lib" "$LIBDIR/$lib" || { echo "!! 复制 $lib 失败"; exit 1; }
  fi
done

if [ -z "$APK" ]; then
  APK=$(ls -1 /data/app/*/io.github.miuzarte.inferencer-*/base.apk 2>/dev/null | head -n1 || true)
fi
if [ -z "$APK" ] || [ ! -f "$APK" ]; then
  echo "!! 找不到 androidApp APK，请传路径: $0 /path/to/base.apk"
  exit 1
fi

echo "==> 从 $APK 提取 QNN 库"
TMP=$(mktemp -d)
unzip -o -q "$APK" 'lib/arm64-v8a/*' -d "$TMP"
for lib in libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so \
           libQnnHtpV73Skel.so libQnnHtpV73Stub.so; do
  src="$TMP/lib/arm64-v8a/$lib"
  if [ -f "$LIBDIR/$lib" ]; then
    echo "skip $lib (exists)"
  else
    cp -v "$src" "$LIBDIR/$lib"
  fi
done
rm -rf "$TMP"

echo "==> 完成。运行推理时设置:"
echo "    export LD_LIBRARY_PATH=$LIBDIR"
echo "    export ADSP_LIBRARY_PATH=$LIBDIR"
