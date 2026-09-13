#!/data/data/com.termux/files/usr/bin/bash
# 部署 QNN HTP 运行库到 $PREFIX/lib（Termux 非 root、SELinux Enforcing 可用）。
#
# 依赖来源：
#   /vendor/lib64    -> fastrpc / DSP HAL / vmmem（及其递归 NEEDED）
#   androidApp APK   -> QNN 2.46 runtime 库（libQnnHtp/libQnnSystem/libQnnHtpV<ARCH>{Skel,Stub}）
#
# 用法：
#   bash setup_libs.sh                        # arch 默认 73，自动找设备上的 androidApp APK
#   bash setup_libs.sh -a 81                  # 小米17 / 8 Elite
#   bash setup_libs.sh -a 81 -q /path/qnnlibs # APK 不在设备上时，用本地解出来的 QNN 库目录
#   bash setup_libs.sh /path/to/base.apk      # 位置参数等价于 -q 的 APK 来源
#
# 关键点：vendor 库的依赖必须递归解析，不能手写清单。
#   Android 13 上 libcdsprpc.so 只 NEEDED vendor.qti.hardware.dsp@1.0.so；
#   Android 16 上还多了一个 AIDL NDK 变体 vendor.qti.hardware.dsp-V1-ndk.so，
#   漏掉它 dlopen 会直接失败（报 "library not found: needed by libcdsprpc.so"）。
#   本脚本用 readelf 递归把 /vendor/lib64 里的依赖复制过来。
#
# 反向的坑（比漏复制更隐蔽）：**已经在 /system/lib64 里的库绝对不要复制**。
#   default namespace 本来就能找到它们；复制一份 /vendor 版本到 $PREFIX/lib 会
#   因为 LD_LIBRARY_PATH 优先而被链接器选中，遮蔽系统版本，然后炸在别的库上：
#     cannot locate symbol "_ZNSt3__113__hash_memoryEPKvm" referenced by
#     "/system/lib64/liblog.so"      <- 就是 vendor 的 libc++.so 顶掉了系统的
#   所以：soname 在 /system/lib64 存在 -> 既不复制也不递归。
#
# 需要 binutils 提供 readelf：pkg install binutils
set -euo pipefail

PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
LIBDIR="$PREFIX/lib"
ARCH="73"
QNN_SRC=""
APK=""

while [ $# -gt 0 ]; do
  case "$1" in
    -a|--arch) ARCH="$2"; shift 2 ;;
    -q|--qnn-dir) QNN_SRC="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) APK="$1"; shift ;;
  esac
done

if ! command -v readelf >/dev/null 2>&1; then
  echo "!! 缺少 readelf，先 pkg install binutils"
  exit 1
fi

mkdir -p "$LIBDIR"

echo "==> 从 APK 提取 QNN 库 (arch v$ARCH)"
if [ -z "$QNN_SRC" ]; then
  if [ -z "$APK" ]; then
    APK=$(ls -1 /data/app/*/io.github.miuzarte.inferencer-*/base.apk 2>/dev/null | head -n1 || true)
  fi
  if [ -n "$APK" ] && [ -f "$APK" ]; then
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    unzip -o -q "$APK" 'lib/arm64-v8a/*' -d "$TMP"
    QNN_SRC="$TMP/lib/arm64-v8a"
  else
    echo "    APK 不在设备上，跳过（假定 QNN 库已在 $LIBDIR，或用 -q 指定目录）"
  fi
fi

QNN_LIBS="libQnnHtp.so libQnnSystem.so libQnnHtpV${ARCH}Skel.so libQnnHtpV${ARCH}Stub.so libQnnHtpPrepare.so"
if [ -n "$QNN_SRC" ]; then
  for lib in $QNN_LIBS; do
    if [ -f "$LIBDIR/$lib" ]; then
      echo "    skip $lib (exists)"
    elif [ -f "$QNN_SRC/$lib" ]; then
      cp "$QNN_SRC/$lib" "$LIBDIR/" && echo "    copy $lib"
    else
      echo "    !! $QNN_SRC 里没有 $lib"
    fi
  done
fi

echo "==> 递归复制 /vendor/lib64 的依赖"
# 入口是 qnn/bridge.c 里按依赖序预加载的那三个，加上 QNN 自身
QUEUE="libvmmem.so vendor.qti.hardware.dsp@1.0.so libcdsprpc.so"
for lib in $QNN_LIBS; do
  [ -f "$LIBDIR/$lib" ] && QUEUE="$QUEUE $lib"
done
SEEN=""
COPIED=0
while [ -n "$QUEUE" ]; do
  # shellcheck disable=SC2086
  set -- $QUEUE
  lib=$1
  shift
  QUEUE="$*"
  case " $SEEN " in *" $lib "*) continue ;; esac
  SEEN="$SEEN $lib"

  # /system/lib64 已覆盖的 soname 一律不碰（见文件头的"反向的坑"）
  if [ -e "/system/lib64/$lib" ]; then
    if [ -e "$LIBDIR/$lib" ]; then
      SHADOW="$SHADOW $lib"
    fi
    continue
  fi

  src="$LIBDIR/$lib"
  if [ ! -e "$src" ]; then
    if [ -e "/vendor/lib64/$lib" ]; then
      cp "/vendor/lib64/$lib" "$LIBDIR/" && echo "    copy $lib" && COPIED=$((COPIED + 1))
      chmod u+rw "$LIBDIR/$lib" 2>/dev/null || true
      src="$LIBDIR/$lib"
    else
      echo "    !! 找不到 $lib（既不在 $LIBDIR 也不在 /vendor/lib64）"
      continue
    fi
  fi

  # 递归 NEEDED；只关心能从 /vendor/lib64 或 $LIBDIR 拿到的
  for dep in $(readelf -d "$src" 2>/dev/null | sed -n 's/.*NEEDED.*\[\(.*\)\].*/\1/p'); do
    if [ -e "/system/lib64/$dep" ]; then
      continue
    fi
    if [ -e "$LIBDIR/$dep" ] || [ -e "/vendor/lib64/$dep" ]; then
      QUEUE="$QUEUE $dep"
    fi
  done
done
echo "    新增 $COPIED 个库"

if [ -n "${SHADOW:-}" ]; then
  echo "!! 发现遮蔽系统库的副本（会让链接器选中版本不对的那个），建议删掉："
  for lib in $SHADOW; do
    echo "     rm -f $LIBDIR/$lib"
  done
fi

echo "==> 完成（arch v$ARCH）。运行推理时:"
echo "    export ADSP_LIBRARY_PATH=$LIBDIR"
echo "    export LD_LIBRARY_PATH=$LIBDIR"
