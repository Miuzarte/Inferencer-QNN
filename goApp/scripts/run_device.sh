#!/data/data/com.termux/files/usr/bin/bash
# 启动 QNN HTP 远程推理客户端, 连接 GoCVStreamer 的 /stream WebSocket
#
# 前置:
#   1. pkg install libjpeg-turbo   (turbojpeg 解码, 交叉编译时已写入 rpath)
#   2. QNN 运行库放到 $PREFIX/lib  (见 scripts/setup_libs.sh, 首次部署一次即可)
#
# 用法:
#   ./run.sh                             # 默认: 串行 + 绑超大核 + HTP burst 档
#   STREAM_PIPELINE=1 ./run.sh           # 三段流水线 (吞吐更高, 30fps 下多 ~1.1ms)
#   STREAM_HTP_PERF=default ./run.sh     # 不下发性能档, 回到 DVFS 默认 (慢 63%)
#   STREAM_HTP_PERF=sustained_high_performance ./run.sh   # 保守档 (只慢 0.37ms)
#   STREAM_URL=ws://... ./run.sh -bench  # 换地址 / 透传参数
#
# 两台机器共享这一个脚本, 差异只在 ./device.env (不入库, 见下):
#   小米13 (SD8Gen2, HTP v73)  -> 默认 ctx_v73.bin / -arch 73, 不需要 device.env
#   小米17 (8 Elite, HTP v81)  -> device.env 里写:
#     STREAM_ARCH=81
#     STREAM_CTX=models/ctx_v81.bin
#
# 实测 (小米13, 30fps 源, 640x640 JPEG, 2026-09-13, 成对交替测):
#   execute 不投票 = 14.8ms, burst = 5.5ms (-63%), sustained = 5.8ms
#   4 分钟连续跑 burst 无衰减 (execute 5.43~5.51, NSP 温度只 +3C), fps 稳定 30
#   串行 + -affinity 7      latency 29.1ms = net 8.0 + cpu 6.3 + execute 14.8 (未投票)
#   流水线 + -affinity 3-7  latency 30.2ms (cpu 7.2), 但 60fps 源也能 58.6fps 不掉帧
#   串行 + -affinity 3-7    cpu 10.5ms (线程被调度到低频大核) — 串行别用宽集合
#   adb reverse (USB)       net 23.5ms, 比 WiFi 的 8.0ms 差得多, 不要走这条路
set -euo pipefail
cd "$(dirname "$0")"

# 非交互 ssh (zsh) 下 $PREFIX 可能没导出, 给个兜底
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
export ADSP_LIBRARY_PATH="$PREFIX/lib"
export LD_LIBRARY_PATH="$PREFIX/lib"

# 机型差异只有这两项 (小米13 = v73, 小米17/8 Elite = v81)。
# 放一个不入库的 ./device.env 覆盖, 两台机器的 run.sh 就能保持一字不差。
if [ -f ./device.env ]; then
  # shellcheck disable=SC1091
  . ./device.env
fi
ARCH="${STREAM_ARCH:-73}"
CTX="${STREAM_CTX:-models/ctx_v73.bin}"

# 防止息屏后 Termux 被冻结
if command -v termux-wake-lock >/dev/null 2>&1; then termux-wake-lock; fi

AFFINITY="${STREAM_AFFINITY:-7}"
if [ -n "${STREAM_PIPELINE:-}" ]; then
  # 流水线三段并行, 需要多个核才有意义 (只给一个核三段会互相抢)
  AFFINITY="${STREAM_AFFINITY:-3-7}"
  set -- -pipeline "$@"
fi

# HTP 性能档位: 不投票时 DSP 只跑 DVFS 默认档, 实测慢 63%, 所以默认就用 burst。
# 想退回不投票用 STREAM_HTP_PERF=default; 想省电用 sustained_high_performance。
exec ./streamer \
  -url "${STREAM_URL:-ws://192.168.1.100:9090/stream}" \
  -ctx "$CTX" \
  -arch "$ARCH" \
  -affinity "$AFFINITY" \
  -htp-perf "${STREAM_HTP_PERF:-burst}" \
  "$@"
