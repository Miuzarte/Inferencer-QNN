# goApp — Termux + Go 直连 QNN HTP 运行 YOLO26n

在 Termux（非 root、SELinux Enforcing）里用 Go 直接调用 QNN SDK C API，加载官方
`yolo26n_v73_qnn.onnx` 内嵌的 EPContext 二进制，跑 YOLO26n 目标检测；同时作为
GoCVStreamer 的远程推理客户端，通过 WebSocket 回传检测结果。

当前状态：P0/P1/P2 已完成，单图推理与 WS 流式链路均已跑通（详情见
[PLAN.md](PLAN.md)）。

## 特点

- 无 JVM、无 ONNX Runtime 依赖，单文件二进制约 4.5MB，贴近"开销最小化"目标；
- 小米 13（SD 8 Gen 2 / HTP v73）实测单帧推理约 19-23ms；
- 使用官方 release 模型内嵌的 QNN EPContext（`models/ctx_v73.bin` /
  `models/ctx_v81.bin`）与 QNN SDK 2.46 头文件（`qnn-headers/` 不入库）；
- 流式推理协议与 flutterApp / androidApp 一致，支持断线自动重连。
- 推理线程可固定到超大核（`-affinity`），无需 root；断线重连后自动重新绑定。

## 目录结构

```text
cmd/extract_ctx   从官方 ONNX 提取 EPContext（需先自行下载 onnx）
cmd/infer         单张 JPEG 推理，可输出带框图片
cmd/streamer      GoCVStreamer 远程推理客户端
qnn/              cgo 桥接（dlopen + QNN provider vtable）
yolo/             预处理（letterbox/量化）与后处理（argmax + NMS）
scripts/          部署 QNN 库到 Termux 的脚本
models/           EPContext binary（ctx_v73 / ctx_v81），onnx 不入库
qnn-headers/      QNN SDK 头文件
```

## 快速开始（Termux）

```bash
# 依赖：Go、clang、unzip
pkg install golang clang unzip

# 部署 QNN 运行库到 $PREFIX/lib（vendor 库 + APK 里的 QNN 2.46 runtime）
bash scripts/setup_libs.sh /path/to/androidApp.apk

go build -o infer ./cmd/infer
go build -o streamer ./cmd/streamer

# 单图推理
export ADSP_LIBRARY_PATH=$PREFIX/lib
./infer -image bus.jpg -ctx models/ctx_v73.bin -out boxes.png

# 连接 GoCVStreamer
./streamer -url ws://192.168.1.100:9090/stream -affinity 7
```

`models/` 只随仓库分发提取好的 `ctx_v73.bin` / `ctx_v81.bin`，官方 onnx
（`yolo26n_v73_qnn.onnx` / `yolo26n_v81_qnn.onnx`，来自
`ultralytics/yolo-flutter-app` release）需要时自行下载，再用
`go run ./cmd/extract_ctx -model xxx.onnx -out models/ctx_xxx.bin` 提取。

## 常用参数

`cmd/infer`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-image` | 无 | 输入 JPEG 路径（必填） |
| `-ctx` | `models/ctx_v73.bin` | QNN context binary |
| `-lib` | `$PREFIX/lib` | QNN 库目录 |
| `-arch` | `73` | HTP arch |
| `-conf` | `0.45` | 置信度阈值 |
| `-out` | 空 | 可选：输出带框图片 |

`cmd/streamer`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-url` | `ws://192.168.1.100:9090/stream` | GoCVStreamer WS 地址 |
| `-ctx` | `models/ctx_v73.bin` | QNN context binary |
| `-lib` | `$PREFIX/lib` | QNN 库目录 |
| `-arch` | `73` | HTP arch |
| `-class` | `0` | 只回传指定类别（0=person，-1=全部） |
| `-conf` | `0.45` | 置信度阈值 |
| `-affinity` | `-1` | 把推理线程固定到指定 CPU（如 7=超大核，-1=不绑定） |
| `-bench` | `false` | 每 5s 输出各阶段耗时统计 |
| `-v` | `false` | verbose |

`-affinity` 说明：Android 调度器只在进程持续负载下才把大核加入可调度集合，
streamer 启动时会忙等并重试绑定（约 1-2 秒），推理期间每 5s 输出
`[aff] cpu7 pinned=7 proc=7 busy=..% freq=..` 便于确认大核利用率（busy 为
推理线程 CPU 占用，proc 为当前所在核）；断线重连后会自动重新绑定。
实测（小米 13，`-affinity 7`）：绑定后 30fps 稳定，decode/quant/argmax 约
5.5/2.5/2.6ms，CPU7 占用约 40%；execute 18.3ms 基本不变（瓶颈在 NPU）。

## 关键坑位

- vendor 库必须复制到 `$PREFIX/lib` 并按依赖顺序预加载
  `libvmmem.so → vendor.qti.hardware.dsp@1.0.so → libcdsprpc.so`，否则 fastrpc 失败；
- 官方模型的 QNN graph I/O 是 quint16，需要量化/反量化（输入
  `scale=1/65536`，输出 `scale≈0.010455`，offset 均为 0）；
- execute 时必须带上 binaryInfo 里的 tensor id，否则报 error 6004；
- 直连 QNN 的输出坐标是 640 尺度像素，归一化时不要额外 `*640`；
- Termux 下 `QnnContext_free` 会触发 SIGABRT，当前销毁策略是跳过 QNN free，
  交给进程退出回收。
- MIUI/HyperOS 调度器会动态收窄 Termux 进程的可调度核集：空闲时
  `sched_getaffinity` 常返回 `0x5f`（不含 CPU7），此时设置含 CPU7 的掩码会
  返回 EINVAL；进程产生持续负载约 1 秒后系统放开到 `0xff`，绑定才成功。
  `-affinity` 内部就是忙等重试；空闲约 5 秒后系统会重新收窄并把进程赶出
  大核，所以每次断线重连后都会重新确认绑定。
- `sched_setaffinity(pid=0)` 只影响调用线程：推理主循环里先
  `runtime.LockOSThread()` 再绑定，capture/网络线程保持全核可用。
- Android 普通用户读不了 `/proc/stat`（SELinux 拒绝），所以 `[aff]` 的
  busy% 取自推理线程自己的 `/proc/self/task/<tid>/stat` CPU 时间，而不是
  整核统计。

详细方案、验证记录与风险备选见 [PLAN.md](PLAN.md)。
