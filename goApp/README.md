# goApp — Termux + Go 直连 QNN HTP 运行 YOLO26n

在 Termux（非 root、SELinux Enforcing）里用 Go 直接调用 QNN SDK C API，加载官方
`yolo26n_v73_qnn.onnx` 内嵌的 EPContext 二进制，跑 YOLO26n 目标检测；同时作为
GoCVStreamer 的远程推理客户端，通过 WebSocket 回传检测结果。

当前状态：P0~P4 均已完成，单图推理、WS 流式链路、HTP 性能档位调优都已落地（详情见
[PLAN.md](PLAN.md)）。

## 特点

- 无 JVM、无 ONNX Runtime 依赖：`htpbench` 4.8MB，`streamer`（带 turbojpeg 解码）
  10.3MB，贴近"开销最小化"目标；
- 手机侧单帧 `execute`（NPU）：不投性能票时 小米13 15.8ms / 小米17 16.6ms，
  投 `burst` 后 **5.5ms / 3.3ms**（-63% / -80%，见下节）；
- 使用官方 release 模型内嵌的 QNN EPContext（`models/ctx_v73.bin` /
  `models/ctx_v81.bin`）与 QNN SDK 2.46 头文件（`qnn-headers/` 不入库）；
- 流式推理协议与 flutterApp / androidApp 一致，支持断线自动重连。
- 推理线程可固定到超大核（`-affinity`），无需 root；断线重连后自动重新绑定。
- **HTP 性能档位可下发**（`-htp-perf`），默认 `burst`；`default` 可退回不投票。

## 目录结构

```text
cmd/extract_ctx   从官方 ONNX 提取 EPContext（需先自行下载 onnx）
cmd/infer         单张 JPEG 推理，可输出带框图片
cmd/streamer      GoCVStreamer 远程推理客户端
cmd/htpbench      HTP 性能档位对比工具（单进程内轮询多档位）
qnn/              cgo 桥接（dlopen + QNN provider vtable + perf infrastructure）
qnn/perf          纯 Go 的档位表与枚举常量（不依赖 cgo，可离线 go test）
yolo/             预处理（letterbox/量化）与后处理（argmax + NMS）
scripts/          部署 QNN 库到 Termux / 交叉编译的脚本
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

### 在 PC 上交叉编译 (设备上不必装 Go)

```powershell
.\scripts\build_android.ps1          # 只编译, 产物 tmp/streamer-arm64
.\scripts\build_android.ps1 -Push    # 编译并推送到 192.168.1.103:~/streamer
.\scripts\build_android.ps1 -Cmd ./cmd/htpbench -Out tmp/htpbench-arm64 `
    -RemoteName htpbench -Push       # 档位对比工具
```

`-Cmd` / `-Out` / `-RemoteName` 之外的新参数必须先加进脚本的 `param`
块：PowerShell 会把**未声明的参数静默丢进 `$args`**（不报错），
结果是"以为换了目标，其实又编了一遍 streamer"。

脚本会设置 `GOOS=android` / `GOARCH=arm64` / `CGO_ENABLED=1` 等**进程级**环境
变量且不会还原，所以**同一个 shell 里不要再接着跑 `go test`**（会尝试执行
android 产物并报 `%1 is not a valid Win32 application`），另开一条命令即可。

需要 Android NDK (自动取 `$env:ANDROID_SDK_ROOT\ndk` 下版本号最大的一个) 和设备侧
`pkg install libjpeg-turbo`。脚本首次运行会把 turbojpeg 的头文件与 `.so` 拉到
`tmp/android-arm64-sysroot/` 供交叉链接, 并写入
`RUNPATH=/data/data/com.termux/files/usr/lib`。

设备侧一次性准备 (QNN 运行库放 `$PREFIX/lib`; 不装 `libQnnHtpPrepare.so` 也能
加载预编译的 EPContext binary):

```bash
pkg install libjpeg-turbo binutils   # binutils 提供 setup_libs.sh 要用的 readelf
bash setup_libs.sh -a 73             # 自动找设备上的 androidApp APK; 81 = 小米17
bash setup_libs.sh -a 81 -q /path/to/qnnlibs   # APK 不在设备上时, 用本地解好的目录
cd ~/streamer && ./run.sh            # 见 scripts/run_device.sh
```

`run.sh` 等价于
`./streamer -url ws://192.168.1.100:9090/stream -ctx models/ctx_v<ARCH>.bin -arch <ARCH> -affinity 7 -htp-perf burst`，
机型差异只有 `ARCH`/`CTX` 两项，放在同目录的 `device.env` 里（不入库），
所以两台机器的 `run.sh` 一字不差。地址可用 `STREAM_URL` 覆盖，额外参数透传
(`./run.sh -bench -v`)。

`models/` 只随仓库分发提取好的 `ctx_v73.bin` / `ctx_v81.bin`，官方 onnx
（`yolo26n_v73_qnn.onnx` / `yolo26n_v81_qnn.onnx`，来自
`ultralytics/yolo-flutter-app` release）需要时自行下载，再用
`go run ./cmd/extract_ctx -model xxx.onnx -out models/ctx_xxx.bin` 提取。

## 新增设备部署（换机清单）

实测走过一遍（小米13 v73 → 小米17 v81），按这个顺序做：

1. `pkg install libjpeg-turbo binutils`。
2. 从**该机型的 androidApp APK**里取出 QNN 库（`libQnnHtp.so`、`libQnnSystem.so`、
   `libQnnHtpV<ARCH>{Skel,Stub}.so`）。不要用 SDK 里的
   `lib/hexagon-v<ARCH>/unsigned/libQnnHtpV<ARCH>Skel.so` —— 零售机走签名校验，
   用 APK 里那份（已验证可用）。
   APK 不在设备上时，在 PC 端 `unzip` 出 `lib/arm64-v8a/*` 再 `scp` 到
   `$PREFIX/lib`，然后 `bash setup_libs.sh -a <ARCH> -q $PREFIX/lib` 补齐 vendor 依赖。
3. `bash setup_libs.sh -a <ARCH>`：它会按 ELF `NEEDED` **递归**把 `/vendor/lib64`
   里的依赖复制过来。两个方向都不能靠手写清单：
   - **漏复制**：Android 16（小米17）的 `libcdsprpc.so` 多依赖一个 AIDL NDK 变体
     `vendor.qti.hardware.dsp-V1-ndk.so`，缺了直接 `dlopen failed: library not found`；
   - **多复制**：`/system/lib64` 里已有的 soname（`libc++.so`、`libutils.so`、
     `libbase.so`、`libdmabufheap.so`…）**绝对不能复制过去**，`LD_LIBRARY_PATH`
     会让链接器选中 vendor 那份并遮蔽系统版本，然后炸在别的库上：
     `cannot locate symbol "_ZNSt3__113__hash_memoryEPKvm" referenced by
     "/system/lib64/liblog.so"`。脚本对 `/system/lib64` 已有的 soname 既不复制也不递归，
     发现闲置的遮蔽副本会打印 `rm` 建议。
4. `pkg install` 之外还要确认 `readelf` 可用（`binutils`）。
5. 二进制**不用重编**：`-arch` 只是 flag（真机上 QNN 会忽略它、按 ctx 实际架构走），
   同一个 arm64 产物两台机器通用。
6. `scp` 二进制 + `models/ctx_v<ARCH>.bin` + `run.sh`，再写 `device.env`：
   ```
   STREAM_ARCH=81
   STREAM_CTX=models/ctx_v81.bin
   ```
7. 自检：`./htpbench -arch <ARCH> -ctx models/ctx_v<ARCH>.bin -profiles default,burst -rounds 2 -iters 90 -interval 33ms`
   —— 期望看到 `platform="arch=<ARCH> ..."`、`perf applied`、burst 明显快于 default、
   最后一行报"所有档位输出完全一致"。
8. 端到端：PC 起 streamer，手机 `./run.sh -bench`，看 `stats` 里的 `htp_perf=burst`
   与 `fps≈30 dropped=0`。

两台机器的实测对照：

| | 小米13（v73） | 小米17（v81） |
|---|---|---|
| Termux 用户 / 地址 | `u0_a441` / 192.168.1.103:8022 | `u0_a352` / 192.168.1.102:8022 |
| HTP 平台 | `arch=73 soc=43 vtcm=8MB` | `arch=81 soc=87 vtcm=8MB` |
| `execute` 不投票 | 15.8ms | 16.6ms |
| `execute` burst | 5.47ms | **3.32ms** |
| 手机 CPU（decode+quant） | 4.7 + 3.1ms | 2.0 + 2.9ms |
| `read`（每 33ms 里的空闲） | 22.2ms | 25.3ms |
| 默认档位 | `burst`（`run.sh`） | `burst`（`run.sh`） |
| Android / 坑 | 13；`/proc/<pid>/comm` 不可读 → `pgrep -x` 失效 | 16；`libcdsprpc.so` 多一个 AIDL 依赖 |

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
| `-affinity` | `-1` | worker 线程绑定的 CPU 集合，支持 `7` / `4-7` / `4,5,6,7`（`-1`=不绑） |
| `-pipeline` | `false` | 三段流水线（吞吐上限 = 1/max(CPU 前处理, NPU execute)；30fps 下比串行多 ~1.1ms 延迟） |
| `-fastjpeg` | `true` | turbojpeg `FASTDCT\|FASTUPSAMPLE`（更快，像素略变，见下） |
| `-argmax-all` | `false` | 后处理对全部 80 类求 argmax（默认只算 `-class` 那一类） |
| `-compress` | `false` | permessage-deflate（帧是 JPEG，压不动，只白吃 CPU） |
| `-htp-perf` | `default` | HTP 性能档位，见下节；`default` = 不下发，保持 QNN 默认行为（`run.sh` 默认传 `burst`） |
| `-rpc-control-latency` | `-1` | 覆盖档位的 RPC control latency（us）；`-1`=用档位默认，`0`=不下发 |
| `-rpc-polling-time` | `-1` | 覆盖档位的 RPC polling time（us，上限 9999）；`-1`=用档位默认，`0`=不下发 |
| `-htp-info` | `false` | 启动时打印性能基础设施状态与平台信息（arch/SOC/VTCM/核数） |
| `-bench` | `false` | 每 5s 输出各阶段耗时统计 |
| `-v` | `false` | verbose |

### HTP 性能档位（`-htp-perf`）

不给 HTP 投性能票时，DSP 跑在 DCVS 默认档上；`QnnDevice_getInfrastructure` →
`createPowerConfigId` → `setPowerConfig` 可以把电压角/功率模式钉住。
参数定义在 `qnn/perf/perf.go`（纯 Go，可离线 `go test`），下发在
`qnn/bridge.c`（与 QIDK / onnxruntime-qnn / ai-engine-direct-helper 三家实现一致）。

| 档位 | powerMode | 电压角 | sleepLatency | rpcPolling |
|---|---|---|---|---|
| `default` | 不下发 | — | — | — |
| `burst` | PERFORMANCE | MAX 0xa0 | 40us | 9999us |
| `burst_nosleep` | PERFORMANCE | MAX 0xa0 | 40us（且禁止休眠） | 9999us |
| `sustained_high_performance` / `high_performance` | PERFORMANCE | TURBO 0x80 | 100us | 9999us |
| `balanced` | PERFORMANCE（开 DCVS） | NOM_PLUS 0x70 | 1000us | 0 |
| `power_saver` | PERFORMANCE（开 DCVS） | SVS 0x40 | 1000us | 0 |

`cmd/htpbench` 用它在**单进程内轮询多档位**、按 33ms 占空比测 `graphExecute`
的 wall time（背靠背测会让 DSP 一直醒着，掩盖休眠唤醒成本）：

```bash
./htpbench -ctx models/ctx_v73.bin -rounds 3 -iters 120 -interval 33ms \
  -profiles default,burst,burst_nosleep,sustained_high_performance,balanced,power_saver
./htpbench -dry-run -profiles all      # 只打印解析后的档位参数, 不加载 QNN
./htpbench -profiles default -rounds 20 -iters 120   # 纯基线跑法, 看热衰减时间序列
```

`-hash-output`（默认开）对每轮结束时的输出张量做 FNV-1a 64 并汇总，用来确认
档位没有改变数值结果。<br>
`-rounds` 的每一行就是一段时间序列，所以"连续跑某一档"直接用它即可。

小米 13（HTP v73，`arch=73 soc=43 vtcm=8MB`，30fps 占空比，2026-09-13）：

| 档位 | execute avg | vs default |
|---|---|---|
| `default`（不下发） | **15.81ms** | — |
| `burst` | **5.80ms** | **-63.3%** |
| `burst_nosleep` | 5.73ms | -63.7% |
| `sustained_high_performance` | 6.04ms | -61.8% |
| `balanced` | 7.34ms | -53.6% |
| `power_saver` | 11.22ms | -29.1% |

- 单调性（角越高越快、`power_saver` 比 default 还慢）说明这确实是 DVFS 效应而非噪声；
- 4 个**全新进程**复测一致：default 15.59/15.52ms、burst 5.67ms；
- 背靠背（`-interval 0`）下 burst 5.07ms vs 33ms 下的 5.67ms，即每帧仍有约
  0.6ms 的休眠唤醒成本，但远小于不投票时损失的 ~10ms（后者是频率差，不是唤醒）；
- `sleepDisable` 几乎无差别（burst 5.80 vs burst_nosleep 5.73），收益全在电压角。

**注意**：投票是**进程级且不可撤销**的，`default` 只在一个进程的第一次（尚未
投票时）有效，htpbench 因此只在还没投过票的轮次里测 default。要复测 baseline
就重跑进程。

小米 17（HTP v81，`arch=81 soc=87 vtcm=8MB`，同样 30fps 占空比，2026-09-13）：

| 档位 | execute avg | vs default |
|---|---|---|
| `default`（不下发） | 16.57ms | — |
| `burst` | **3.53ms** | **-78.7%** |
| `burst_nosleep` | 3.54ms | -78.6% |
| `sustained_high_performance` | 4.05ms | -75.6% |
| `high_performance` | 4.22ms | -74.5% |
| `balanced` | 4.96ms | -70.1% |
| `power_saver` | 7.65ms | -53.9% |

同一个规律（单调、`power_saver` 仍慢于 default、7 个档位输出哈希完全相同），
只是绝对值更低：这代 HTP 满频只要 3.5ms。真机跑 `run.sh`（30fps / 122KB 帧）:
`execute 3.32ms`、`decode 2.0`、`quant 2.9`、`read 25.3`、fps 29.6~29.8、`dropped=0`。

### 端到端确认（真实 streamer 链路，3 对交替，2026-09-13）

htpbench 只是单个 `graphExecute` 的仪器；下面是把同样的档位交给真实
`cmd/streamer`（收 640×640 JPEG → 解码 → 量化 → execute → 后处理 → 回包）
跑出来的结果，每腿 42 秒、腿间留 8 秒沉降、跑前确认**没有残留进程**：

| leg | 档位 | execute | decode | quant | read（空闲） | execute 首窗 → 末窗 |
|---|---|---|---|---|---|---|
| A1 | `default` | 14.81ms | 4.82 | 3.20 | 10.02 | 14.86 → 14.82 |
| B1 | `burst` | **5.47ms** | 3.36 | 2.22 | 22.14 | 5.48 → 5.47 |
| A2 | `default` | 14.71ms | 4.54 | 2.99 | 10.72 | 14.88 → 14.86 |
| B2 | `burst` | **5.47ms** | 3.31 | 2.22 | 22.19 | 5.47 → 5.47 |
| A3 | `default` | 14.79ms | 4.82 | 3.20 | 10.20 | 14.87 → 14.77 |
| B3 | `burst` | **5.47ms** | 3.29 | 2.21 | 22.21 | 5.49 → 5.47 |

- `execute` **14.77 ± 0.04ms → 5.47 ± 0.00ms（-63%）**，三对完全一致；
- `read`（等下一帧的空闲）从 10.2ms 涨到 22.2ms，说明手机端余量翻倍；
- 顺带 CPU 段也变快（decode 4.7→3.3、quant 3.1→2.2）：不投票时线程长时间阻塞在
  fastrpc 上，CPU 频率被压下来；投票把整机状态带起来后 CPU 段也跟着受益。

**4 分钟连续跑（8 次温度采样 + 52 个 execute 窗口）没有任何热衰减**：

| 档位 | execute 逐窗范围 | 均值 | 首窗 → 末窗 | fps | 温度变化 |
|---|---|---|---|---|---|
| `default` | 14.03~14.94 | 14.72 | 14.75 → 14.65 | 30 | — |
| `burst` | 5.43~5.51 | **5.47** | 5.48 → 5.48 | 30 | DSP `nspss-*` 46→50°C（+3°C） |
| `sustained_high_performance` | 5.79~5.87 | 5.84 | 5.86 → 5.83 | 30 | 同上 |

CPU 50~51°C、`battery` 41~42°C、`quiet_therm` 44°C，离热关很远。所以
`run.sh` 的默认档位已经改成 **`burst`**（`STREAM_HTP_PERF=default` 退回不投票，
`sustained_high_performance` 是保守档，只慢 0.37ms 但用 TURBO 角而非 MAX 角）。

**只有两个旋钮真正起作用**（同进程轮换，burst 基准 5.79ms）：

| 配置 | execute | 结论 |
|---|---|---|
| `burst`（rpcPolling=9999, rpcLatency=100） | 5.79ms | 基准 |
| `burst -rpc-polling-time 0` | 6.24ms | polling 值 **0.44ms** |
| `burst -rpc-control-latency 0` | 5.78ms | `RPC_CONTROL_LATENCY` 在本机**无效果** |
| `burst_nosleep`（两者都 0） | 6.30ms | `sleepDisable` 也无差别 |

即收益几乎全在 **DCVS_V3 的电压角 + RPC polling**。

**数值结果不变**：`htpbench` 默认对输出张量做 FNV-1a 哈希，`default`/`burst`/
`sustained_high_performance`/`burst_nosleep` 在同一输入下哈希完全相同
（`9be3fc3889d96fc7`），跨轮次也一致 → 投票只改频率/电压，**不存在精度回归**。

**方法论警告（踩过的坑）**：小米13 上 `/proc/<pid>/comm` 不可读，`pgrep -x` /
`pkill -x` 会**静默返回空**，一个进程都杀不掉。第一次跑 A/B 时 6 条腿全部并发
运行、6 个 QNN 会话抢同一个 HTP，测出 `execute` 65ms 的完全错误的结论。
杀进程必须用 `pgrep -f '^\./streamer'` + `kill -9`，或者记下 `$!` 直接杀；
每条腿跑之前都断言残留进程数为 0。

### RGB→quint16 展开用 NEON（2026-09-14，小米17，60fps 源）

QNN 输入是 quint16（`scale=1/65536`、`offset=0`）的 NHWC RGB，而 turbojpeg 解出来是每通道
1 字节，所以每帧都要把 1.2MB 展成 2.4MB。旧实现逐字节查表：

```go
for i, v := range rgb { q := quint16LUT[v]; out[i*2] = byte(q); out[i*2+1] = byte(q>>8) }
```

这一步的信息量是零，而且能化简成纯字节复制。因为

```
LUT[b] = round(b/255*65536) = b*257 + (128 <= b <= 254 ? 1 : 0)
```

（`b=255` 时 `b*257 = 65535` 正好顶到 16 位上限，那个 `+1` 被 clamp 掉了），而 `b*257`
在小端内存里就是 `[b, b]` —— 正是 `vzip1q/vzip2q` 把向量和自身交错的结果。于是主体用
vzip、只给 128..254 补 1（**必须排除 255**，否则低位字节进位会污染高位字节），输出逐位相同。

真机上量 `cmd/streamer` 真正调用的那个包函数（3 个输出缓冲轮转，模拟流水线的冷缓存）：

| 实现 | median | 加速 |
|---|---|---|
| 标量查表 | 1.908ms | 1.0x |
| NEON `vzip` + 精确修正 | **0.126ms** | **15.1x** |

60fps 现场 A/B（`-pipeline -affinity 3-7 -htp-perf burst`，`/metrics` 各采样 35s）：

| 指标 | 旧（标量） | 新（NEON） |
|---|---|---|
| `quant` | 7.39ms | **0.42ms** |
| `cpu`（decode+quant+post） | 9.36ms | **4.01ms** |
| `read`（等下一帧的空闲） | 3.77ms | **13.04ms** |
| `latency` | 15.42ms | **12.05ms** |
| `execute` | 2.61ms | 3.42ms |

两腿之间 `execute`/`net` 漂了 ~1.3x（上面警告过 CPU 会随大核频率漂移），归一化后
`quant` 是 **~23x**、`cpu` **-67%**；`read` 从 3.8ms 涨到 13ms，即 60fps 下手机从
"刚好跑满一个周期"变成 3/4 个周期空闲。代价是 **0 带宽**。

**`read` 才是饱和判据**：`read + decode + quant` 正好等于一个帧周期时，说明这一段已经
跑满；只看 `cpu` 会漏掉"到底还有没有余量"。

**验证方式**（改的是推理输入，必须证明数值不变，不能只看快了多少）：

1. `yolo/quint16_test.go` 在真机上跑：覆盖全部 256 个取值、NEON 主循环边界长度（15/16/17/31/32/33）、
   以及 `out` 装不下时不越界写；
2. `htpbench -image <640x640.jpg>` 比对新旧二进制的输出张量 FNV 哈希 —— 完全相同
   （`7516f9c6f58cd39b` / `d1ccc00a9ea91f68`），即识别行为零变化。

### 实测延迟分解（小米13, 30fps 源, 640×640 JPEG ~50KB, 成对交替测）

```
latency 29.1ms = net 8.0 (链路+PC) + cpu 6.3 (手机 CPU) + execute 14.8 (NPU)

  net     ≈ Wi-Fi RTT 5.4ms + 上行传输 ~2.5ms   (与手机 ICMP 实测吻合)
  cpu     = decode 2.9 + quant 2.7 + post 0.07  (post 只剩 argmax 一类 + NMS)
  read_ms ≈ 11ms  每 33ms 周期里手机闲 1/3: 手机不是瓶颈
```

| 配置 | latency | cpu | 说明 |
|---|---|---|---|
| 串行 + `-affinity 7` | **29.1ms** | 6.3ms | 默认, 单帧延迟最低 |
| 串行 + `-affinity 7` + `-htp-perf burst` | ~19ms | 5~6ms | 上表的 14.8ms `execute` 降到 5.5ms; net 不变, 精确 latency 待用 PC metrics 在有游戏前台时复测 |
| 流水线 + `-affinity 3-7` | 30.2ms | 7.2ms | 60fps 源也能 58.6fps 不掉帧（串行上限 ~49fps） |
| 串行 + `-affinity 3-7` | 33.0ms | 10.5ms | 串行别给宽集合: 线程被调度到低频大核反而更慢 |
| adb reverse (USB) | 43.8ms | — | net 23.5ms, 比 Wi-Fi 差得多, **不要走这条路** |

注意: 上面 `cpu` 会随手机状态漂移（实测同一配置 5.7~10.6ms，worker 核频率 0.86~1.6GHz，
屏幕/热状态影响很大），所以任何 A/B 都要**交替成对**测，不能跨时段比。

`-fastjpeg` 的精度代价（同一帧对比, 810×1080 测试图 + 真实 640 流帧）: 平均像素差
1.16/255、最大 41；真实流帧上前 3 个 person 分数完全一致、第 4 个差 1.7%、
框边缘差 ≤5px@640。要位级一致就用 `-fastjpeg=false`（122KB 帧上 decode 9.5ms vs 4.8ms）。

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
- **cgo 的 C 代码可能按 `-O0` 编译，而且两条构建路径不一致**：Go 的
  `CGO_CFLAGS` 默认值是 `-O2 -g`，但 `build_android.ps1` 早先是**直接赋值**
  `-I$sys/include`，把它丢掉了。于是设备上自编（Termux 的 Go 保留默认 `-O2`）与
  PC 交叉编译的产物性能差一个数量级：同一个 NEON 内核 `-O0` 下 **6.16ms**、
  `-O2` 下 **0.126ms**，`-O0` 时甚至比 Go 标量循环还慢 3 倍 —— 表现为
  "交叉编译出来的 `streamer` 性能反而回退，设备上自编的却正常"。
  已修：脚本改成**追加**默认值，热路径另外在源码里用 `#cgo CFLAGS: -O2` 钉死。
  **改 cgo 代码后务必 `llvm-objdump -d` 看一眼反汇编**，`-O0` 的特征很明显：
  没有 `movi`、常量向量靠逐字节 `ldr b` + `mov v.b[N]` 拼、满屏 `ldr`/`str`。

详细方案、验证记录与风险备选见 [PLAN.md](PLAN.md)。
