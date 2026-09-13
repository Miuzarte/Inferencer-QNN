# 交叉编译 goApp 的某个 cmd 为 android/arm64 单文件二进制, 可直接推送到 Termux 运行
#
# 用法:
#   .\scripts\build_android.ps1                          # 只编译 cmd/streamer 到 tmp/streamer-arm64
#   .\scripts\build_android.ps1 -Push                     # 编译并推送到手机
#   .\scripts\build_android.ps1 -Push -Remote 192.168.1.103
#   .\scripts\build_android.ps1 -Cmd ./cmd/htpbench -Out tmp/htpbench-arm64 `
#       -RemoteName htpbench -Push                        # 编译并推送 htpbench
#
# 依赖:
#   - Android NDK (默认取 $env:ANDROID_SDK_ROOT\ndk 下版本号最大的一个)
#   - 设备侧 libjpeg-turbo: pkg install libjpeg-turbo
#     首次编译会自动把 turbojpeg.h 与 libturbojpeg.so 拉到
#     tmp/android-arm64-sysroot/ (该目录不入库)
#
# 说明: 交叉链接时写入 RUNPATH=/data/data/com.termux/files/usr/lib,
# 因此产物只在 Termux ($PREFIX/lib) 下能找到 libturbojpeg.so.0。
# 注意: 本脚本的 param 块不含 -Cmd 之外的捕获, 传了未声明的参数会被 PowerShell
# 静默丢进 $args (不报错), 于是"编译出来的还是 streamer" — 加新参数务必同时加进 param。
param(
    [switch]$Push,
    [string]$Remote = '192.168.1.103',
    [int]$Port = 8022,
    [string]$Ndk,
    [string]$Api = '26',
    [string]$Cmd = './cmd/streamer',
    [string]$Out = 'tmp/streamer-arm64',
    [string]$RemoteName,
    [string]$RemoteDir = '~/streamer'
)

# 未指定 -RemoteName 时用 -Cmd 的目录名, 保持旧调用 (只传 -Out) 的行为可预期
if (-not $RemoteName) {
    $RemoteName = Split-Path $Cmd -Leaf
}
# 防呆: 未声明的参数会让 PowerShell 静默忽略, 导致编译目标与预期不符
if ($args.Count -gt 0) {
    throw "收到未声明的参数: $($args -join ' ')。请检查参数名拼写 (可用参数见本脚本 param 块)"
}

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# NDK: 未指定时取 SDK 下版本号最大的
if (-not $Ndk) {
    $sdk = if ($env:ANDROID_SDK_ROOT) { $env:ANDROID_SDK_ROOT } else { $env:ANDROID_HOME }
    if (-not $sdk) { throw 'ANDROID_SDK_ROOT / ANDROID_HOME 未设置, 请用 -Ndk 指定 NDK 路径' }
    $Ndk = Get-ChildItem (Join-Path $sdk 'ndk') -Directory |
        Sort-Object { [version]($_.Name -replace '[^0-9.].*$') } |
        Select-Object -Last 1 -ExpandProperty FullName
}
if (-not (Test-Path $Ndk)) { throw "NDK 不存在: $Ndk" }
$toolchain = Join-Path $Ndk 'toolchains/llvm/prebuilt/windows-x86_64/bin'
$cc = Join-Path $toolchain "aarch64-linux-android$Api-clang.cmd"
if (-not (Test-Path $cc)) { throw "找不到 $cc" }

# turbojpeg sysroot: 缺失时从设备拉取 (Termux 是 arm64 Android, 头与库可直接用于交叉链接)
$sys = Join-Path $root 'tmp/android-arm64-sysroot'
if (-not (Test-Path (Join-Path $sys 'include/turbojpeg.h')) -or
    -not (Test-Path (Join-Path $sys 'lib/libturbojpeg.so'))) {
    Write-Host "==> 从 $Remote 拉取 turbojpeg 头文件与库"
    New-Item -ItemType Directory -Force -Path (Join-Path $sys 'include'), (Join-Path $sys 'lib') | Out-Null
    $p = '/data/data/com.termux/files/usr'
    foreach ($h in 'turbojpeg.h', 'jconfig.h', 'jmorecfg.h', 'jpeglib.h') {
        scp -P $Port "${Remote}:$p/include/$h" (Join-Path $sys 'include')
    }
    # .so 是符号链接, 直接取实体文件
    $real = ssh -p $Port $Remote "readlink -f $p/lib/libturbojpeg.so"
    if (-not $real) { throw "设备上找不到 libturbojpeg.so, 先执行 pkg install libjpeg-turbo" }
    scp -P $Port "${Remote}:$real" (Join-Path $sys 'lib/libturbojpeg.so')
}

$env:CC = $cc
$env:CXX = Join-Path $toolchain "aarch64-linux-android$Api-clang++.cmd"
$env:CGO_ENABLED = '1'
$env:GOOS = 'android'
$env:GOARCH = 'arm64'
$env:CGO_CFLAGS = "-I$sys/include"
# rpath 指向 Termux 的 $PREFIX/lib, 运行时才能找到 libturbojpeg.so.0
$env:CGO_LDFLAGS = "-L$sys/lib -lturbojpeg -Wl,-rpath,/data/data/com.termux/files/usr/lib"

Write-Host "==> go build $Cmd ($env:GOOS/$env:GOARCH, NDK $((Split-Path $Ndk -Leaf)))"
go build -o $Out $Cmd
if ($LASTEXITCODE -ne 0) { throw "go build 失败 ($LASTEXITCODE)" }
Write-Host ("==> 产物 {0} ({1:N1} MB)" -f $Out, ((Get-Item $Out).Length / 1MB))

if ($Push) {
    Write-Host "==> 推送到 ${Remote}:$RemoteDir/$RemoteName"
    # 覆盖运行中的可执行文件会 ETXTBSY 且 scp 只打一行错误, 这里显式拦一道
    # 注意: 这台小米13 上 /proc/<pid>/comm 不可读, 所以 pgrep -x / pkill -x 一个都匹配不到
    # (静默返回空), 必须用 -f 匹配 cmdline, 且模式要锚定 "./name" 才不会匹到 ssh 自己的 shell
    $pattern = '^\./' + [regex]::Escape($RemoteName)
    $running = ssh -p $Port $Remote "pgrep -f '$pattern' | head -5 | tr '\n' ' '"
    if ($running) {
        throw "设备上 $RemoteName 正在运行 (pid: $running) — 先 ssh 上去 kill -9 它们; 注意 pgrep -x/pkill -x 在这台设备上无效"
    }
    ssh -p $Port $Remote "mkdir -p $RemoteDir/models"
    scp -P $Port $Out "${Remote}:$RemoteDir/$RemoteName"
    if ($LASTEXITCODE -ne 0) { throw "scp $Out 失败 ($LASTEXITCODE)" }
    ssh -p $Port $Remote "chmod +x $RemoteDir/$RemoteName"
    # 模型只在设备上缺失时补传
    $hasCtx = ssh -p $Port $Remote "test -f $RemoteDir/models/ctx_v73.bin && echo yes"
    if ($hasCtx -ne 'yes') {
        scp -P $Port models/ctx_v73.bin "${Remote}:$RemoteDir/models/ctx_v73.bin"
    }
    Write-Host "==> 完成: $RemoteDir/$RemoteName"
}
