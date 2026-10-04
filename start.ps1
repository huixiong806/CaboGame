<#
.SYNOPSIS
  Cabo 一键开服：自动构建 → 清理旧实例 → 启动服务 → 打开浏览器 → 打印局域网地址。

.EXAMPLE
  .\start.ps1                 # 默认 8080 端口，自动构建并打开浏览器
  .\start.ps1 -Port 9000      # 换端口
  .\start.ps1 -SkipBuild      # 跳过构建（改过代码就要重新构建）
  .\start.ps1 -NoBrowser      # 不自动开浏览器
  .\start.ps1 -Stop           # 停止正在运行的服务
#>
[CmdletBinding()]
param(
    [int]$Port = 8080,
    [switch]$SkipBuild,
    [switch]$NoBrowser,
    [switch]$Stop,
    [string]$LogLevel = "info"
)

$ErrorActionPreference = "Stop"

# 控制台切 UTF-8：否则服务端输出的中文日志在 GBK 代码页下会变成乱码
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $null = chcp 65001
} catch { }
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$serverExe = Join-Path $root "target\release\cabo-server.exe"
if (-not (Test-Path $serverExe)) { $serverExe = Join-Path $root "target\release\cabo-server" }

function Get-CaboServerProcesses {
    Get-Process -Name "cabo-server" -ErrorAction SilentlyContinue
}

function Stop-CaboServer {
    param([switch]$Quiet)
    $procs = Get-CaboServerProcesses
    if (-not $procs) {
        if (-not $Quiet) { Write-Host "没有正在运行的 cabo-server。" -ForegroundColor DarkGray }
        return
    }
    foreach ($p in $procs) {
        if (-not $Quiet) { Write-Host "停止 cabo-server (PID $($p.Id))…" -ForegroundColor Yellow }
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 600
}

if ($Stop) {
    Stop-CaboServer
    Write-Host "已停止。" -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------- 构建
function Test-NeedsBuild {
    if (-not (Test-Path $serverExe)) { return $true }
    $exeTime = (Get-Item $serverExe).LastWriteTimeUtc
    $watched = @("src", "templates", "static") |
        Where-Object { Test-Path (Join-Path $root $_) } |
        ForEach-Object { Get-ChildItem (Join-Path $root $_) -Recurse -File -ErrorAction SilentlyContinue }
    $watched += Get-Item (Join-Path $root "Cargo.toml")
    foreach ($f in $watched) {
        if ($f.LastWriteTimeUtc -gt $exeTime) { return $true }
    }
    return $false
}

if (-not $SkipBuild) {
    if (Test-NeedsBuild) {
        Write-Host "发现源码比二进制新（或首次运行），正在构建 release…" -ForegroundColor Cyan
        & cargo build --release --bin cabo-server
        if ($LASTEXITCODE -ne 0) {
            Write-Host "构建失败，已中止。" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "二进制已是最新，跳过构建。" -ForegroundColor DarkGray
    }
} elseif (-not (Test-Path $serverExe)) {
    Write-Host "找不到 $serverExe，且指定了 -SkipBuild。请先 cargo build --release。" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- 端口
$occupant = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($occupant) {
    $owner = Get-Process -Id $occupant.OwningProcess -ErrorAction SilentlyContinue
    if ($owner -and $owner.ProcessName -eq "cabo-server") {
        Write-Host "端口 $Port 已被旧的服务实例占用，先停掉它…" -ForegroundColor Yellow
        Stop-CaboServer -Quiet
    } else {
        $who = if ($owner) { "$($owner.ProcessName) (PID $($owner.Id))" } else { "未知进程" }
        Write-Host "端口 $Port 被 $who 占用。换一个端口：.\start.ps1 -Port 9000" -ForegroundColor Red
        exit 1
    }
}

# ---------------------------------------------------------------- 启动
$env:PORT = "$Port"
$env:RUST_LOG = $LogLevel

$lanIps = @()
try {
    $lanIps = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -ne "127.0.0.1" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.PrefixOrigin -ne "WellKnown"
        } | Select-Object -ExpandProperty IPAddress -Unique
} catch { }

Write-Host ""
Write-Host "  Cabo 服务启动中…" -ForegroundColor Green
Write-Host "  本机：      http://localhost:$Port"
foreach ($ip in $lanIps) {
    Write-Host "  局域网：    http://${ip}:$Port   （把房间号告诉同一网络的朋友）" -ForegroundColor Cyan
}
if ($lanIps.Count -gt 0) {
    Write-Host "  提示：若朋友连不上，检查 Windows 防火墙是否放行 cabo-server（首次监听会弹窗）。" -ForegroundColor DarkGray
}
Write-Host "  停止服务：  Ctrl+C，或另开一个窗口执行 .\start.ps1 -Stop"
Write-Host ""

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($u)
        Start-Sleep -Milliseconds 900
        Start-Process $u
    } -ArgumentList "http://localhost:$Port" | Out-Null
}

try {
    & $serverExe
} finally {
    Write-Host ""
    Write-Host "服务已退出。" -ForegroundColor DarkGray
}
