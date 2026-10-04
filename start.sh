#!/usr/bin/env bash
# Cabo 一键开服（Linux / macOS）
#
#   ./start.sh              # 默认 8080，自动构建并打开浏览器
#   ./start.sh 9000         # 换端口
#   ./start.sh 9000 --no-build --no-browser
#   ./start.sh --stop       # 停止正在运行的服务
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT=8080
DO_BUILD=1
OPEN_BROWSER=1
STOP=0

for arg in "$@"; do
  case "$arg" in
    --stop) STOP=1 ;;
    --no-build) DO_BUILD=0 ;;
    --no-browser) OPEN_BROWSER=0 ;;
    ''|*[!0-9]*) echo "未知参数：$arg" >&2; exit 2 ;;
    *) PORT="$arg" ;;
  esac
done

EXE="$ROOT/target/release/cabo-server"

stop_server() {
  if pgrep -f "cabo-server" >/dev/null 2>&1; then
    pkill -f "cabo-server" || true
    sleep 0.6
    echo "已停止 cabo-server。"
  else
    echo "没有正在运行的 cabo-server。"
  fi
}

if [ "$STOP" = "1" ]; then stop_server; exit 0; fi

if [ "$DO_BUILD" = "1" ]; then
  echo "构建 release（源码有改动或首次运行）…"
  cargo build --release --bin cabo-server
elif [ ! -x "$EXE" ]; then
  echo "找不到 $EXE，且指定了 --no-build。请先 cargo build --release。" >&2
  exit 1
fi

# 端口占用：若是自己的旧实例就先停掉
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  if pgrep -f "cabo-server" >/dev/null 2>&1; then
    echo "端口 $PORT 被旧的 cabo-server 占用，先停掉…"
    stop_server
  else
    echo "端口 $PORT 已被其它进程占用，请换端口：./start.sh 9000" >&2
    exit 1
  fi
fi

export PORT RUST_LOG="${RUST_LOG:-info}"

# 局域网地址（朋友用房间号加入时用得到）
LAN_IPS="$( (hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || true) | tr ' ' '\n' | grep -E '^[0-9]+\.' || true)"

echo
echo "  Cabo 服务启动中…"
echo "  本机：      http://localhost:$PORT"
if [ -n "$LAN_IPS" ]; then
  for ip in $LAN_IPS; do
    echo "  局域网：    http://$ip:$PORT   （把房间号告诉同一网络的朋友）"
  done
fi
echo "  停止服务：  Ctrl+C，或 ./start.sh --stop"
echo

if [ "$OPEN_BROWSER" = "1" ]; then
  ( sleep 0.9; (xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || open "http://localhost:$PORT" >/dev/null 2>&1 || true) ) &
fi

exec "$EXE"
