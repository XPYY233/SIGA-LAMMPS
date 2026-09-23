#!/bin/bash
# SIGA-LAMMPS 启动入口 —— 双击即可运行。
#
# 双击时 macOS 会用你的主目录作为工作目录，所以下面第一件事是把目录切到本
# 文件所在位置。若这一步失败，后面所有相对路径都会指错地方，而且报错会看起
# 来毫不相关。
#
# 出错时不会立刻关窗：双击运行的程序如果一闪而过，读不到任何信息，等于没有
# 报错。

set -uo pipefail

cd "$(dirname "$0")" || {
  echo "无法切换到脚本所在目录。"
  read -n 1 -s -r -p "按任意键关闭…"
  exit 1
}

VENV_PYTHON=".venv/bin/python"

clear
cat <<'BANNER'
  ┌──────────────────────────────────────────────┐
  │   SIGA-LAMMPS · 模拟任务控制台                │
  │   M（程序性记忆）· R（检索）· X（校验）· S（门控）│
  └──────────────────────────────────────────────┘

BANNER

# ---- 前置检查：每条失败都给出可执行的下一步 -------------------------------
fail() {
  echo
  echo "启动失败：$1"
  echo
  [ -n "${2:-}" ] && { echo "$2"; echo; }
  read -n 1 -s -r -p "按任意键关闭…"
  exit 1
}

[ -f "$VENV_PYTHON" ] || fail "未找到虚拟环境" \
"请先在终端里执行：
  cd $(pwd)
  python3 -m venv .venv
  .venv/bin/python -m pip install -e '.[dev,web]'
  .venv/bin/python -m adapter.cli build-index"

[ -f start.py ] || fail "未找到 start.py" "请确认这个文件放在仓库根目录。"

# 凭据来源与 start.py 一致：环境变量优先，其次 harness 自己的凭据库。
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ ! -f "$HOME/.dsh/.credentials.yaml" ]; then
  fail "找不到 DeepSeek 凭据" \
"请二选一：
  1. 在 ~/.dsh/.credentials.yaml 中配置 DEEPSEEK_API_KEY
  2. 或在 ~/.zshrc 里 export DEEPSEEK_API_KEY=..."
fi

# ---- 启动 ----------------------------------------------------------------
echo "正在启动，首次启动约需 40–60 秒…"
echo "（harness 在 3081，控制台在 8090；你日常的 GUI 在 3080，不会被触碰）"
echo

# --replace：上次退出时留下的进程可能还占着 3081 / 8090。没有它，启动会以
# “端口被占用”失败，而用户无法判断占用的到底是谁。start.py 只会结束能确认属
# 于本项目的进程，别的程序一律只报告、不触碰。
"$VENV_PYTHON" start.py --replace "$@"
status=$?

if [ $status -ne 0 ]; then
  echo
  echo "程序退出，退出码 $status。"
  read -n 1 -s -r -p "按任意键关闭…"
fi
exit $status
