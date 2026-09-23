#!/bin/bash
# 安装并修复 OVITO（macOS arm64）。
#
#   ./setup_ovito.sh
#
# 为什么需要这个脚本：pip 的 macOS arm64 wheel 有一个打包缺陷，
# `ovito_bindings.so` 链接的是 `libospray.3.2.0.dylib`，而包里提供的是
# `libospray.3.dylib`。两者是同一个 ABI 的两个名字，但动态链接器只认字面
# 名字，于是 `import ovito` 直接失败并报 "Library not loaded"。
#
# 修复方式分两级，优先用更持久的那一种：
#   1. install_name_tool —— 直接改正 .so 里的引用，重装 python 包不受影响；
#      但修改 Mach-O 会使签名失效，arm64 上必须重新 ad-hoc 签名。
#   2. 符号链接 —— 退路，简单可靠，但重装 ovito 后需要重新执行本脚本。
#
# 脚本是幂等的：重复执行只会重新确认状态，不会破坏什么。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
PLUGINS_REL="lib/python3.12/site-packages/ovito/plugins"

MISSING_LIB="libospray.3.2.0.dylib"
SHIPPED_LIB="libospray.3.dylib"

say() { printf '%s\n' "$*"; }
fail() { say ""; say "失败：$*"; exit 1; }

# ---------------------------------------------------------------- 前置检查
[ -f "$VENV_PYTHON" ] || fail "未找到虚拟环境：$VENV_PYTHON
请先创建：python3 -m venv .venv"

PY_VER="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PLUGINS="$REPO_ROOT/.venv/lib/python$PY_VER/site-packages/ovito/plugins"

if [ "$(uname -s)" != "Darwin" ]; then
  say "非 macOS，跳过 OVITO 修复（该缺陷只影响 macOS arm64 的 wheel）。"
  exit 0
fi

# ---------------------------------------------------------------- 安装
say "安装 ovito…"
"$VENV_PYTHON" -m pip install -q --upgrade ovito || fail "pip install ovito 失败"

[ -d "$PLUGINS" ] || fail "安装后仍找不到 $PLUGINS"

# ---------------------------------------------------------------- 检查是否需要修
if [ ! -f "$PLUGINS/$SHIPPED_LIB" ]; then
  say "未发现 $SHIPPED_LIB，该版本可能已修复此问题。"
else
  # 判断依据是「.so 是否仍引用那个缺名字」，而不是「那个名字的文件在不在」。
  #
  # 修复成功之后，libospray.3.2.0.dylib 本来就不该存在——.so 已经改指
  # libospray.3.dylib。早先的版本却把「文件不存在」当成需要修复，于是每次
  # 重跑都重新改一遍并报告"修复完成"，幂等性形同虚设。
  NEEDS_FIX=0
  if otool -L "$PLUGINS/ovito_bindings.so" 2>/dev/null | grep -q "$MISSING_LIB"; then
    NEEDS_FIX=1   # 引用仍指向缺名字：需要修
  fi

  if [ "$NEEDS_FIX" = "1" ]; then
    say "应用修复：$MISSING_LIB -> $SHIPPED_LIB"

    # 优先：改正 .so 里的引用，这样重装 python 包也不受影响。
    # 必须用加载命令里的完整写法。install_name_tool -change 要求旧名逐字匹配，
    # 传一个去掉 @loader_path/ 前缀的名字不会匹配任何东西，而且**静默什么都不做**
    # 并返回 0 —— 于是"修复成功"而引用丝毫未变。
    MISSING_REF="@loader_path/$MISSING_LIB"
    SHIPPED_REF="@loader_path/$SHIPPED_LIB"

    FIXED=0
    if command -v install_name_tool >/dev/null 2>&1; then
      if install_name_tool -change "$MISSING_REF" "$SHIPPED_REF" \
           "$PLUGINS/ovito_bindings.so" 2>/dev/null; then
        # 修改 Mach-O 会让签名失效；arm64 上必须重新 ad-hoc 签名，否则加载被拒。
        codesign --force --sign - "$PLUGINS/ovito_bindings.so" 2>/dev/null
        # 验证引用真的变了，而不是相信命令的退出码。
        if otool -L "$PLUGINS/ovito_bindings.so" 2>/dev/null | grep -q "$MISSING_LIB"; then
          say "  install_name_tool 未生效（引用未变），改用符号链接"
        else
          say "  install_name_tool 修复完成（已重新 ad-hoc 签名，引用已确认改变）"
          FIXED=1
        fi
      else
        say "  install_name_tool 执行失败，改用符号链接"
      fi
    else
      say "  未找到 install_name_tool，改用符号链接"
    fi

    if [ "$FIXED" != "1" ]; then
      rm -f "$PLUGINS/$MISSING_LIB"
      ln -sf "$SHIPPED_LIB" "$PLUGINS/$MISSING_LIB"
      say "  已建立符号链接（重装 ovito 后需重跑本脚本）"
    fi
  else
    say "修复已就位。"
  fi
fi

# ---------------------------------------------------------------- 验证
say ""
say "验证导入与渲染…"
"$VENV_PYTHON" - <<'PY' || fail "验证未通过"
import sys, tempfile, os
from pathlib import Path
try:
    import ovito
except Exception as exc:
    print(f"  import ovito 失败：{exc}")
    sys.exit(1)
print(f"  ovito {ovito.version_string}")

# 真造一个结构文件再渲染，而不是只 import —— 空场景的渲染也会"成功"，
# 所以必须确认输出里有东西。
tmp = Path(tempfile.mkdtemp())
xyz = tmp / "probe.xyz"
xyz.write_text("2\nprobe\nAr 0 0 0\nAr 0.1 0.1 0.1\n")
try:
    from ovito.io import import_file
    from ovito.vis import Viewport, TachyonRenderer
    p = import_file(str(xyz))
    p.add_to_scene()                     # 缺这一步会渲染出空图
    p.compute()
    vp = Viewport(type=Viewport.Type.Ortho)
    vp.zoom_all()
    out = tmp / "probe.png"
    vp.render_image(filename=str(out), size=(160, 120), renderer=TachyonRenderer())
    size = out.stat().st_size
    print(f"  渲染输出 {size} 字节")
    if size < 1000:
        print("  输出过小，可能是空场景")
        sys.exit(1)
except Exception as exc:
    print(f"  渲染失败：{type(exc).__name__}: {exc}")
    sys.exit(1)
print("  OK")
PY

say ""
say "OVITO 就绪。控制台里的轨迹文件（.dump / .xyz / .cfg / .data）点击即可渲染。"
