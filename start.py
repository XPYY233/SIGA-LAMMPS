#!/usr/bin/env python3
"""Start the SIGA-LAMMPS console.

    .venv/bin/python start.py

Starts two processes and prints the URL:

* the DeepSeekHarness with the SIGA adapter mounted, on ``--harness-port``
  (default 3081), and
* the researcher console, on ``--console-port`` (default 8090).

Ctrl-C stops both. Nothing is left running, and nothing outside this repository
is written to.

Why the harness runs on its own port and its own home: the everyday GUI is the
``web`` profile at ``~/.dsh`` on 3080, and mounting this adapter there would hand
every unrelated conversation a stop gate and an LPC toolset. ``DSH_HOME`` is kept
inside the repository for the same reason — the harness's own session store stays
untouched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
HARNESS_ROOT = Path(os.environ.get("SIGA_HARNESS_ROOT", "/Users/fanjunran/deepseek-harness"))
DSH_HOME = REPO_ROOT / ".dsh-web"
PATCH = REPO_ROOT / "harness" / "siga-patch.yml"
CREDENTIALS = Path.home() / ".dsh" / ".credentials.yaml"

CONFIGURATIONS = ("vanilla", "m", "mr", "mrsx")


class StartupError(RuntimeError):
    """A precondition failed, with a message that says what to do about it."""


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def wait_for(port: int, *, seconds: float, label: str) -> bool:
    """Poll until the port answers, so the URL is never printed too early."""
    deadline = time.monotonic() + seconds
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    return False


def load_api_key() -> str:
    """Environment first, then the harness's own credential store.

    The key is read, never written and never printed. The harness already knows
    how to hold it; this only passes it through to the process that needs it.
    """
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if CREDENTIALS.is_file():
        try:
            import yaml

            data = yaml.safe_load(CREDENTIALS.read_text(encoding="utf-8")) or {}
            key = str(data.get("DEEPSEEK_API_KEY", "")).strip()
        except (OSError, ValueError):
            key = ""
    if not key:
        raise StartupError(
            "找不到 DeepSeek 凭据。请设置环境变量 DEEPSEEK_API_KEY，"
            f"或确认 {CREDENTIALS} 里存在该条目。"
        )
    return key


def preflight(harness_port: int, console_port: int, configuration: str) -> None:
    """Fail before starting anything, with an actionable reason."""
    if not VENV_PYTHON.is_file():
        raise StartupError(
            f"未找到虚拟环境：{VENV_PYTHON}\n"
            "请先创建：python3 -m venv .venv && .venv/bin/python -m pip install -e '.[dev,web]'"
        )
    if not (HARNESS_ROOT / "apps" / "cli" / "src" / "bin.ts").is_file():
        raise StartupError(
            f"未找到 DeepSeekHarness 检出：{HARNESS_ROOT}\n"
            "可用环境变量 SIGA_HARNESS_ROOT 指向它。"
        )
    if not (HARNESS_ROOT / "node_modules" / ".bin" / "tsx").is_file():
        raise StartupError(f"{HARNESS_ROOT} 尚未安装依赖，请先在该目录执行 pnpm install。")
    for port, label in ((harness_port, "harness"), (console_port, "控制台")):
        if not port_free(port):
            raise StartupError(
                f"端口 {port}（{label}）已被占用。换一个端口，或先关掉占用它的进程。"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness-port", type=int, default=3081)
    parser.add_argument("--console-port", type=int, default=8090)
    parser.add_argument(
        "--configuration", choices=CONFIGURATIONS, default="mrsx",
        help="挂载哪个 adapter 配置（vanilla / m / mr / mrsx）",
    )
    parser.add_argument("--no-browser", action="store_true", help="不要自动打开浏览器")
    args = parser.parse_args(argv)

    try:
        preflight(args.harness_port, args.console_port, args.configuration)
        api_key = load_api_key()
    except StartupError as exc:
        print(f"\n启动失败：{exc}\n", file=sys.stderr)
        return 1

    # One overlay for this configuration. Generated, never hand-edited, because
    # it carries absolute paths and would otherwise hard-code a checkout location.
    print(f"生成 overlay（配置 {args.configuration}）…")
    subprocess.run(
        [str(VENV_PYTHON), str(REPO_ROOT / "harness" / "make_patch.py"),
         "--preset", args.configuration,
         "--port", str(args.harness_port),
         "--mode", "web"],
        check=True,
    )

    DSH_HOME.mkdir(parents=True, exist_ok=True)
    harness_env = {
        **os.environ,
        "DSH_HOME": str(DSH_HOME),
        # The workspace-write sandbox cannot start on this host, and an escalation
        # request would wait for an approval nothing answers. The deployment sets
        # the mode rather than code special-casing it. The remote side is
        # unaffected: the HPC layer has no arbitrary-shell tool at all.
        "DSH_PERMISSION_MODE": "danger-full-access",
        "DEEPSEEK_API_KEY": api_key,
    }

    processes: list[tuple[str, subprocess.Popen]] = []

    def shutdown(*_ignored: object) -> None:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for name, process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"  强制结束 {name}")
        print("\n已停止。")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"启动 harness（端口 {args.harness_port}，DSH_HOME={DSH_HOME}）…")
    harness = subprocess.Popen(
        ["node", "--import", "tsx/esm", "apps/cli/src/bin.ts", "web", "--patch", str(PATCH)],
        cwd=HARNESS_ROOT, env=harness_env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    processes.append(("harness", harness))

    if not wait_for(args.harness_port, seconds=90, label="harness"):
        shutdown()
        print(f"\nharness 未能在 90 秒内响应，请检查 {HARNESS_ROOT} 下的输出。", file=sys.stderr)
        return 1

    print(f"启动控制台（端口 {args.console_port}）…")
    console_env = {**os.environ, "SIGA_HARNESS_URL": f"http://127.0.0.1:{args.harness_port}"}
    console = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "uvicorn", "web.backend.app:create_app", "--factory",
         "--host", "127.0.0.1", "--port", str(args.console_port), "--log-level", "warning"],
        cwd=REPO_ROOT, env=console_env,
    )
    processes.append(("console", console))

    if not wait_for(args.console_port, seconds=60, label="控制台"):
        shutdown()
        print("\n控制台未能在 60 秒内响应。", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{args.console_port}"
    print(
        f"\n{'=' * 56}\n"
        f"  控制台已就绪：{url}\n"
        f"  配置：{args.configuration}      （切换配置需重启本程序）\n"
        f"  按 Ctrl-C 停止\n"
        f"{'=' * 56}\n"
    )
    if not args.no_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - opening a browser is a convenience
            pass

    try:
        # Wait on either child, so a crashed harness surfaces instead of leaving a
        # console whose every panel is empty.
        while all(process.poll() is None for _, process in processes):
            time.sleep(1.0)
        if any(process.poll() is not None for _, process in processes):
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n{name} 已退出（退出码 {process.returncode}）。", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
