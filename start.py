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


#: Ports this launcher must never terminate a process on. 3080 is the user's own
#: DeepSeekHarness GUI, which is a separate application that happens to share a
#: checkout; killing it would take down the window they are working in.
PROTECTED_PORTS = frozenset({3080})

#: Command-line fragments that identify a process as *this* repository's own
#: harness or console. Matching on these is what makes automatic cleanup safe:
#: an unrecognised process on the port is never touched, only reported.
_OWN_MARKERS = (
    ("apps/cli/src/bin.ts", "siga-patch.yml"),   # our harness, with our overlay
    ("uvicorn", "web.backend.app"),              # our console
)


def port_holder(port: int) -> tuple[int, str] | None:
    """The PID and command line listening on *port*, if anything is."""
    try:
        listing = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in listing.stdout.split():
        if not line.strip().isdigit():
            continue
        pid = int(line)
        try:
            described = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        command = described.stdout.strip()
        if command:
            return pid, command
    return None


def is_our_process(command: str) -> bool:
    """Whether a command line is this project's own harness or console.

    Deliberately strict. Automatic cleanup is offered for leftovers from a
    previous session, so anything that cannot be positively identified as ours is
    reported instead of killed — a stranger's server on the port is not ours to
    stop.
    """
    return any(all(marker in command for marker in group) for group in _OWN_MARKERS)


def reclaim_port(port: int, label: str) -> str | None:
    """Terminate a stale instance of our own on *port*.

    Returns a description of what was stopped, or None if the port was already
    free. Raises when the occupant is not ours, because that is a decision for
    the person running this, not for the launcher.
    """
    if port in PROTECTED_PORTS:
        raise StartupError(
            f"拒绝操作端口 {port}：那是你日常使用的 GUI。"
            "本程序只会使用 3081（harness）和 8090（控制台）。"
        )
    if port_free(port):
        return None
    holder = port_holder(port)
    if holder is None:
        raise StartupError(f"端口 {port}（{label}）被占用，但无法确定是哪个进程。")
    pid, command = holder
    if not is_our_process(command):
        raise StartupError(
            f"端口 {port}（{label}）被另一个程序占用，我不会动它：\n"
            f"  PID {pid}: {command[:120]}\n"
            f"请自行关闭它，或用 --console-port / --harness-port 换端口。"
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        raise StartupError(f"无法结束残留进程 {pid}：{exc}") from exc
    for _ in range(20):
        if port_free(port):
            return f"PID {pid}（{label} 残留进程）"
        time.sleep(0.5)
    # It ignored SIGTERM. Escalate, since we have already established it is ours.
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    for _ in range(10):
        if port_free(port):
            return f"PID {pid}（{label} 残留进程，需强制结束）"
        time.sleep(0.5)
    raise StartupError(f"残留进程 {pid} 无法结束，请手动执行：kill -9 {pid}")


def console_is_up(port: int) -> bool:
    """Whether a *working* console already answers on this port.

    Distinguishes the two cases that used to look identical: a leftover process
    holding the port, and the workbench already running. Only the first should be
    cleared away; the second should just be opened.
    """
    if port in PROTECTED_PORTS or port_free(port):
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as reply:
            return reply.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _exit_description(returncode: int | None) -> str:
    """Describe how a child ended, naming signals instead of negative numbers.

    A segfault reports as -11 in Python. Printing "退出码 -11" invites the reader
    to look for a bug in the child's own error handling, when the process never
    got that far: it was killed. Naming the signal is the difference.
    """
    if returncode is None:
        return "状态未知"
    if returncode >= 0:
        return f"退出码 {returncode}"
    signals = {6: "SIGABRT（abort）", 9: "SIGKILL", 11: "SIGSEGV（段错误）", 15: "SIGTERM"}
    number = -returncode
    return f"被信号终止：{signals.get(number, f'信号 {number}')}"


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


def preflight(
    harness_port: int, console_port: int, configuration: str, *, replace: bool = False
) -> None:
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
        if port_free(port):
            continue
        if not replace:
            holder = port_holder(port)
            who = f"\n  当前占用者：PID {holder[0]}: {holder[1][:100]}" if holder else ""
            hint = (
                "加 --replace 可自动清理本项目的残留进程（本程序只结束能确认属于自己的进程）。"
                if holder and is_our_process(holder[1])
                else "换一个端口，或先关掉占用它的进程。"
            )
            raise StartupError(f"端口 {port}（{label}）已被占用。{who}\n{hint}")
        stopped = reclaim_port(port, label)
        if stopped:
            print(f"  已清理{stopped}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness-port", type=int, default=3081)
    parser.add_argument("--console-port", type=int, default=8090)
    parser.add_argument(
        "--configuration", choices=CONFIGURATIONS, default="mrsx",
        help="挂载哪个 adapter 配置（vanilla / m / mr / mrsx）",
    )
    parser.add_argument("--no-browser", action="store_true", help="不要自动打开浏览器")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="自动清理本项目上次退出时残留的进程（只结束能确认属于自己的进程）",
    )
    args = parser.parse_args(argv)

    # Already running is not an error. Double-clicking the launcher a second time
    # should show the workbench, not a port conflict — and that conflict is what
    # the old message reduced to "close whatever is using it", which is exactly
    # the leftover-from-last-time case the user cannot act on.
    if console_is_up(args.console_port):
        url = f"http://127.0.0.1:{args.console_port}"
        print(f"\n控制台已经在运行：{url}\n（如需重启，先按 Ctrl-C 停止原来的窗口，或用 --replace 强制重启）\n")
        if not args.no_browser:
            try:
                import webbrowser

                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - opening a browser is a convenience
                pass
        return 0

    try:
        preflight(args.harness_port, args.console_port, args.configuration, replace=args.replace)
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
    # The child's output goes to a file, not to a PIPE nobody drains. An
    # unread PIPE can block the child once it fills, and it hid every startup
    # error: a harness that died on a bad overlay looked identical to one that
    # never started.
    #
    # Opened for append, with a separator per session. Truncating on start
    # destroyed the evidence of the previous run's death — which is exactly the
    # evidence needed to find out why it died. The console's segfault was only
    # diagnosable at all because macOS kept its own crash report.
    harness_log_path = REPO_ROOT / ".dsh-web" / "harness.log"
    harness_log = open(harness_log_path, "a", encoding="utf-8")
    harness_log.write(
        f"\n===== harness start {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(port {args.harness_port}) =====\n"
    )
    harness_log.flush()
    harness = subprocess.Popen(
        ["node", "--import", "tsx/esm", "apps/cli/src/bin.ts", "web", "--patch", str(PATCH)],
        cwd=HARNESS_ROOT, env=harness_env,
        stdout=harness_log, stderr=subprocess.STDOUT,
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

    reported: set[str] = set()
    try:
        while True:
            alive = [(name, proc) for name, proc in processes if proc.poll() is None]
            for name, process in processes:
                if process.poll() is None or name in reported:
                    continue
                reported.add(name)
                other = "console" if name == "harness" else "harness"
                print(f"\n{name} 已退出：{_exit_description(process.returncode)}", file=sys.stderr)
                if name == "harness":
                    tail = harness_log_path.read_text(errors="replace").strip().splitlines()
                    for line in tail[-8:]:
                        print(f"  {line}", file=sys.stderr)
                    print(f"  （完整输出：{harness_log_path}）", file=sys.stderr)

                # The surviving half is left running on purpose. Tearing it down
                # turns one crash into a total outage, and it destroys the one
                # thing still working: after a console segfault the harness was
                # killed too, so an unrelated threading bug looked like "the
                # harness keeps dying" and the evidence went with it.
                if alive:
                    url = f"http://127.0.0.1:{args.console_port}"
                    print(
                        f"\n{other} 仍在运行，未受影响："
                        + (url if other == "console" else f"端口 {args.harness_port}")
                        + "\n按 Ctrl-C 停止全部。",
                        file=sys.stderr,
                    )

            # Keep waiting while anything is still alive. Exiting this loop on the
            # first death and falling through to `shutdown()` would kill the
            # survivor immediately — which is exactly the behaviour the paragraph
            # above says it is not doing.
            if not alive:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
