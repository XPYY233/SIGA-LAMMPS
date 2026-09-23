"""SSH/SFTP transport for the HPC layer, confined to one remote workspace.

The safety requirement is that the agent never gets an unrestricted remote
shell. That is enforced structurally rather than by policy:

* **Every remote command is built here, from a fixed set of shapes.** Callers do
  not pass command strings. They call `list_dir`, `read_text`, `submit`, and so
  on, and each one composes its own argv internally. There is no method that
  forwards an arbitrary string.
* **Every path is resolved and asserted to be under the workspace root** after
  normalisation, so `..` and absolute paths cannot escape it.
* **Authentication is delegated entirely to the user's own SSH configuration.**
  We resolve the `~/.ssh/config` alias and never read, store, or transmit key
  material, passphrases, or passwords. The certificate file beside the key is
  picked up by the SSH library, which is why certificate-based clusters work
  without us knowing anything about certificates.

Preflight distinguishes an authentication failure from every other kind, because
on this cluster certificate expiry is routine rather than exceptional — the
certificate is valid for about a month. A generic connection error would send
someone hunting for a network problem that does not exist.
"""

from __future__ import annotations

import posixpath
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import paramiko

from config.loader import HpcSettings

__all__ = [
    "HpcClient",
    "HpcError",
    "HpcAuthError",
    "HpcUnreachableError",
    "WorkspaceEscapeError",
    "RemoteFile",
    "DEFAULT_RESULT_SUFFIXES",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
]

# What a LAMMPS run leaves behind that is worth carrying home: the log with its
# thermo table, trajectories for visualisation, and structured data. The slurm
# script and job stdout are handled separately by `hpc_read_log`, and `.err`
# files are almost always empty on success, so fetching them adds noise.
DEFAULT_RESULT_SUFFIXES = (
    ".dump",
    ".lammpstrj",
    ".xyz",
    ".cfg",
    ".data",
    ".dat",
    ".csv",
    ".lammps",
    ".log",
    ".out",
    ".txt",
    ".restart",
)

# A dump grows with atom count times steps. These bounds keep one careless
# trajectory from filling the local disk or stalling the console; anything over
# them is reported as skipped with its size, so the decision is visible.
DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 128 * 1024 * 1024


class HpcError(RuntimeError):
    """A remote operation failed."""


class HpcAuthError(HpcError):
    """Authentication was refused — almost always an expired certificate.

    Kept distinct because the remedy is specific and actionable, and because
    reporting it as a generic connection error wastes the reader's time.
    """


class HpcUnreachableError(HpcError):
    """The host could not be reached at all."""


class WorkspaceEscapeError(HpcError):
    """A path resolved outside the configured remote workspace."""


@dataclass(frozen=True)
class RemoteFile:
    """One entry in a remote listing."""

    name: str
    size: int
    is_dir: bool

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "is_dir": self.is_dir}


class HpcClient:
    """A confined SSH/SFTP session against one cluster.

    Use as a context manager; the connection is closed on exit even when an
    operation raises.
    """

    def __init__(self, settings: HpcSettings, *, timeout: int = 30) -> None:
        self.settings = settings
        self.timeout = timeout
        self.workspace = settings.workspace.rstrip("/")
        self._ssh: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    # -- connection --------------------------------------------------------- #

    def _ssh_config(self) -> dict[str, Any]:
        """Resolve the host alias through the user's own SSH configuration.

        The alias is the whole point: it carries the real hostname, user, port,
        identity file and certificate file, and the user maintains it. We read it
        rather than duplicating any of it into project configuration.
        """
        config = paramiko.SSHConfig()
        path = Path("~/.ssh/config").expanduser()
        if path.is_file():
            with path.open() as handle:
                config.parse(handle)
        return dict(config.lookup(self.settings.host))

    def connect(self) -> HpcClient:
        """Open the connection.

        Raises:
            HpcAuthError: authentication was refused.
            HpcUnreachableError: the host could not be reached.
        """
        if self._ssh is not None:
            return self

        resolved = self._ssh_config()
        hostname = resolved.get("hostname", self.settings.host)
        username = resolved.get("user") or self.settings.user
        port = int(resolved.get("port", self.settings.port or 22))
        identity = resolved.get("identityfile")
        if isinstance(identity, str):
            identity = [identity]

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=hostname,
                username=username,
                port=port,
                key_filename=identity,
                timeout=self.timeout,
                allow_agent=True,
                look_for_keys=False,
            )
        except paramiko.AuthenticationException as exc:
            raise HpcAuthError(
                f"authentication refused by {self.settings.host}. On this cluster the "
                "usual cause is an expired SSH certificate; renew it, then retry. "
                f"({exc})"
            ) from exc
        except (paramiko.SSHException, OSError) as exc:
            raise HpcUnreachableError(f"cannot reach {self.settings.host}: {exc}") from exc

        self._ssh = client
        return self

    def close(self) -> None:
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        if self._ssh is not None:
            self._ssh.close()
            self._ssh = None

    def __enter__(self) -> HpcClient:
        return self.connect()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def sftp(self) -> paramiko.SFTPClient:
        if self._ssh is None:
            self.connect()
        if self._sftp is None:
            assert self._ssh is not None
            self._sftp = self._ssh.open_sftp()
        return self._sftp

    # -- confinement -------------------------------------------------------- #

    def resolve(self, subpath: str = "") -> str:
        """Resolve *subpath* under the workspace root.

        Raises:
            WorkspaceEscapeError: the result would fall outside the workspace.
                Refusing rather than clamping, because a silently corrected path
                would hide that the caller asked for something it should not.
        """
        subpath = (subpath or "").strip()
        if subpath.startswith("/"):
            raise WorkspaceEscapeError(
                f"remote paths must be relative to the workspace, got absolute {subpath!r}"
            )
        candidate = posixpath.normpath(posixpath.join(self.workspace, subpath))
        if candidate != self.workspace and not candidate.startswith(f"{self.workspace}/"):
            raise WorkspaceEscapeError(
                f"{subpath!r} resolves to {candidate!r}, outside the workspace {self.workspace!r}"
            )
        return candidate

    # -- fixed-shape operations --------------------------------------------- #

    def _exec(self, argv: Iterable[str], *, timeout: int | None = None) -> tuple[int, str, str]:
        """Run one command built from already-validated parts.

        Arguments are shell-quoted individually, so a filename containing a
        space, a quote, or a semicolon is passed through as data rather than
        being interpreted.
        """
        if self._ssh is None:
            self.connect()
        assert self._ssh is not None
        command = " ".join(shlex.quote(part) for part in argv)
        _stdin, stdout, stderr = self._ssh.exec_command(command, timeout=timeout or self.timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return stdout.channel.recv_exit_status(), out, err

    def check_alive(self) -> str:
        """Confirm the session works and report the host it landed on."""
        code, out, err = self._exec(["hostname"])
        if code != 0:
            raise HpcError(f"hostname failed: {err.strip()}")
        return out.strip()

    def ensure_workspace(self) -> str:
        """Create the workspace directory if absent, and confirm it is writable.

        Returns:
            The workspace path.

        Raises:
            HpcError: the workspace cannot be created or is not writable.
        """
        workspace = self.resolve("")
        code, _out, err = self._exec(["mkdir", "-p", workspace])
        if code != 0:
            raise HpcError(f"cannot create workspace {workspace}: {err.strip()}")
        if not self.is_dir(""):
            raise HpcError(f"workspace {workspace} exists but is not a directory")
        return workspace

    def is_dir(self, subpath: str = "") -> bool:
        try:
            return stat.S_ISDIR(self.sftp.stat(self.resolve(subpath)).st_mode or 0)
        except FileNotFoundError:
            return False

    def list_dir(self, subpath: str = "") -> list[RemoteFile]:
        """List one workspace directory. Never recurses."""
        target = self.resolve(subpath)
        try:
            entries = self.sftp.listdir_attr(target)
        except FileNotFoundError as exc:
            raise HpcError(f"no such remote directory: {target}") from exc
        return [
            RemoteFile(
                name=entry.filename,
                size=int(entry.st_size or 0),
                is_dir=stat.S_ISDIR(entry.st_mode or 0),
            )
            for entry in sorted(entries, key=lambda e: e.filename)
        ]

    def _mkdir_abs(self, absolute: str) -> None:
        """Create a directory given an already-resolved absolute path.

        Separate from `make_dir`, which takes a workspace-relative path and
        re-resolves it. `upload_tree` composes absolute paths as it walks, so
        feeding those back into `make_dir` tripped the escape check and refused
        every upload — a bug that only a live transfer could reveal.
        """
        code, _out, err = self._exec(["mkdir", "-p", absolute])
        if code != 0:
            raise HpcError(f"cannot create {absolute}: {err.strip()}")

    def make_dir(self, subpath: str) -> str:
        target = self.resolve(subpath)
        self._mkdir_abs(target)
        return target

    def read_text(self, subpath: str, *, max_bytes: int = 256_000) -> str:
        """Read one remote file, truncated to a bounded size.

        The bound exists because a LAMMPS log or dump can be enormous, and
        pulling one into memory to show a status line would be a self-inflicted
        outage. Truncation is reported in the returned text rather than silent.
        """
        target = self.resolve(subpath)
        try:
            handle = self.sftp.open(target, "rb")
        except FileNotFoundError as exc:
            raise HpcError(f"no such remote file: {target}") from exc
        with handle:
            data = handle.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        text = data[:max_bytes].decode("utf-8", "replace")
        if truncated:
            text += f"\n\n[... truncated at {max_bytes} bytes; the remote file is larger ...]"
        return text

    def upload_file(self, local: Path | str, subpath: str) -> str:
        """Upload one file into the workspace."""
        local = Path(local)
        if not local.is_file():
            raise HpcError(f"local file not found: {local}")
        target = self.resolve(subpath)
        parent = posixpath.dirname(target)
        if parent and parent != self.workspace:
            self._mkdir_abs(parent)
        self.sftp.put(str(local), target)
        return target

    def upload_tree(
        self,
        local_dir: Path | str,
        subpath: str = "",
        *,
        exclude: Iterable[str] = (),
    ) -> list[str]:
        """Upload a directory into the workspace, returning the remote paths.

        Symlinks are skipped rather than followed: a link pointing outside the
        workspace would otherwise upload something the caller never intended to
        send.
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise HpcError(f"local directory not found: {local_dir}")
        skip = set(exclude)
        remote_root = self.resolve(subpath)
        self._exec(["mkdir", "-p", remote_root])

        uploaded: list[str] = []
        for path in sorted(local_dir.rglob("*")):
            if path.is_symlink() or any(part in skip for part in path.parts):
                continue
            relative = path.relative_to(local_dir).as_posix()
            remote = posixpath.join(remote_root, relative)
            if path.is_dir():
                self._mkdir_abs(remote)
            elif path.is_file():
                self._mkdir_abs(posixpath.dirname(remote))
                self.sftp.put(str(path), remote)
                uploaded.append(remote)
        # A directory with no subdirectories never hits the loop's dir branch,
        # so make sure the root itself exists.
        self._mkdir_abs(remote_root)
        return uploaded

    def write_text(self, subpath: str, text: str) -> str:
        """Write one text file into the workspace."""
        target = self.resolve(subpath)
        parent = posixpath.dirname(target)
        if parent and parent != self.workspace:
            self._mkdir_abs(parent)
        with self.sftp.open(target, "w") as handle:
            handle.write(text)
        return target

    def download_file(
        self,
        subpath: str,
        local: Path | str,
        *,
        max_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> int:
        """Download one remote file into *local*, returning the bytes written.

        The size is checked before the transfer starts rather than after: a
        truncated dump is worse than no dump, because it looks like a complete
        trajectory to every downstream reader.
        """
        target = self.resolve(subpath)
        try:
            size = int(self.sftp.stat(target).st_size or 0)
        except FileNotFoundError as exc:
            raise HpcError(f"no such remote file: {target}") from exc
        if size > max_bytes:
            raise HpcError(
                f"{subpath} is {size} bytes, over the {max_bytes} byte limit for one file"
            )
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(target, str(local))
        return local.stat().st_size

    def fetch_results(
        self,
        subpath: str,
        local_dir: Path | str,
        *,
        suffixes: Iterable[str] = DEFAULT_RESULT_SUFFIXES,
        only: Iterable[str] = (),
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        exclude: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Bring a finished run's output files back to the local workspace.

        This is the return half of the HPC round trip. Without it a run could be
        submitted, polled and its log read, but its trajectory files stayed on
        the cluster — so the local file list showed only the input script and a
        completed run looked like it had produced nothing.

        Getting *everything* is often the wrong move, though: a trajectory can
        run to hundreds of megabytes, and most of it is never looked at. Passing
        `only` fetches named files one at a time, so a caller can list the remote
        directory and pull back just the log it wants to read or the one dump it
        wants to visualise.

        Selection is by suffix and every exclusion is reported with a reason
        rather than dropped quietly, because the interesting failure is a result
        file that did not come back and nobody noticing.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        wanted = {s.lower() for s in suffixes}
        named = set(only)
        if named:
            # An explicit request overrides the suffix filter: the caller has
            # already decided this file matters, whatever it is called.
            wanted = set()
        skip = set(exclude)

        fetched: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        total = 0

        for entry in self.list_dir(subpath):
            if named and entry.name not in named:
                continue
            if entry.is_dir:
                skipped.append({"name": entry.name, "reason": "directory"})
                continue
            if entry.name in skip:
                skipped.append({"name": entry.name, "reason": "excluded"})
                continue
            if not named and Path(entry.name).suffix.lower() not in wanted:
                skipped.append({"name": entry.name, "reason": "suffix not a result type"})
                continue
            if entry.size == 0:
                skipped.append({"name": entry.name, "reason": "empty"})
                continue
            if entry.size > max_file_bytes:
                skipped.append(
                    {
                        "name": entry.name,
                        "reason": f"{entry.size} bytes exceeds the per-file limit {max_file_bytes}",
                    }
                )
                continue
            if total + entry.size > max_total_bytes:
                skipped.append(
                    {
                        "name": entry.name,
                        "reason": f"would exceed the total budget {max_total_bytes}",
                    }
                )
                continue

            destination = local_dir / entry.name
            written = self.download_file(
                posixpath.join(subpath, entry.name), destination, max_bytes=max_file_bytes
            )
            total += written
            fetched.append({"name": entry.name, "bytes": written, "local": str(destination)})

        missing = sorted(named - {f["name"] for f in fetched} - {s["name"] for s in skipped})
        for name in missing:
            skipped.append({"name": name, "reason": "not present in the remote directory"})

        return {
            "remote_dir": self.resolve(subpath),
            "local_dir": str(local_dir),
            "fetched": fetched,
            "skipped": skipped,
            "total_bytes": total,
        }
