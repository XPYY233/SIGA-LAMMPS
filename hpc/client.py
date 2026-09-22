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
]


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
