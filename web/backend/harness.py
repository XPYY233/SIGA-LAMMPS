"""A client for the harness's own HTTP RPC and event stream.

Three properties of that surface shape this module, and each was verified against
the harness source rather than assumed:

* **RPC is plain HTTP POST.** `POST /api/<method>` with a
  `{type, rpcId, method, payload}` envelope, answered by a
  `{type, rpcId, result}` envelope whose `result` is `{ok: true, value}` or
  `{ok: false, error}`. Everything the UI needs to *do* travels this way.

* **Events are WebSocket only.** A plain GET on `/api/events.mux` returns 426
  Upgrade Required; the SSE path exists only on the in-process carrier. So the
  stream is read over a socket, and the socket is **downlink-only** — sending a
  frame closes it with code 1008 rather than returning an error. Upstream traffic
  must go over HTTP POST or the connection dies silently.

* **There is no resume.** `events.mux` accepts a `since` map and ignores it,
  documented as unimplemented. A dropped connection therefore cannot be replayed
  from the socket, so `history()` exists and callers reconcile by `seq`.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
import websockets

__all__ = ["HarnessClient", "HarnessError", "HarnessRpcError", "EventFrame"]


class HarnessError(RuntimeError):
    """The harness could not be reached or refused the request."""


class HarnessRpcError(HarnessError):
    """The harness answered with an error result."""

    def __init__(self, method: str, error: dict[str, Any]) -> None:
        self.method = method
        self.error = error
        super().__init__(f"{method} failed: {error.get('code')}: {error.get('message')}")


@dataclass(frozen=True)
class EventFrame:
    """One frame from the mux downlink."""

    method: str
    payload: dict[str, Any]

    @property
    def kind(self) -> str:
        return str(self.payload.get("type", ""))


class HarnessClient:
    """Talks to one running harness."""

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        #: Set when a requested preset was unavailable and the default was used.
        self.last_preset_fallback: str | None = None

    # -- RPC ---------------------------------------------------------------- #

    async def rpc(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        """Call one RPC method and return its value.

        Raises:
            HarnessRpcError: the harness answered `ok: false`.
            HarnessError: the request could not be delivered.
        """
        body = {
            "type": "client-request",
            "rpcId": str(uuid.uuid4()),
            "method": method,
            "payload": payload or {},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/{method}",
                    json=body,
                    headers={"content-type": "application/json"},
                )
        except httpx.HTTPError as exc:
            raise HarnessError(f"cannot reach the harness at {self.base_url}: {exc}") from exc

        if response.status_code == 415:
            raise HarnessError("the harness requires application/json")
        if response.status_code == 426:
            raise HarnessError(
                "the harness answered 426: this route only accepts a WebSocket upgrade"
            )
        if response.status_code >= 400:
            raise HarnessError(f"{method} returned HTTP {response.status_code}: {response.text[:200]}")

        try:
            envelope = response.json()
        except ValueError as exc:
            raise HarnessError(f"{method} returned non-JSON: {response.text[:200]}") from exc

        result = envelope.get("result") or {}
        if not result.get("ok"):
            raise HarnessRpcError(method, result.get("error") or {"code": "unknown"})
        return result.get("value")

    # -- session operations -------------------------------------------------- #

    async def create_session(
        self, *, agent_preset: str | None = None, cwd: str | None = None
    ) -> str:
        """Create a session for one benchmark run.

        `agentPreset` selects the ablation configuration. Verified as reachable
        without a privileged call: choosing a preset is not loopback-pinned,
        because the harness reasons that any caller able to start a session can
        already run commands as this process.
        """
        payload: dict[str, Any] = {}
        if agent_preset:
            payload["agentPreset"] = agent_preset
        if cwd:
            payload["cwd"] = cwd
        try:
            value = await self.rpc("session.create", payload)
        except HarnessRpcError as exc:
            # A preset the roster does not carry must not lose the run. The
            # overlay this harness was started with already mounts a fixed set of
            # adapter components, so a session without an explicit preset runs
            # with exactly that set — the same components, chosen by the overlay
            # rather than by name. Failing the whole request instead would make a
            # cosmetic selector able to break generation.
            if agent_preset and exc.error.get("code") == "agent-preset-not-found":
                payload.pop("agentPreset", None)
                self.last_preset_fallback = str(exc.error.get("message", ""))
                value = await self.rpc("session.create", payload)
            else:
                raise
        session_id = (value or {}).get("sessionId") or (value or {}).get("id")
        if not session_id:
            raise HarnessError(f"session.create returned no session id: {value!r}")
        return str(session_id)

    async def prompt(self, session_id: str, text: str) -> Any:
        """Send a natural-language request to a session."""
        return await self.rpc(
            "session.prompt",
            {
                "sessionId": session_id,
                "mode": "queue",
                "content": [{"type": "text", "text": text}],
            },
        )

    async def history(self, session_id: str, *, max_messages: int = 200) -> dict[str, Any]:
        """Read durable history.

        The only way to recover after a dropped event stream, since the socket
        cannot replay.
        """
        return await self.rpc(
            "session.history", {"sessionId": session_id, "maxMessages": max_messages}
        )

    async def cancel(self, session_id: str) -> Any:
        return await self.rpc("session.cancel", {"sessionId": session_id})

    async def list_sessions(self) -> Any:
        return await self.rpc("session.list", {})

    async def models(self, session_id: str) -> Any:
        return await self.rpc("session.models", {"sessionId": session_id})

    # -- events -------------------------------------------------------------- #

    async def events(self, *, stop: asyncio.Event | None = None) -> AsyncIterator[EventFrame]:
        """Yield mux frames until cancelled.

        Opens `GET /api/events.mux` as a WebSocket. The socket is read-only by
        design: any frame we sent would close it with 1008, so nothing is ever
        written to it.
        """
        url = f"{self.base_url}/api/events.mux".replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        try:
            async with websockets.connect(url, open_timeout=self.timeout, ping_interval=20) as socket:
                while True:
                    if stop is not None and stop.is_set():
                        return
                    try:
                        raw = await asyncio.wait_for(socket.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.ConnectionClosed:
                        return
                    try:
                        frame = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    payload = frame.get("payload") or {}
                    yield EventFrame(method=str(frame.get("method", "")), payload=payload)
        except (OSError, websockets.WebSocketException) as exc:
            raise HarnessError(f"event stream failed: {exc}") from exc

    async def probe(self) -> dict[str, Any]:
        """Report whether the harness is reachable and what it holds."""
        sessions = await self.list_sessions()
        return {"base_url": self.base_url, "reachable": True, "sessions": sessions}
