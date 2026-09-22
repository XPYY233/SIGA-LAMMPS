"""Layer 6 — the researcher-facing web application.

`create_app()` returns the FastAPI application. It binds loopback only: it
proxies the harness's `/api`, whose own fence is a DNS-rebinding guard rather
than an authentication layer, so exposing this app more widely would hand out the
ability to start sessions that run commands as the local user.
"""

from web.backend.app import Run, create_app

__all__ = ["Run", "create_app"]
