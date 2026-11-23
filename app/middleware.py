import contextlib
import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

SECURITY_HEADERS = [
    (
        b"content-security-policy",
        b"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
        b"connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    ),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]
# Swagger UI loads its assets from a CDN, so /docs gets the other headers but not the strict CSP.
_NO_CSP = ("/docs", "/redoc")


class SecurityHeaders:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        skip_csp = scope["path"].startswith(_NO_CSP)

        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = SECURITY_HEADERS[1:] if skip_csp else SECURITY_HEADERS
                message["headers"] = [*message.get("headers", []), *headers]
            await send(message)

        await self.app(scope, receive, wrapped)


class _TooLarge(Exception):
    pass


class BodyLimit:
    """Stops reading once a body passes the limit, whether or not it declared a length."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(send)
        seen, started, over = 0, False, False

        async def limited() -> Message:
            nonlocal seen, over
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    over = True
                    raise _TooLarge
            return message

        async def tracked(message: Message) -> None:
            nonlocal started
            if over and not started:
                return  # the framework turns the aborted read into its own 400; ours is clearer
            started = started or message["type"] == "http.response.start"
            await send(message)

        with contextlib.suppress(_TooLarge):
            await self.app(scope, limited, tracked)
        if over and not started:
            await self._reject(send)

    async def _reject(self, send: Send) -> None:
        mb = self.max_bytes // (1024 * 1024)
        body = json.dumps({"detail": f"The file is larger than {mb} MB."}).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"connection", b"close"),
        ]
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})
