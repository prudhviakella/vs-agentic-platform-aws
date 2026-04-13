"""logging_mw.py — Request/response logging middleware."""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        t0   = time.perf_counter()
        resp = await call_next(request)
        ms   = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(
            f"[PLATFORM] {request.method} {request.url.path}"
            f"  status={resp.status_code}  latency_ms={ms}"
        )
        return resp
