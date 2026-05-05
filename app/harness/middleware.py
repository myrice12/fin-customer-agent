"""
FastAPI 中间件 — 请求ID、全局异常处理

为每个请求注入唯一 ID，统一异常响应格式，防止信息泄露。
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """注册所有中间件"""

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """为每个请求注入唯一 X-Request-ID"""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常处理：返回统一格式，不暴露内部细节"""
        logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )
