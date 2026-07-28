"""
认证与授权 — API Key 认证依赖项

提供 API 级别的认证保护。未配置 API_KEYS 时自动跳过（开发模式）。
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _configured_api_keys() -> set[str]:
    return get_settings().api_key_set


async def verify_api_key(
    key: str | None = Depends(api_key_header),
) -> None:
    """
    FastAPI 依赖项：验证 API Key。

    - 未配置 API_KEYS 环境变量时跳过验证（开发模式）
    - 配置后要求所有请求携带有效的 X-API-Key
    """
    valid_keys = _configured_api_keys()
    if not valid_keys:
        return
    if not key or key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def auth_enabled() -> bool:
    """供启动日志等场景判断当前是否启用了 API Key 认证。"""
    return bool(_configured_api_keys())