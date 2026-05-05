"""
认证与授权 — API Key 认证中间件

提供 API 级别的认证保护。未配置 API Key 时自动跳过（开发模式）。
"""

from __future__ import annotations

import logging
import os

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_valid_api_keys: set[str] = set()


def load_api_keys() -> set[str]:
    """从环境变量加载 API Key 集合"""
    global _valid_api_keys
    keys_str = os.getenv("API_KEYS", "")
    if keys_str:
        _valid_api_keys = {k.strip() for k in keys_str.split(",") if k.strip()}
    return _valid_api_keys


async def verify_api_key(key: str | None = Depends(api_key_header)):
    """
    FastAPI 依赖项：验证 API Key。

    - 未配置 API_KEYS 环境变量时跳过验证（开发模式）
    - 配置后要求所有请求携带有效的 X-API-Key
    """
    if not _valid_api_keys:
        return  # 开发模式，跳过验证
    if not key or key not in _valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
