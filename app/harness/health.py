"""
综合健康检查 — 检测各依赖组件的可用性

返回结构化的健康状态，区分 healthy / degraded。
"""

from __future__ import annotations

from typing import Any


async def comprehensive_health_check(
    short_term_memory: Any = None,
    long_term_memory: Any = None,
) -> dict:
    """
    综合健康检查。

    Returns:
        {
            "status": "healthy" | "degraded",
            "redis": bool,
            "faiss": bool,
            "embedding": bool,
        }
    """
    redis_ok = False
    faiss_ok = False
    embedding_ok = False

    if short_term_memory is not None:
        try:
            redis_ok = await short_term_memory.health_check()
        except Exception:
            pass

    if long_term_memory is not None:
        faiss_ok = long_term_memory._index is not None
        embedding_ok = long_term_memory._embedding_available

    all_ok = redis_ok and faiss_ok
    status = "healthy" if all_ok else "degraded"

    return {
        "status": status,
        "redis": redis_ok,
        "faiss": faiss_ok,
        "embedding": embedding_ok,
    }
