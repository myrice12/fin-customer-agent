"""
短期记忆 — 基于Redis的会话级记忆
存储最近N轮对话上下文，设置TTL自动过期。
适合维护多轮对话的连续性。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    短期记忆：基于Redis的会话缓存。

    特点：
    - Redis存储，支持分布式部署
    - TTL自动过期（默认30分钟）
    - 保留最近N轮对话
    - 支持滑动窗口淘汰
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_turns: int = 20,
        ttl_seconds: int = 1800,
    ):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._redis_url = redis_url
        self._redis: Any = None
        self._fallback_store: dict[str, list] = {}
        self._fallback_activity: dict[str, float] = {}

    async def _get_redis(self):
        """懒加载Redis连接"""
        if self._redis is None:
            if aioredis is None:
                return None
            try:
                self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception as e:
                logger.warning("Redis connection failed, using in-memory fallback: %s", e)
                self._redis = None
        return self._redis

    async def health_check(self) -> bool:
        """检查 Redis 连接是否正常"""
        try:
            r = await self._get_redis()
            if r is not None:
                await r.ping()
                return True
        except Exception:
            pass
        return False

    def _session_key(self, session_id: str) -> str:
        return f"smartcs:short_term:{session_id}"

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加一条对话消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        r = await self._get_redis()

        if r is not None:
            key = self._session_key(session_id)
            await r.rpush(key, json.dumps(message, ensure_ascii=False))
            await r.ltrim(key, -self.max_turns, -1)
            await r.expire(key, self.ttl_seconds)
        else:
            if session_id not in self._fallback_store:
                self._fallback_store[session_id] = []
            self._fallback_store[session_id].append(message)
            if len(self._fallback_store[session_id]) > self.max_turns:
                self._fallback_store[session_id] = self._fallback_store[session_id][-self.max_turns:]
            self._fallback_activity[session_id] = time.time()

    async def get_history(self, session_id: str, last_n: int | None = None) -> list[dict]:
        """获取对话历史"""
        r = await self._get_redis()

        if r is not None:
            key = self._session_key(session_id)
            n = last_n or self.max_turns
            raw = await r.lrange(key, -n, -1)
            return [json.loads(item) for item in raw]
        else:
            history = self._fallback_store.get(session_id, [])
            if last_n:
                return history[-last_n:]
            return list(history)

    async def clear(self, session_id: str) -> None:
        """清除指定session的短期记忆"""
        r = await self._get_redis()
        if r is not None:
            await r.delete(self._session_key(session_id))
        else:
            self._fallback_store.pop(session_id, None)

    async def get_context_window(self, session_id: str, max_tokens: int = 4000) -> str:
        """获取适配上下文窗口大小的对话历史文本"""
        history = await self.get_history(session_id)

        context_parts = []
        estimated_tokens = 0

        for msg in reversed(history):
            msg_text = f"{msg['role']}: {msg['content']}"
            msg_tokens = len(msg_text) // 2  # 粗略估算
            if estimated_tokens + msg_tokens > max_tokens:
                break
            context_parts.insert(0, msg_text)
            estimated_tokens += msg_tokens

        return "\n".join(context_parts)

    async def get_optimized_context(self, session_id: str, max_tokens: int = 4000) -> dict[str, Any]:
        """
        获取带指标的优化上下文。

        返回压缩后的上下文、原始token估算、注入token估算和压缩率，
        方便在Agent路由和/api/metrics中观察上下文治理效果。
        """
        history = await self.get_history(session_id)
        raw_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history)
        raw_tokens = max(1, len(raw_text) // 2)

        context = await self.get_context_window(session_id, max_tokens=max_tokens)
        injected_tokens = len(context) // 2 if context else 0

        return {
            "context": context,
            "raw_tokens": raw_tokens,
            "injected_tokens": injected_tokens,
            "compression_ratio": injected_tokens / raw_tokens if raw_tokens else 0.0,
            "max_tokens": max_tokens,
            "message_count": len(history),
        }

    async def cleanup_expired(self) -> int:
        """清理内存回退存储中过期的session，返回清理数量"""
        now = time.time()
        expired = [
            sid for sid, ts in self._fallback_activity.items()
            if now - ts > self.ttl_seconds
        ]
        for sid in expired:
            self._fallback_store.pop(sid, None)
            self._fallback_activity.pop(sid, None)
        return len(expired)

    @property
    def fallback_session_count(self) -> int:
        """内存回退存储中的活跃session数量"""
        return len(self._fallback_store)
