"""WorkingMemory 异步接口单元测试。"""

from __future__ import annotations

import asyncio

import pytest

from app.memory.working_memory import WorkingMemory


@pytest.mark.asyncio
async def test_update_and_get_context():
    mem = WorkingMemory()
    await mem.update("s1", {"last_intent": "knowledge_rag"})
    ctx = mem.get_context("s1")
    assert ctx["last_intent"] == "knowledge_rag"
    assert mem.session_count == 1


@pytest.mark.asyncio
async def test_concurrent_updates_do_not_exceed_max_entries():
    mem = WorkingMemory(max_entries_per_session=5)

    async def updater(i: int) -> None:
        await mem.update("s", {"i": i})

    await asyncio.gather(*(updater(i) for i in range(20)))
    history = mem.get_history("s", last_n=100)
    assert len(history) == 5


@pytest.mark.asyncio
async def test_cleanup_expired_removes_old_sessions():
    mem = WorkingMemory()
    await mem.update("old", {"x": 1})
    cleaned = await mem.cleanup_expired(max_age_seconds=0)
    assert cleaned == 1
    assert mem.session_count == 0