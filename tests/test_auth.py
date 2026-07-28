"""认证依赖单元测试。"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.config import HarnessSettings
from app.harness.auth import verify_api_key


@pytest.mark.asyncio
async def test_verify_api_key_passes_when_not_configured(monkeypatch):
    from app import harness

    monkeypatch.setattr(
        harness.auth,
        "_configured_api_keys",
        lambda: set(),
    )
    await verify_api_key(key=None)


@pytest.mark.asyncio
async def test_verify_api_key_rejects_missing_key_when_configured(monkeypatch):
    from app import harness

    monkeypatch.setattr(
        harness.auth,
        "_configured_api_keys",
        lambda: {"secret"},
    )
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(key=None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_api_key_rejects_invalid_key_when_configured(monkeypatch):
    from app import harness

    monkeypatch.setattr(
        harness.auth,
        "_configured_api_keys",
        lambda: {"secret"},
    )
    with pytest.raises(HTTPException):
        await verify_api_key(key="wrong")


@pytest.mark.asyncio
async def test_verify_api_key_accepts_valid_key_when_configured(monkeypatch):
    from app import harness

    monkeypatch.setattr(
        harness.auth,
        "_configured_api_keys",
        lambda: {"secret"},
    )
    await verify_api_key(key="secret")


def test_harness_settings_parses_csv_api_keys():
    s = HarnessSettings(API_KEYS=" a , b , c ")
    assert s.api_key_set == {"a", "b", "c"}


def test_harness_settings_empty_api_keys():
    s = HarnessSettings(API_KEYS="")
    assert s.api_key_set == set()