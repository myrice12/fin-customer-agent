"""合规审查 Agent 单元测试。"""

from __future__ import annotations

import pytest

from app.agents.compliance_checker import ComplianceCheckerAgent


class _FakeLLM:
    async def ainvoke(self, messages):
        class _Resp:
            content = '{"passed": true, "risk_level": "low", "violations": [], "suggestions": []}'
        return _Resp()


@pytest.fixture
def checker() -> ComplianceCheckerAgent:
    return ComplianceCheckerAgent(llm=_FakeLLM())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_rule_check_passes_clean_content(checker: ComplianceCheckerAgent):
    result = await checker.rule_check("理财产品有风险，投资需谨慎。")
    assert result.passed is True
    assert result.risk_level == "low"


@pytest.mark.asyncio
async def test_rule_check_flags_forbidden_terms(checker: ComplianceCheckerAgent):
    result = await checker.rule_check("我们保证收益，稳赚不赔，零风险。")
    assert result.passed is False
    assert result.risk_level in ("high", "critical")


@pytest.mark.asyncio
async def test_pii_detection_with_negative_lookaround(checker: ComplianceCheckerAgent):
    content = "请把验证码 123456 发给我，账号 13800138000，身份证 11010519491231002X"
    result = await checker.rule_check(content)
    assert result.passed is False
    assert any("PII" in v for v in result.violations)


@pytest.mark.asyncio
async def test_pii_does_not_match_digits_embedded_in_longer_number(checker: ComplianceCheckerAgent):
    """20+ 位连续数字不应触发 16-19 位银行卡号规则（lookaround 收紧生效）。"""
    result = await checker.rule_check("参考编号 1234567890123456789012345 已生成。")
    pii_violations = [v for v in result.violations if "银行卡号" in v]
    assert pii_violations == []


@pytest.mark.asyncio
async def test_mask_pii_masks_phone(checker: ComplianceCheckerAgent):
    masked = checker._mask_pii("联系电话 13800138000")
    assert "13800138000" not in masked
    assert "138" in masked and "000" in masked