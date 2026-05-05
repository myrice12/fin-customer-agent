"""
合规审查Agent — 金融/电商场景合规检查
负责对所有Agent的回复进行合规审查，包括：
- 敏感词检测
- PII（个人身份信息）保护
- 金融合规用语检查
- 越权承诺检测
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import asyncio
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from app.tracing.otel_config import trace_agent_call

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    """合规审查结果"""
    passed: bool
    risk_level: str  # low, medium, high, critical
    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    sanitized_content: str = ""


SENSITIVE_PATTERNS = {
    "phone": r"1[3-9]\d{9}",
    "id_card": r"\d{17}[\dXx]",
    "bank_card": r"\b\d{16,19}\b",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "verification_code": r"(验证码|校验码|动态码)[^\d]{0,8}\d{4,8}",
}

FORBIDDEN_TERMS = [
    "保证收益", "稳赚不赔", "零风险", "保本保息",
    "最高收益", "承诺回报", "固定回报",
    "内部消息", "内幕", "暗箱操作",
]

SEMANTIC_RISK_PATTERNS = {
    "收益或本金确定性承诺": [
        r"(肯定|一定|绝对|保证).{0,8}(不会亏|不亏|安全|达到|通过|全额到账)",
        r"(本金|本息).{0,8}(绝对|一定|肯定).{0,8}(安全|保障)",
        r"(过往收益).{0,8}(未来收益)",
        r"(保本).{0,8}(无需|不用).{0,8}风险提示",
    ],
    "适当性或风控绕过": [
        r"(绕过|跳过|规避).{0,8}(测评|风险测评|适当性评估|审核)",
        r"(高风险产品).{0,8}(包装|说成).{0,8}(稳健|低风险)",
        r"(弱化|不提示|隐藏).{0,8}(风险|手续费|费用)",
    ],
    "敏感凭证处理不当": [
        r"(保存|记录|公开|完整回显|完整返回).{0,12}(密码|验证码|证件|身份证|银行卡)",
        r"(验证码|身份证照片|银行卡密码).{0,12}(发给|发送|公开|保存|记录)",
    ],
}

COMPLIANCE_SYSTEM_PROMPT = """你是一个金融合规审查Agent，负责审查客服回复内容的合规性。

审查维度：
1. 是否包含违规金融用语（如"保证收益"、"零风险"等）
2. 是否泄露用户PII信息（手机号、身份证号、银行卡号）
3. 是否存在越权承诺（如擅自承诺退款/赔偿金额）
4. 是否符合金融监管要求（风险提示、免责声明）
5. 是否包含歧视性、侮辱性内容

请以JSON格式返回审查结果：
{
    "passed": true/false,
    "risk_level": "low|medium|high|critical",
    "violations": ["违规项描述"],
    "suggestions": ["修改建议"]
}
"""


class ComplianceCheckerAgent:
    """合规审查Agent"""

    def __init__(self, llm: "ChatOpenAI"):
        self.llm = llm

    def _rule_based_check(self, content: str) -> list[str]:
        """基于规则的快速检查（不依赖LLM，低延迟）"""
        violations = []

        for term in FORBIDDEN_TERMS:
            if term in content:
                violations.append(f"包含违规金融用语: '{term}'")

        for risk_name, patterns in SEMANTIC_RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    violations.append(f"命中语义风险模式: {risk_name}")
                    break

        id_card_spans = [
            match.span()
            for match in re.finditer(SENSITIVE_PATTERNS["id_card"], content)
        ]
        for pii_type, pattern in SENSITIVE_PATTERNS.items():
            matches = list(re.finditer(pattern, content))
            if pii_type in ("phone", "bank_card"):
                matches = [
                    match for match in matches
                    if not any(
                        match.start() < end and match.end() > start
                        for start, end in id_card_spans
                    )
                ]
            if matches:
                label = {
                    "phone": "手机号", "id_card": "身份证号",
                    "bank_card": "银行卡号", "email": "邮箱地址",
                    "verification_code": "验证码",
                }.get(pii_type, pii_type)
                violations.append(f"检测到PII信息泄露: {label}")

        return violations

    def _mask_pii(self, content: str) -> str:
        """对PII信息进行脱敏处理"""
        masked = content
        for pii_type, pattern in SENSITIVE_PATTERNS.items():
            def _mask_match(match):
                text = match.group()
                if len(text) <= 4:
                    return "****"
                return text[:3] + "*" * (len(text) - 6) + text[-3:]
            masked = re.sub(pattern, _mask_match, masked)
        return masked

    async def answer_compliance_question(self, user_message: str) -> str:
        """直接回答用户提出的金融合规类咨询。"""
        rule_result = await self.rule_check(user_message)

        has_forbidden = any("违规金融用语" in violation for violation in rule_result.violations)
        has_pii = any("PII" in violation for violation in rule_result.violations)

        if has_forbidden:
            return (
                "不能承诺“保证收益”“稳赚不赔”“零风险”等结果。"
                "理财产品不等同于存款，收益会受到市场波动、产品期限、风险等级等因素影响。"
                "建议您以产品合同、风险揭示书和适当性评估结果为准，并根据自身风险承受能力谨慎决策。"
            )

        if has_pii:
            return (
                "您提供的信息中可能包含手机号、身份证号、银行卡号或邮箱等敏感个人信息。"
                "为保护账户安全，请不要在公开对话中发送完整证件号、银行卡号或验证码。"
                f"已识别内容将按脱敏方式处理：{rule_result.sanitized_content}"
            )

        return (
            "您的问题涉及金融合规要求。客服回复不能作出保本、保收益、零风险或确定性赔付承诺。"
            "如需办理具体业务，请以合同条款、官方公告和人工审核结果为准。"
        )

    @trace_agent_call("compliance_rule_check")
    async def rule_check(self, content: str) -> ComplianceResult:
        """规则引擎快速检查"""
        violations = self._rule_based_check(content)
        sanitized = self._mask_pii(content)

        if not violations:
            return ComplianceResult(
                passed=True,
                risk_level="low",
                sanitized_content=sanitized,
            )

        has_pii = any("PII" in v for v in violations)
        has_forbidden = any("违规金融用语" in v for v in violations)
        has_semantic_risk = any("语义风险模式" in v for v in violations)

        if has_pii and has_forbidden:
            risk_level = "critical"
        elif has_pii or has_forbidden or has_semantic_risk:
            risk_level = "high"
        else:
            risk_level = "medium"

        return ComplianceResult(
            passed=False,
            risk_level=risk_level,
            violations=violations,
            sanitized_content=sanitized,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_llm(self, messages):
        return await asyncio.wait_for(self.llm.ainvoke(messages), timeout=30.0)

    @trace_agent_call("compliance_llm_check")
    async def llm_check(self, content: str) -> ComplianceResult:
        """LLM深度合规审查（处理规则引擎无法覆盖的场景）"""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
            HumanMessage(content=f"请审查以下客服回复内容的合规性：\n\n{content}"),
        ]

        try:
            response = await self._call_llm(messages)
        except Exception as e:
            logger.warning("Compliance LLM check failed: %s", e)
            return ComplianceResult(
                passed=False, risk_level="high",
                violations=[f"LLM合规审查调用失败: {type(e).__name__}"],
                suggestions=["转人工坐席处理"],
                sanitized_content=self._mask_pii(content),
            )

        import json
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            return ComplianceResult(
                passed=False,
                risk_level="high",
                violations=["LLM合规审查响应解析异常，触发人工审核"],
                suggestions=["转人工坐席处理"],
                sanitized_content=self._mask_pii(content),
            )

        return ComplianceResult(
            passed=result.get("passed", True),
            risk_level=result.get("risk_level", "low"),
            violations=result.get("violations", []),
            suggestions=result.get("suggestions", []),
            sanitized_content=self._mask_pii(content),
        )

    @trace_agent_call("compliance_full_check")
    async def full_check(self, content: str) -> ComplianceResult:
        """
        两阶段合规审查：
        1. 规则引擎快速检查（毫秒级）
        2. 若规则通过，再进行LLM深度审查
        """
        rule_result = await self.rule_check(content)

        if not rule_result.passed and rule_result.risk_level in ("high", "critical"):
            return rule_result

        llm_result = await self.llm_check(content)

        all_violations = rule_result.violations + llm_result.violations
        final_passed = rule_result.passed and llm_result.passed

        risk_priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        final_risk = max(
            rule_result.risk_level, llm_result.risk_level,
            key=lambda r: risk_priority.get(r, 0),
        )

        return ComplianceResult(
            passed=final_passed,
            risk_level=final_risk,
            violations=all_violations,
            suggestions=llm_result.suggestions,
            sanitized_content=rule_result.sanitized_content,
        )

    @trace_agent_call("compliance_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """作为Graph节点处理状态"""
        try:
            return await self._process_impl(state)
        except Exception as e:
            logger.error("Compliance checker process failed: %s", e, exc_info=True)
            return {
                **state,
                "compliance_passed": False,
                "sub_results": {
                    **state.get("sub_results", {}),
                    "compliance": {
                        "passed": False,
                        "risk_level": "critical",
                        "violations": [f"合规审查系统异常: {type(e).__name__}"],
                        "error": str(e),
                    },
                },
            }

    async def _process_impl(self, state: dict[str, Any]) -> dict[str, Any]:
        """合规审查实际实现"""
        sub_results = state.get("sub_results", {})

        content_to_check = ""
        for agent_name, result in sub_results.items():
            if isinstance(result, str):
                content_to_check += result + "\n"

        if not content_to_check.strip():
            messages = state.get("messages", [])
            user_message = messages[-1].content if messages else ""
            direct_answer = await self.answer_compliance_question(user_message)
            return {
                **state,
                "compliance_passed": True,
                "sub_results": {
                    **sub_results,
                    "compliance_checker": direct_answer,
                    "compliance": {
                        "passed": True,
                        "risk_level": "low",
                        "violations": [],
                    },
                },
            }

        compliance_result = await self.full_check(content_to_check)

        sanitized_results = dict(sub_results)
        if not compliance_result.passed:
            for key in sanitized_results:
                if isinstance(sanitized_results[key], str):
                    sanitized_results[key] = compliance_result.sanitized_content

        return {
            **state,
            "compliance_passed": compliance_result.passed,
            "sub_results": {
                **sanitized_results,
                "compliance": {
                    "passed": compliance_result.passed,
                    "risk_level": compliance_result.risk_level,
                    "violations": compliance_result.violations,
                },
            },
        }
