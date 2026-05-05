"""
Skill能力注册器。

Skill用于沉淀可复用的业务SOP、合规话术、UI规范或任务执行方法。
支持基于 Agent 的访问控制（ACL）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    """可被Agent引用的能力说明。"""

    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    entrypoint: str = ""
    path: str = ""
    allowed_agents: list[str] = field(default_factory=list)  # 空=全部可用
    usage_count: int = 0


class SkillRegistry:
    """内存Skill注册与发现，支持访问控制。"""

    def __init__(self, skill_root: str = "skill"):
        self.skill_root = Path(skill_root)
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> bool:
        """注销一个 Skill，返回是否成功"""
        return self._skills.pop(name, None) is not None

    def check_access(self, skill_name: str, agent_name: str) -> bool:
        """检查指定 Agent 是否有权使用该 Skill"""
        skill = self._skills.get(skill_name)
        if not skill:
            return False
        if not skill.allowed_agents:
            return True  # 未设置限制，全部可用
        return agent_name in skill.allowed_agents

    def record_usage(self, skill_name: str) -> None:
        """记录 Skill 使用次数"""
        skill = self._skills.get(skill_name)
        if skill:
            skill.usage_count += 1

    def list_skills(self, tag: str | None = None) -> list[dict]:
        skills = self._skills.values()
        if tag:
            skills = [skill for skill in skills if tag in skill.tags]
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "tags": skill.tags,
                "entrypoint": skill.entrypoint,
                "path": skill.path,
                "allowed_agents": skill.allowed_agents,
                "usage_count": skill.usage_count,
            }
            for skill in skills
        ]

    def find_for_intent(self, intent: str) -> list[dict]:
        """按意图或标签检索Skill。"""
        intent = intent.lower()
        matched = [
            skill for skill in self._skills.values()
            if intent in skill.name.lower() or intent in [tag.lower() for tag in skill.tags]
        ]
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "tags": skill.tags,
                "entrypoint": skill.entrypoint,
                "path": skill.path,
                "allowed_agents": skill.allowed_agents,
                "usage_count": skill.usage_count,
            }
            for skill in matched
        ]


def create_default_skill_registry(skill_root: str = "skill") -> SkillRegistry:
    """创建默认Skill注册表。"""
    registry = SkillRegistry(skill_root=skill_root)
    registry.register(
        Skill(
            name="financial-compliance-sop",
            description="金融客服合规话术、风险提示和禁用承诺表达规范。",
            tags=["compliance", "finance", "customer-service"],
            entrypoint="prompt",
            path="builtin://financial-compliance-sop",
        )
    )
    registry.register(
        Skill(
            name="rag-answering-playbook",
            description="RAG问答流程规范：先检索、再重排、基于引用生成，无法命中时转人工。",
            tags=["knowledge_rag", "rag", "context"],
            entrypoint="prompt",
            path="builtin://rag-answering-playbook",
        )
    )

    ui_skill = Path(skill_root) / "ui-ux-pro-max-skill-cn"
    if ui_skill.exists():
        registry.register(
            Skill(
                name="ui-ux-pro-max-skill-cn",
                description="前端页面和交互体验规范Skill，可用于客服工作台界面设计。",
                tags=["ui", "ux", "frontend"],
                entrypoint="README_ZH.md",
                path=str(ui_skill),
            )
        )

    return registry
