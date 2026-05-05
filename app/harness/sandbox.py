"""
Tool 执行沙箱 — 受控的工具执行环境

提供输入校验、权限检查、超时控制和审计日志，
确保 Tool 调用在安全隔离的环境中执行。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class ToolSandbox:
    """
    Tool 执行沙箱。

    每次工具调用都经过以下受控流程：
    1. 输入校验（required 字段 + 类型检查）
    2. 权限检查（requires_auth 标志）
    3. 带超时的异步执行
    4. 有界审计日志
    """

    def __init__(self, timeout_seconds: float = 30.0, max_audit_log: int = 1000):
        self.timeout = timeout_seconds
        self._audit_log: deque[dict] = deque(maxlen=max_audit_log)

    def validate_input(self, tool_name: str, input_schema: dict, arguments: dict) -> str | None:
        """
        校验工具输入参数。
        返回 None 表示校验通过，返回错误信息字符串表示校验失败。
        """
        required = input_schema.get("required", [])
        properties = input_schema.get("properties", {})

        for field in required:
            if field not in arguments:
                return f"Missing required field: '{field}'"

        for key, value in arguments.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    return f"Field '{key}' expected type '{expected_type}', got {type(value).__name__}"

        return None

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """基本类型检查"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)

    def check_auth(self, requires_auth: bool, caller: dict | None) -> str | None:
        """权限检查。返回 None 表示通过，返回错误信息表示拒绝。"""
        if not requires_auth:
            return None
        if not caller or not caller.get("authenticated"):
            return "Authentication required"
        return None

    async def execute(
        self,
        tool_name: str,
        handler: Any,
        input_schema: dict,
        arguments: dict,
        requires_auth: bool = False,
        caller: dict | None = None,
    ) -> dict:
        """
        在沙箱中执行工具调用。

        Returns:
            {"success": bool, "result": Any, "error": str | None, "duration_ms": float}
        """
        # 1. 输入校验
        validation_error = self.validate_input(tool_name, input_schema, arguments)
        if validation_error:
            return {
                "success": False, "result": None,
                "error": f"Input validation failed: {validation_error}",
                "duration_ms": 0.0,
            }

        # 2. 权限检查
        auth_error = self.check_auth(requires_auth, caller)
        if auth_error:
            return {
                "success": False, "result": None,
                "error": auth_error,
                "duration_ms": 0.0,
            }

        # 3. 带超时执行
        start = time.time()
        try:
            output = await asyncio.wait_for(handler(**arguments), timeout=self.timeout)
            duration_ms = (time.time() - start) * 1000
            result = {"success": True, "result": output, "error": None, "duration_ms": duration_ms}
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start) * 1000
            result = {
                "success": False, "result": None,
                "error": f"Tool '{tool_name}' execution timed out after {self.timeout}s",
                "duration_ms": duration_ms,
            }
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result = {
                "success": False, "result": None,
                "error": f"{type(e).__name__}: {str(e)}",
                "duration_ms": duration_ms,
            }

        # 4. 审计日志
        self._audit(tool_name, arguments, result)
        return result

    def _audit(self, tool_name: str, arguments: dict, result: dict) -> None:
        """记录审计日志（有界队列，自动淘汰旧记录）"""
        self._audit_log.append({
            "tool": tool_name,
            "success": result["success"],
            "duration_ms": result["duration_ms"],
            "error": result.get("error"),
            "timestamp": time.time(),
        })

    def get_audit_log(self, last_n: int = 100) -> list[dict]:
        """获取最近的审计日志"""
        entries = list(self._audit_log)
        return entries[-last_n:]
