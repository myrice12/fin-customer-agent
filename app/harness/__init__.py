"""
Harness — Agent 执行环境基础设施层

提供以下组件：
- ToolSandbox: Tool 执行沙箱（输入校验、权限检查、超时、审计）
- verify_api_key: API 认证依赖项
- setup_middleware: 请求ID、全局异常处理
- comprehensive_health_check: 综合健康检查
"""

from app.harness.sandbox import ToolSandbox
from app.harness.auth import verify_api_key, auth_enabled
from app.harness.middleware import setup_middleware
from app.harness.health import comprehensive_health_check

__all__ = [
    "ToolSandbox",
    "verify_api_key",
    "auth_enabled",
    "setup_middleware",
    "comprehensive_health_check",
]
