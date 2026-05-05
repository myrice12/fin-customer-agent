"""
全链路追踪 — OpenTelemetry集成
为每个Agent调用创建Span，记录延迟、Token消耗、路由决策等关键指标。
支持导出到Jaeger/Zipkin/LangSmith等后端。
包含 Metrics 和 Trace-Log 关联。
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

logger = logging.getLogger(__name__)

_tracer = None
_meter = None

# OTel Metrics instruments (initialized in init_tracer)
_request_duration = None
_request_counter = None
_error_counter = None
_token_counter = None


def init_tracer(
    service_name: str = "fin-customer-agent",
    otlp_endpoint: str | None = None,
) -> None:
    """
    初始化OpenTelemetry追踪器和Metrics。

    Args:
        service_name: 服务名称
        otlp_endpoint: OTLP收集器地址，None则输出到控制台
    """
    global _tracer, _meter, _request_duration, _request_counter, _error_counter, _token_counter

    if not _HAS_OTEL:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    disabled_values = {"", "none", "null", "false", "disabled", "off"}
    if otlp_endpoint and otlp_endpoint.strip().lower() not in disabled_values:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        except ImportError:
            exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        pass
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)

    # Initialize Metrics
    _meter = metrics.get_meter(service_name)
    _request_duration = _meter.create_histogram(
        "agent.request.duration_ms", unit="ms", description="Agent request duration"
    )
    _request_counter = _meter.create_counter(
        "agent.request.count", description="Total agent requests"
    )
    _error_counter = _meter.create_counter(
        "agent.error.count", description="Total agent errors"
    )
    _token_counter = _meter.create_counter(
        "llm.token.usage", description="LLM token usage"
    )

    # Trace-Log correlation: inject trace_id into log records
    _setup_log_correlation()


def _setup_log_correlation():
    """将 trace_id/span_id 注入到 logging 中"""
    class TraceContextFilter(logging.Filter):
        def filter(self, record):
            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.trace_id:
                record.trace_id = format(ctx.trace_id, '032x')
                record.span_id = format(ctx.span_id, '016x')
            else:
                record.trace_id = ""
                record.span_id = ""
            return True

    root_logger = logging.getLogger()
    if not any(isinstance(f, TraceContextFilter) for f in root_logger.filters):
        root_logger.addFilter(TraceContextFilter())


def shutdown_tracer():
    """优雅关闭：刷新所有待发送的 span"""
    if not _HAS_OTEL:
        return
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, 'shutdown'):
            provider.shutdown()
    except Exception as e:
        logger.warning("Failed to shutdown tracer: %s", e)


def get_tracer():
    """获取全局Tracer实例"""
    global _tracer
    if _tracer is None:
        if _HAS_OTEL:
            _tracer = trace.get_tracer("fin-customer-agent")
        else:
            return None
    return _tracer


def trace_agent_call(agent_name: str) -> Callable:
    """
    Agent调用追踪装饰器。

    为每个Agent方法创建一个Span，记录：
    - agent.name: Agent名称
    - agent.duration_ms: 调用耗时
    - agent.input_size: 输入大小
    - agent.success: 是否成功

    用法：
        @trace_agent_call("knowledge_rag")
        async def process(self, state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()

            if tracer is None:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    get_agent_metrics().record_call(agent_name, duration_ms, True)
                    return result
                except Exception:
                    duration_ms = (time.time() - start_time) * 1000
                    get_agent_metrics().record_call(agent_name, duration_ms, False)
                    raise

            span_name = f"agent.{agent_name}.{func.__name__}"

            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("agent.method", func.__name__)

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    span.set_attribute("agent.duration_ms", duration_ms)
                    span.set_attribute("agent.success", True)
                    get_agent_metrics().record_call(agent_name, duration_ms, True)

                    if isinstance(result, dict):
                        span.set_attribute("agent.result_keys", str(list(result.keys())))

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("agent.duration_ms", duration_ms)
                    span.set_attribute("agent.success", False)
                    span.set_attribute("agent.error", str(e))
                    get_agent_metrics().record_call(agent_name, duration_ms, False)
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


class AgentMetrics:
    """Agent调用指标收集器（线程安全）"""

    def __init__(self):
        self._lock = threading.Lock()
        self._call_counts: dict[str, int] = {}
        self._total_duration: dict[str, float] = {}
        self._error_counts: dict[str, int] = {}

    def record_call(self, agent_name: str, duration_ms: float, success: bool):
        with self._lock:
            self._call_counts[agent_name] = self._call_counts.get(agent_name, 0) + 1
            self._total_duration[agent_name] = self._total_duration.get(agent_name, 0.0) + duration_ms
            if not success:
                self._error_counts[agent_name] = self._error_counts.get(agent_name, 0) + 1

        # Also record to OTel metrics if available
        if _request_counter:
            _request_counter.add(1, {"agent": agent_name})
        if _request_duration:
            _request_duration.record(duration_ms, {"agent": agent_name})
        if not success and _error_counter:
            _error_counter.add(1, {"agent": agent_name})

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            summary = {}
            for agent_name in self._call_counts:
                calls = self._call_counts[agent_name]
                total_ms = self._total_duration[agent_name]
                errors = self._error_counts.get(agent_name, 0)
                summary[agent_name] = {
                    "total_calls": calls,
                    "avg_duration_ms": total_ms / calls if calls > 0 else 0,
                    "error_rate": errors / calls if calls > 0 else 0,
                }
            return summary


_agent_metrics = AgentMetrics()


def get_agent_metrics() -> AgentMetrics:
    """获取进程内Agent指标聚合器"""
    return _agent_metrics
