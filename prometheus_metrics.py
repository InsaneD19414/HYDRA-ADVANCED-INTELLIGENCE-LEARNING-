"""
observability/prometheus_metrics.py
────────────────────────────────────
Prometheus metrics exporter for the Hydra Sentinel-X FastAPI server.

METRICS EXPOSED (at /metrics):

  COUNTERS (cumulative, never reset):
    hydra_analyses_total{symbol, action}     — completed analysis runs
    hydra_approvals_total{symbol, action, decision}  — approval decisions
    hydra_got_loops_total                    — total GoT retry loops
    hydra_mcp_calls_total{server, tool}      — MCP tool call counts
    hydra_llm_tokens_total{agent, direction} — token usage

  HISTOGRAMS (latency distributions):
    hydra_node_latency_seconds{node}         — per LangGraph node
    hydra_analysis_latency_seconds{symbol}   — full pipeline duration
    hydra_mcp_latency_seconds{server, tool}  — MCP tool round-trip

  GAUGES (current state):
    hydra_pending_approvals                  — current approval queue depth
    hydra_got_generation{session_id}         — current GoT generation
    hydra_got_survival_rate                  — ratio of thoughts that survive critique
    hydra_system_healthy                     — 1 if all subsystems OK

USAGE IN server.py:
    from observability.prometheus_metrics import (
        init_metrics, metrics_endpoint,
        record_analysis, record_node_timing,
        record_approval, set_queue_depth,
    )

    # In lifespan:
    init_metrics()

    # In FastAPI route:
    @app.get("/metrics")
    async def metrics():
        return metrics_endpoint()

    # In analysis handler:
    with record_node_timing("market_analyst"):
        result = market_analyst_node(state)

    # After analysis completes:
    record_analysis(symbol="BTC", action="buy", duration_s=12.4)

    # After approval decision:
    record_approval(symbol="BTC", action="buy", decision="approved")
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

# ── Optional Prometheus client ────────────────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus-client not installed. pip install prometheus-client")


# ── Metric registry (use default REGISTRY) ───────────────────────────────────

class HydraMetrics:
    """
    Singleton container for all Prometheus metric objects.
    Initialised once at server startup via init_metrics().
    """
    _initialised = False

    # ── Counters ──────────────────────────────────────────────────────────────
    analyses_total:    "Counter"
    approvals_total:   "Counter"
    got_loops_total:   "Counter"
    mcp_calls_total:   "Counter"
    llm_tokens_total:  "Counter"
    errors_total:      "Counter"

    # ── Histograms ────────────────────────────────────────────────────────────
    node_latency:      "Histogram"
    analysis_latency:  "Histogram"
    mcp_latency:       "Histogram"

    # ── Gauges ────────────────────────────────────────────────────────────────
    pending_approvals: "Gauge"
    got_generation:    "Gauge"
    survival_rate:     "Gauge"
    system_healthy:    "Gauge"

    @classmethod
    def init(cls) -> None:
        if cls._initialised or not PROMETHEUS_AVAILABLE:
            return

        cls.analyses_total = Counter(
            "hydra_analyses_total",
            "Completed analysis runs",
            ["symbol", "action"],
        )
        cls.approvals_total = Counter(
            "hydra_approvals_total",
            "Approval decisions made",
            ["symbol", "action", "decision"],  # decision: approved|rejected|expired
        )
        cls.got_loops_total = Counter(
            "hydra_got_loops_total",
            "GoT retry loops (all hypotheses pruned, generation incremented)",
        )
        cls.mcp_calls_total = Counter(
            "hydra_mcp_calls_total",
            "MCP tool calls",
            ["server", "tool"],
        )
        cls.llm_tokens_total = Counter(
            "hydra_llm_tokens_total",
            "LLM token usage",
            ["agent", "direction"],  # direction: input|output
        )
        cls.errors_total = Counter(
            "hydra_errors_total",
            "Errors by component",
            ["component"],
        )

        # Latency buckets calibrated for typical LLM response times (seconds)
        LLM_BUCKETS = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]

        cls.node_latency = Histogram(
            "hydra_node_latency_seconds",
            "LangGraph node execution time",
            ["node"],
            buckets=LLM_BUCKETS,
        )
        cls.analysis_latency = Histogram(
            "hydra_analysis_latency_seconds",
            "Full analysis pipeline duration",
            ["symbol"],
            buckets=[5, 10, 20, 30, 45, 60, 90, 120],
        )
        cls.mcp_latency = Histogram(
            "hydra_mcp_latency_seconds",
            "MCP tool round-trip latency",
            ["server", "tool"],
            buckets=[0.1, 0.25, 0.5, 1, 2, 3, 5, 10],
        )

        cls.pending_approvals = Gauge(
            "hydra_pending_approvals",
            "Current number of pending recommendations awaiting human review",
        )
        cls.got_generation = Gauge(
            "hydra_got_generation",
            "Current GoT generation number in the active session",
        )
        cls.survival_rate = Gauge(
            "hydra_got_survival_rate",
            "Ratio of GoT hypotheses that survive adversarial critique (rolling)",
        )
        cls.system_healthy = Gauge(
            "hydra_system_healthy",
            "1 if all core subsystems are healthy, 0 if any are degraded",
        )
        cls.system_healthy.set(1)

        cls._initialised = True
        logger.info("Prometheus metrics initialised — /metrics endpoint active")


# ── Module-level convenience references ──────────────────────────────────────

_m = HydraMetrics


def init_metrics() -> None:
    """Call once at server startup to register all metric objects."""
    HydraMetrics.init()


def metrics_endpoint() -> tuple[bytes, str]:
    """
    Returns (content_bytes, content_type) for the /metrics endpoint.
    Usage in FastAPI:
        content, ctype = metrics_endpoint()
        return Response(content=content, media_type=ctype)
    """
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return b"# Prometheus not available\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST


# ── Recording helpers ─────────────────────────────────────────────────────────

def record_analysis(symbol: str, action: str, duration_s: float) -> None:
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.analyses_total.labels(symbol=symbol.upper(), action=action).inc()
    _m.analysis_latency.labels(symbol=symbol.upper()).observe(duration_s)


def record_approval(symbol: str, action: str, decision: str) -> None:
    """decision: 'approved', 'rejected', or 'expired'"""
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.approvals_total.labels(
        symbol=symbol.upper(), action=action, decision=decision
    ).inc()


def record_got_loop() -> None:
    """Call when the GoT adversarial critique prunes all hypotheses and we retry."""
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.got_loops_total.inc()


def record_mcp_call(server: str, tool: str, duration_s: float) -> None:
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.mcp_calls_total.labels(server=server, tool=tool).inc()
    _m.mcp_latency.labels(server=server, tool=tool).observe(duration_s)


def record_llm_tokens(agent: str, input_tokens: int, output_tokens: int) -> None:
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.llm_tokens_total.labels(agent=agent, direction="input").inc(input_tokens)
    _m.llm_tokens_total.labels(agent=agent, direction="output").inc(output_tokens)


def record_error(component: str) -> None:
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.errors_total.labels(component=component).inc()


def set_queue_depth(depth: int) -> None:
    """Update the pending approvals gauge — call after every queue change."""
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.pending_approvals.set(depth)


def set_got_generation(gen: int) -> None:
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.got_generation.set(gen)


def update_survival_rate(survived: int, total: int) -> None:
    """Update the rolling GoT hypothesis survival rate."""
    if not PROMETHEUS_AVAILABLE or not _m._initialised or total == 0:
        return
    _m.survival_rate.set(round(survived / total, 4))


def set_system_health(healthy: bool) -> None:
    if not PROMETHEUS_AVAILABLE or not _m._initialised:
        return
    _m.system_healthy.set(1 if healthy else 0)


# ── Context manager: time a node ──────────────────────────────────────────────

@contextmanager
def record_node_timing(node_name: str) -> Generator[None, None, None]:
    """
    Context manager that records the execution time of a LangGraph node.

    Usage:
        with record_node_timing("market_analyst"):
            result = market_analyst_node(state)
    """
    start = time.perf_counter()
    error = False
    try:
        yield
    except Exception:
        error = True
        record_error(node_name)
        raise
    finally:
        duration = time.perf_counter() - start
        if PROMETHEUS_AVAILABLE and _m._initialised:
            _m.node_latency.labels(node=node_name).observe(duration)
        logger.debug(f"[METRICS] {node_name}: {duration*1000:.0f}ms {'ERR' if error else 'OK'}")


@contextmanager
def record_mcp_timing(server: str, tool: str) -> Generator[None, None, None]:
    """
    Context manager that records MCP tool round-trip latency.

    Usage:
        with record_mcp_timing("market_intelligence", "get_price"):
            price = await get_price_tool.invoke({"symbol": "BTC"})
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        record_mcp_call(server=server, tool=tool, duration_s=duration)
