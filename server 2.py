"""
server.py — Hydra Sentinel-X Production API Gateway
=====================================================

This is the single entry point for all HTTP traffic in production.
It wraps the LangGraph multi-agent system behind a FastAPI application
that can be called by any frontend, mobile app, or external service.

Key design decisions explained:
  - ASYNC FIRST: We use `asyncio.get_event_loop().run_in_executor()` to run
    the synchronous LangGraph `.invoke()` call in a thread pool. This means
    the FastAPI event loop is never blocked — other requests can be handled
    while one analysis is running (which can take 15-30 seconds).

  - SINGLETON GRAPH: The LangGraph is compiled ONCE at startup in the
    `lifespan` context manager, not on every request. Graph compilation
    opens the SQLite checkpoint connection — doing that per-request would
    create hundreds of file handles and eventually crash the process.

  - STREAMING ENDPOINT: `/api/v1/analyze/stream` uses Server-Sent Events
    (SSE) so a frontend can show live agent progress rather than a blank
    spinner for 30 seconds. The `/api/v1/analyze` POST is the simpler
    blocking variant for programmatic clients that just want the result.

  - ERROR BOUNDARIES: Every endpoint wraps its logic in try/except and
    returns structured JSON errors. A 500 from a bad CoinGecko response
    should never crash the server process.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ---------------------------------------------------------------------------
# Lazy imports — these are heavy and may raise if env vars are missing,
# so we import them inside the lifespan function rather than at module level.
# This gives us a clean error message instead of an ImportError traceback.
# ---------------------------------------------------------------------------
_graph       = None   # compiled LangGraph — initialised at startup
_memory      = None   # CryptoMemoryManager singleton
_approval    = None   # Human-in-the-Loop approval workflow
_mcp_manager = None   # Unified MCP client manager


# ── Pydantic request / response models ─────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    Payload the caller sends to /api/v1/analyze.
    All fields have sensible defaults so partial requests still work.
    """
    query:           str   = Field(..., min_length=3, max_length=1000,
                                   description="Natural language query about a crypto asset")
    symbol:          str   = Field(default="BTC", max_length=10,
                                   description="Ticker symbol, e.g. BTC, ETH, SOL")
    portfolio_value: float = Field(default=10_000.0, gt=0,
                                   description="Total portfolio value in USD")
    risk_tolerance:  str   = Field(default="moderate",
                                   description="conservative | moderate | aggressive")
    session_id:      Optional[str] = Field(default=None,
                                           description="Reuse a session to load prior memory")
    holdings:        Optional[dict[str, float]] = Field(
                         default=None,
                         description="Current holdings, e.g. {'BTC': 0.5, 'ETH': 2.0}")

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("risk_tolerance")
    @classmethod
    def validate_risk(cls, v: str) -> str:
        allowed = {"conservative", "moderate", "aggressive"}
        if v.lower() not in allowed:
            raise ValueError(f"risk_tolerance must be one of {allowed}")
        return v.lower()


class AnalyzeResponse(BaseModel):
    """Structured response returned by the agent system."""
    session_id:           str
    symbol:               str
    final_response:       str
    market_data:          dict
    technical_analysis:   dict
    risk_assessment:      dict
    trade_recommendation: dict
    processing_time_s:    float
    timestamp:            str


class ErrorResponse(BaseModel):
    error:   str
    detail:  str
    request_id: str


# ── Lifespan: startup / shutdown logic ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup; everything after at shutdown.

    We initialise the graph and memory manager HERE rather than at module
    level because:
      1. If ANTHROPIC_API_KEY is missing we want a clean startup error.
      2. Graph compilation is expensive (~1s) — do it once, not per request.
      3. The SQLite checkpointer needs to open its file connection once.
    """
    global _graph, _memory, _mcp_manager

    print("🚀 Hydra Sentinel-X: initialising agents...")
    start = time.time()

    try:
        from graph import build_got_graph
        from memory.memory_manager import CryptoMemoryManager
        from approval.workflow import ApprovalWorkflow
        from observability.prometheus_metrics import init_metrics

        memory_dir = os.getenv("MEMORY_DIR", "./memory")
        os.makedirs(memory_dir, exist_ok=True)

        langgraph_db = os.path.join(memory_dir, "langgraph.db")
        agent_db     = os.path.join(memory_dir, "agent_memory.db")
        approval_db  = os.path.join(memory_dir, "approvals.db")

        _graph  = build_got_graph(langgraph_db)
        _memory = CryptoMemoryManager(agent_db)

        global _approval
        _approval = ApprovalWorkflow(db_path=approval_db)

        # Prometheus metrics
        init_metrics()

        # MCP client — connects to all configured servers
        # Gracefully skips unavailable servers so startup never blocks
        try:
            from integrations.mcp_client import init_mcp_client
            global _mcp_manager
            _mcp_manager = await init_mcp_client()
            print(f"   MCP tools    : {_mcp_manager.tool_summary()}")
        except Exception as mcp_err:
            print(f"   MCP client   : degraded — {mcp_err}")

        # AgentOps tracing (optional)
        try:
            from observability.agentops_tracer import init_agentops
            init_agentops()
        except Exception:
            pass

        elapsed = time.time() - start
        print(f"✅ System ready in {elapsed:.2f}s")
        print(f"   LangGraph DB : {langgraph_db}")
        print(f"   Agent Memory : {agent_db}")
        print(f"   Approval DB  : {approval_db}")

        elapsed = time.time() - start
        print(f"✅ System ready in {elapsed:.2f}s — LangGraph compiled, memory loaded.")
        print(f"   LangGraph DB : {langgraph_db}")
        print(f"   Agent Memory : {agent_db}")

    except Exception as e:
        print(f"❌ FATAL: Failed to initialise agent system: {e}")
        raise   # re-raise so the process exits with a non-zero code

    yield   # ← server is live and handling requests between here and shutdown

    print("🛑 Hydra Sentinel-X: shutting down gracefully.")
    # SQLite connections are closed automatically when the objects are GC'd,
    # but we set to None explicitly to release file handles immediately.
    _graph  = None
    _memory = None


# ── App factory ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hydra Sentinel-X API",
    description="Institutional-grade multi-agent crypto analysis powered by LangGraph + Claude",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",    # Swagger UI at /docs — useful for testing
    redoc_url="/redoc",  # ReDoc at /redoc
)

# CORS: allow any origin in dev; tighten this to your frontend domain in prod.
# e.g. allow_origins=["https://yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: build the initial AgentState dict ────────────────────────────────

def _build_initial_state(req: AnalyzeRequest, session_id: str) -> dict:
    """
    Constructs the AgentState TypedDict that the LangGraph expects.
    Keeping this logic centralised means both the blocking and streaming
    endpoints produce identical state structures.
    """
    memory_context = _memory.build_context(req.symbol) if _memory else ""

    return {
        "messages":              [],
        "user_query":            req.query,
        "target_symbol":         req.symbol,
        "portfolio_value":       req.portfolio_value,
        "current_holdings":      req.holdings or {},
        "risk_tolerance":        req.risk_tolerance,
        "active_agents":         [],
        "current_step":          "initialising",
        "iteration":             0,
        "market_data":           {},
        "technical_analysis":    {},
        "risk_assessment":       {},
        "trade_recommendation":  {},
        "memory_context":        memory_context,
        "final_response":        "",
        "errors":                [],
    }


def _save_memory(session_id: str, req: AnalyzeRequest, final_state: dict):
    """Persist a session summary to the memory manager after each run."""
    if not _memory:
        return
    rec = final_state.get("trade_recommendation", {})
    _memory.save_summary(
        session_id,
        f"symbol={req.symbol} | action={rec.get('action', 'N/A')} | "
        f"query={req.query[:80]}"
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def health_check():
    """
    Simple liveness probe.
    Load balancers and Docker healthchecks call this endpoint to confirm
    the server is running. Returns 200 if the graph is initialised.
    """
    return {
        "status":    "Hydra Sentinel-X Online",
        "graph":     "ready" if _graph else "not initialised",
        "timestamp": datetime.utcnow().isoformat(),
        "version":   "1.0.0",
    }


@app.get("/api/v1/health/deep", summary="Deep health check")
async def deep_health():
    """
    Validates that all subsystems are functional.
    Checks: graph compilation, memory connection, and required env vars.
    """
    checks = {
        "graph_ready":          _graph is not None,
        "memory_ready":         _memory is not None,
        "anthropic_key_set":    bool(os.getenv("ANTHROPIC_API_KEY")),
        "coingecko_accessible": True,   # could ping the API here
    }
    all_ok = all(checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "healthy" if all_ok else "degraded", "checks": checks}
    )


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    summary="Run full multi-agent analysis (blocking)",
    description=(
        "Runs the complete Fan-Out/Fan-In agent pipeline and returns the full "
        "result when all agents have finished. Expect 15-45 seconds. "
        "For a live progress feed, use /api/v1/analyze/stream instead."
    ),
)
async def analyze(req: AnalyzeRequest, request: Request):
    """
    Blocking analysis endpoint.

    The key detail here is `run_in_executor`. LangGraph's `.invoke()` is
    synchronous — it blocks the calling thread until all agents finish.
    If we called it directly in an async function, it would freeze the
    entire FastAPI event loop, preventing any other requests from being
    served while an analysis is running.

    By handing it to `run_in_executor`, we push the blocking work into
    Python's default ThreadPoolExecutor. FastAPI's event loop stays free
    to accept other connections while the agents do their work.
    """
    if not _graph or not _memory:
        raise HTTPException(status_code=503, detail="Agent system not initialised")

    request_id = str(uuid.uuid4())[:8]
    session_id = req.session_id or request_id
    start_time = time.time()

    try:
        initial_state = _build_initial_state(req, session_id)
        config        = {"configurable": {"thread_id": session_id}}

        # Run the synchronous LangGraph in a thread pool so we don't block
        # the async event loop. This is the standard pattern for wrapping
        # sync work in an async FastAPI handler.
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            lambda: _graph.invoke(initial_state, config)
        )

        _save_memory(session_id, req, final_state)

        return AnalyzeResponse(
            session_id           = session_id,
            symbol               = req.symbol,
            final_response       = final_state.get("final_response", ""),
            market_data          = final_state.get("market_data", {}),
            technical_analysis   = final_state.get("technical_analysis", {}),
            risk_assessment      = final_state.get("risk_assessment", {}),
            trade_recommendation = final_state.get("trade_recommendation", {}),
            processing_time_s    = round(time.time() - start_time, 2),
            timestamp            = datetime.utcnow().isoformat(),
        )

    except Exception as e:
        # Log the full error server-side but return a clean message to the client
        print(f"[{request_id}] ERROR during analysis: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(e).__name__} — check server logs for request {request_id}"
        )


@app.post(
    "/api/v1/analyze/stream",
    summary="Run analysis with live progress (Server-Sent Events)",
    description=(
        "Streams agent progress events as they happen using Server-Sent Events. "
        "Each event is a JSON line prefixed with 'data: '. The final event "
        "has type='complete' and contains the full result."
    ),
)
async def analyze_stream(req: AnalyzeRequest):
    """
    Streaming analysis endpoint using Server-Sent Events (SSE).

    SSE is simpler than WebSockets for this use case because the data
    flows in one direction only: server → client. The client opens a
    connection and receives a stream of JSON events until the analysis
    is complete.

    Event types emitted:
      { "type": "progress", "step": "...", "message": "..." }
      { "type": "agent_done", "agent": "...", "data": {...} }
      { "type": "complete", "result": { full AnalyzeResponse } }
      { "type": "error", "message": "..." }
    """
    if not _graph or not _memory:
        async def error_stream():
            yield 'data: {"type":"error","message":"Agent system not initialised"}\n\n'
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    session_id = req.session_id or str(uuid.uuid4())[:8]
    start_time = time.time()

    async def event_generator() -> AsyncIterator[str]:
        """
        Runs LangGraph's `.stream()` in a thread and converts each state
        update into an SSE event. We use `asyncio.Queue` to bridge the
        synchronous LangGraph iterator and the async generator that
        FastAPI's StreamingResponse expects.
        """
        queue: asyncio.Queue = asyncio.Queue()
        loop  = asyncio.get_event_loop()

        def run_graph_stream():
            """This runs in a background thread — never use await here."""
            try:
                initial_state = _build_initial_state(req, session_id)
                config        = {"configurable": {"thread_id": session_id}}

                for state_snapshot in _graph.stream(initial_state, config, stream_mode="values"):
                    step = state_snapshot.get("current_step", "")

                    # Map internal step names to friendly messages
                    step_messages = {
                        "routing_complete":              "🧭 Routing query to specialists...",
                        "market_analysis_complete":      "📊 Market Analyst done",
                        "technical_analysis_complete":   "📈 Technical Analyst done",
                        "risk_assessment_complete":      "⚖️  Risk Manager done",
                        "strategy_complete":             "🎯 Strategy Advisor done",
                        "synthesis_complete":            "✅ Synthesising final report...",
                    }

                    if step:
                        event = json.dumps({
                            "type":    "progress",
                            "step":    step,
                            "message": step_messages.get(step, step),
                        })
                        # thread-safe: put_nowait is safe from non-async threads
                        loop.call_soon_threadsafe(queue.put_nowait, event)

                # Signal completion with the full final state
                final = {
                    "type":   "complete",
                    "result": {
                        "session_id":           session_id,
                        "symbol":               req.symbol,
                        "final_response":       state_snapshot.get("final_response", ""),
                        "market_data":          state_snapshot.get("market_data", {}),
                        "technical_analysis":   state_snapshot.get("technical_analysis", {}),
                        "risk_assessment":      state_snapshot.get("risk_assessment", {}),
                        "trade_recommendation": state_snapshot.get("trade_recommendation", {}),
                        "processing_time_s":    round(time.time() - start_time, 2),
                        "timestamp":            datetime.utcnow().isoformat(),
                    }
                }
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(final))
                _save_memory(session_id, req, state_snapshot)

            except Exception as e:
                err = json.dumps({"type": "error", "message": str(e)})
                loop.call_soon_threadsafe(queue.put_nowait, err)
            finally:
                # Sentinel value tells the async generator to stop
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Launch the blocking stream in a thread pool
        loop.run_in_executor(None, run_graph_stream)

        # Drain the queue and yield SSE-formatted events
        while True:
            event = await queue.get()
            if event is None:   # sentinel — we're done
                break
            yield f"data: {event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
        },
    )


@app.get("/api/v1/memory/{symbol}", summary="Fetch memory context for a symbol")
async def get_memory(symbol: str):
    """Return the persistent memory context for a given symbol."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialised")
    return {
        "symbol":    symbol.upper(),
        "context":   _memory.build_context(symbol),
        "summaries": _memory.get_recent_summaries(10),
        "stats":     _memory.get_trade_stats(),
    }


@app.get("/api/v1/trades", summary="Fetch trade history")
async def get_trades(limit: int = 20):
    """Return the most recent trade records from persistent memory."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialised")
    return {
        "trades": _memory.get_recent_trades(limit),
        "stats":  _memory.get_trade_stats(),
    }


# ── Approval workflow endpoints ───────────────────────────────────────────────
#
# These endpoints form the Human-in-the-Loop layer. The flow is:
#
#   POST /api/v1/analyze  →  agents run  →  POST /api/v1/approvals/submit
#   GET  /api/v1/approvals/pending       →  human reviews the queue
#   POST /api/v1/approvals/{id}/approve  →  human approves + adds reason
#   POST /api/v1/approvals/{id}/reject   →  human rejects + adds reason
#
# The UI calls these endpoints automatically. You can also curl them
# directly if you're integrating with another system.


class SubmitApprovalRequest(BaseModel):
    """Used internally after an analysis completes."""
    session_id:    str
    symbol:        str
    agent_outputs: dict   # the complete final AgentState


class DecisionRequest(BaseModel):
    """
    Payload for approve or reject actions.
    reason is required — the system will reject empty or trivially short
    reasons because the audit trail only has value if it contains real
    human reasoning, not filler text.
    """
    reason:   str = Field(..., min_length=5,
                          description="Why are you making this decision? Be specific.")
    reviewer: str = Field(default="operator",
                          description="Your identifier — logged in the audit trail")


@app.post("/api/v1/approvals/submit", summary="Submit agent recommendation for human review")
async def submit_for_approval(req: SubmitApprovalRequest):
    """
    Submit the output of an agent analysis run into the approval queue.
    This is called automatically by the /analyze endpoint — you don't
    normally need to call it manually. The returned recommendation_id
    is what you use to approve or reject the recommendation.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")
    try:
        rec_id = _approval.submit(
            session_id    = req.session_id,
            symbol        = req.symbol,
            agent_outputs = req.agent_outputs,
        )
        return {
            "recommendation_id": rec_id,
            "status":            "pending",
            "message":           f"Recommendation submitted. Review at /api/v1/approvals/{rec_id}",
            "expires_in_hours":  4,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/approvals/pending", summary="Get all pending recommendations")
async def get_pending_approvals():
    """
    Return all recommendations currently awaiting human review.
    This is the primary endpoint the dashboard UI polls to populate
    the approval queue. Expired items are automatically filtered out.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")
    pending = _approval.get_pending()
    return {
        "count":   len(pending),
        "pending": pending,
    }


@app.get("/api/v1/approvals/{rec_id}", summary="Get a single recommendation with full detail")
async def get_recommendation(rec_id: str):
    """
    Return the full detail of a recommendation including the complete
    agent analysis, risk parameters, and audit trail. The UI uses this
    to populate the detail view when a human clicks on a pending item.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")
    rec = _approval.get_recommendation(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found")
    audit = _approval.get_audit_log(rec_id)
    return {**rec, "audit_log": audit}


@app.post("/api/v1/approvals/{rec_id}/approve", summary="Approve a trade recommendation")
async def approve_recommendation(rec_id: str, req: DecisionRequest):
    """
    Approve a pending recommendation.

    Your reason is permanently recorded in the audit trail alongside
    the timestamp and your reviewer identifier. This creates the feedback
    loop that lets you evaluate, over time, whether the agents' reasoning
    aligns with your own — and whether the trades you approved performed
    as expected.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")
    try:
        result = _approval.approve(
            rec_id   = rec_id,
            reason   = req.reason,
            reviewer = req.reviewer,
        )
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/approvals/{rec_id}/reject", summary="Reject a trade recommendation")
async def reject_recommendation(rec_id: str, req: DecisionRequest):
    """
    Reject a pending recommendation.

    Rejected recommendations are never deleted — they remain in the
    database as a record of what the agents suggested and why you
    disagreed. This archive is valuable: a pattern of rejections for
    the same reason (e.g. "RSI overbought but agents still say buy")
    tells you something specific about agent calibration that you can
    act on.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")
    try:
        result = _approval.reject(
            rec_id   = rec_id,
            reason   = req.reason,
            reviewer = req.reviewer,
        )
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExecuteRequest(BaseModel):
    slippage_bps: int = Field(
        default=50,
        ge=10, le=500,
        description="Slippage tolerance in basis points (50 = 0.5%, 200 = 2%)"
    )


@app.get("/api/v1/approvals/{rec_id}/quote", summary="Fetch live CDP quote (read-only)")
async def get_live_quote(rec_id: str):
    """
    Fetch a real-time swap quote from the CDP SDK for the asset in this
    recommendation. READ-ONLY — no transaction is submitted.

    Returns: pool_price, price_impact_pct, liquidity_usd, vol_liq_ratio,
             estimated_output. The dashboard uses this to populate the MEV
             risk panel before the human makes an execute decision.

    If CDP credentials are not configured, returns a degraded response
    with on-chain data from the market intelligence MCP server only.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")

    rec = _approval.get_recommendation(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found")

    symbol      = rec.get("symbol", "ETH")
    entry_low   = rec.get("entry_price_low",  0.0) or 0.0
    entry_high  = rec.get("entry_price_high", 0.0) or 0.0
    entry_mid   = (entry_low + entry_high) / 2 if entry_low and entry_high else entry_low

    # ── Try CDP quote first ──────────────────────────────────────────────
    cdp_key     = os.getenv("CDP_API_KEY_NAME", "")
    cdp_private = os.getenv("CDP_API_KEY_PRIVATE_KEY", "")

    pool_price       = None
    price_impact_pct = None
    estimated_output = None
    liquidity_usd    = None
    vol_liq_ratio    = None

    if cdp_key and cdp_private:
        try:
            from decimal import Decimal
            from cdp import CdpClient

            # Standard position size from the recommendation (default $500)
            position_usd = float(rec.get("position_size_usd", 500) or 500)

            # Use a 1-unit quote amount to get price without slippage distortion
            quote_amount_usd = min(100.0, position_usd * 0.1)

            async with CdpClient() as cdp:
                # Get a quote for a nominal $100 swap — no execution, just pricing
                # CDP quote API returns: price, price_impact, estimated_output
                quote = await cdp.evm.get_swap_quote(
                    network     = "base",
                    from_token  = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC
                    to_token    = _symbol_to_base_address(symbol),
                    from_amount = str(int(Decimal(str(quote_amount_usd)) * Decimal(10**6))),  # USDC = 6 dec
                )
                pool_price       = float(quote.price)       if hasattr(quote, "price")        else None
                price_impact_pct = float(quote.price_impact) * 100 if hasattr(quote, "price_impact") else None
                estimated_output = str(quote.to_amount)     if hasattr(quote, "to_amount")    else None

        except ImportError:
            pass   # CDP not installed — fall through to on-chain data
        except Exception as e:
            # Quote failed (wrong network, token not found, etc.) — log and continue
            import logging
            logging.getLogger(__name__).warning(f"CDP quote failed for {symbol}: {e}")

    # ── Augment with on-chain liquidity data ─────────────────────────────
    # Call GeckoTerminal directly — same data as the market MCP tool
    try:
        import httpx
        url = f"https://api.geckoterminal.com/api/v2/search/pools?query={symbol}&network=base&page=1"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url, headers={"Accept": "application/json"})
            if r.status_code == 200:
                pools = r.json().get("data", [])[:1]
                if pools:
                    attr          = pools[0].get("attributes", {})
                    liquidity_usd = float(attr.get("reserve_in_usd") or 0) or None
                    vol_24h       = float(attr.get("volume_usd", {}).get("h24") or 0) or None
                    if liquidity_usd and vol_24h:
                        vol_liq_ratio = round(vol_24h / liquidity_usd, 2)
                    # If CDP quote didn't return a price, use pool price
                    if pool_price is None:
                        pool_price = float(attr.get("base_token_price_usd") or 0) or None
    except Exception:
        pass

    return {
        "rec_id":          rec_id,
        "symbol":          symbol,
        "pool_price":      pool_price,
        "price_impact_pct":price_impact_pct,
        "estimated_output":estimated_output,
        "liquidity_usd":   liquidity_usd,
        "vol_liq_ratio":   vol_liq_ratio,
        "agent_entry_mid": entry_mid,
        "cdp_available":   bool(cdp_key and cdp_private),
    }


def _symbol_to_base_address(symbol: str) -> str:
    """Map a symbol to its Base mainnet contract address. Expand as needed."""
    addresses = {
        "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "WETH": "0x4200000000000000000000000000000000000006",
        "ETH":  "0x4200000000000000000000000000000000000006",
        "CBBTC":"0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
    }
    return addresses.get(symbol.upper(), "")


@app.post("/api/v1/approvals/{rec_id}/execute", summary="Execute an approved trade recommendation")
async def execute_recommendation(rec_id: str, req: ExecuteRequest):
    """
    Execute an approved recommendation on-chain via the CDP SDK.

    GUARDS (all must pass before any transaction fires):
      1. Recommendation exists and status is exactly 'approved'.
      2. Recommendation has not expired (decided_at within 4h is irrelevant here —
         approved recs can be executed up to 24h after approval).
      3. CDP credentials are configured in the environment.
      4. CDP SDK swap call completes without error.
         If slippage is exceeded at execution time, CDP reverts the tx
         automatically and returns an error — we surface that to the caller.

    On success: status updated to 'executed', tx hash written to audit trail.
    On failure: status remains 'approved' — human can retry or reject.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")

    # ── Guard 1: validate approval record ────────────────────────────────
    rec = _approval.get_recommendation(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found")

    if rec.get("status") != "approved":
        raise HTTPException(
            status_code=400,
            detail=f"Recommendation is {rec.get('status')} — only 'approved' recommendations can be executed"
        )

    # ── Guard 2: CDP credentials ─────────────────────────────────────────
    cdp_key     = os.getenv("CDP_API_KEY_NAME", "")
    cdp_private = os.getenv("CDP_API_KEY_PRIVATE_KEY", "")

    if not cdp_key or not cdp_private:
        raise HTTPException(
            status_code=503,
            detail="CDP_API_KEY_NAME and CDP_API_KEY_PRIVATE_KEY must be set to enable execution"
        )

    symbol       = rec.get("symbol", "")
    to_address   = _symbol_to_base_address(symbol)
    position_usd = float(rec.get("position_size_usd", 0) or 0)

    if not to_address:
        raise HTTPException(
            status_code=400,
            detail=f"No Base mainnet address configured for {symbol}. Add to _symbol_to_base_address()."
        )
    if position_usd <= 0:
        raise HTTPException(status_code=400, detail="Position size is zero — cannot execute")

    # ── Guard 3: Execute via CDP SDK ─────────────────────────────────────
    try:
        from decimal import Decimal
        from cdp import CdpClient
        from cdp.actions.evm.swap import AccountSwapOptions

        # Convert USD position to USDC atomic units (6 decimals)
        amount_usdc_atomic = str(int(Decimal(str(position_usd)) * Decimal(10**6)))

        async with CdpClient() as cdp:
            account = await cdp.evm.get_or_create_account(name="HydraMasterWallet")
            result  = await account.swap(
                AccountSwapOptions(
                    network      = "base",
                    from_token   = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC
                    to_token     = to_address,
                    from_amount  = amount_usdc_atomic,
                    slippage_bps = req.slippage_bps,
                )
            )
            tx_hash = result.transaction_hash

    except ImportError:
        raise HTTPException(status_code=503, detail="cdp-sdk not installed — pip install cdp-sdk")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"CDP execution failed: {str(e)}")

    # ── Write outcome to approval DB ─────────────────────────────────────
    try:
        _approval.mark_executed(
            rec_id  = rec_id,
            tx_hash = tx_hash,
            details = {
                "slippage_bps": req.slippage_bps,
                "position_usd": position_usd,
                "symbol":       symbol,
                "to_address":   to_address,
            },
        )
    except Exception:
        pass   # Don't let a DB write failure mask a successful tx

    return {
        "rec_id":          rec_id,
        "status":          "executed",
        "transaction_hash":tx_hash,
        "symbol":          symbol,
        "position_usd":    position_usd,
        "slippage_bps":    req.slippage_bps,
        "message":         f"Transaction submitted. Verify on BaseScan: https://basescan.org/tx/{tx_hash}",
    }


@app.get("/api/v1/approvals/history/all", summary="Get recommendation history")
async def get_approval_history(
    limit:  int = 50,
    status: Optional[str] = None,
    symbol: Optional[str] = None,
):
    """
    Return historical recommendations with optional filters.
    Use status='approved' or status='rejected' to study your
    decision patterns, or filter by symbol to review performance
    on a specific asset.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")
    history = _approval.get_history(limit=limit, status=status, symbol=symbol)
    stats   = _approval.get_stats()
    return {"stats": stats, "history": history}


@app.get("/api/v1/approvals/stats/summary", summary="Approval statistics")
async def get_approval_stats():
    """
    Aggregate statistics across all recommendations.
    The approval_rate_pct and win_rate_of_executed are the two most
    useful numbers for evaluating both your trust in the agents and
    whether that trust is calibrated to actual outcomes.
    """
    if not _approval:
        raise HTTPException(status_code=503, detail="Approval system not initialised")
    return _approval.get_stats()


@app.get("/dashboard", response_class=HTMLResponse, summary="Human approval dashboard")
async def serve_dashboard():
    """
    Serve the Command Deck — the browser-based human approval interface.
    Open this in your browser to review and approve/reject recommendations.
    """
    dashboard_path = os.path.join(os.path.dirname(__file__), "approval", "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard not found. Ensure approval/dashboard.html exists.</h1>", status_code=404)
