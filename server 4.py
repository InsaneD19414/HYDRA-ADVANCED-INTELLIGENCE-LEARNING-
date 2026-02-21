"""
mcp_servers/memory_server/server.py
────────────────────────────────────
Semantic Memory MCP server backed by Qdrant vector database.
Deploy on Alpic alongside the market intelligence server.

This server replaces the flat SQLite trade summaries with a vector
memory layer that supports semantic similarity search. When SAMUEL
asks "show me situations similar to today's ETH setup," he gets the
most semantically relevant past analyses — not just the most recent.

TOOLS EXPOSED:
  store_analysis      — embed and store a completed analysis in Qdrant
  search_similar      — semantic search: find past analyses like this one
  get_symbol_history  — retrieve all stored analyses for a specific symbol
  store_rejection     — record human rejection with reasoning (feedback loop)
  get_rejections      — retrieve rejection patterns for a symbol
  store_thought_node  — persist a GoT ThoughtNode with its adversarial record
  search_patterns     — search for past attack patterns by type (flash_crash, etc.)

EMBEDDING:
  Uses sentence-transformers/all-MiniLM-L6-v2 via the fastembed library.
  Runs entirely locally inside the server — no OpenAI/external embedding calls.
  384-dimensional vectors; fast, lightweight, good semantic quality for
  financial text.

COLLECTIONS:
  hydra_analyses      — one document per completed full analysis cycle
  hydra_rejections    — one document per human rejection with reasoning
  hydra_thought_nodes — one document per GoT ThoughtNode (for adversarial audit)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

# ── Optional deps — fail gracefully if not installed ────────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from fastembed import TextEmbedding
    EMBED_AVAILABLE = True
except ImportError:
    EMBED_AVAILABLE = False

# ── Server bootstrap ─────────────────────────────────────────────────────────
mcp = FastMCP(
    "Hydra Memory",
    stateless_http=True,
    instructions=(
        "Semantic memory layer for Hydra Sentinel-X. Store and retrieve "
        "past analyses, rejection patterns, and GoT reasoning chains. "
        "Uses vector similarity search for contextually relevant recall."
    ),
)

QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY    = os.getenv("QDRANT_API_KEY", "")
HYDRA_API_KEY = os.getenv("HYDRA_API_KEY", "")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM    = 384

COLLECTION_ANALYSES    = "hydra_analyses"
COLLECTION_REJECTIONS  = "hydra_rejections"
COLLECTION_THOUGHTS    = "hydra_thought_nodes"


# ── Client / embedder singletons ─────────────────────────────────────────────

_qdrant:  QdrantClient   | None = None
_embedder: TextEmbedding  | None = None

def _get_qdrant() -> "QdrantClient":
    global _qdrant
    if _qdrant is None:
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client not installed. Run: pip install qdrant-client")
        _qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_KEY or None,
        )
        _ensure_collections(_qdrant)
    return _qdrant

def _get_embedder() -> "TextEmbedding":
    global _embedder
    if _embedder is None:
        if not EMBED_AVAILABLE:
            raise RuntimeError("fastembed not installed. Run: pip install fastembed")
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder

def _embed(text: str) -> list[float]:
    embedder = _get_embedder()
    vectors  = list(embedder.embed([text]))
    return vectors[0].tolist()

def _ensure_collections(client: "QdrantClient") -> None:
    existing = {c.name for c in client.get_collections().collections}
    for name in [COLLECTION_ANALYSES, COLLECTION_REJECTIONS, COLLECTION_THOUGHTS]:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

def _auth(ctx: Context) -> bool:
    if not HYDRA_API_KEY:
        return True
    return ctx.request_context.request.headers.get("x-api-key", "") == HYDRA_API_KEY


# ── Tool 1: Store Analysis ────────────────────────────────────────────────────

@mcp.tool(
    title="Store Analysis",
    description=(
        "Embed and persist a completed analysis cycle in the Qdrant vector store. "
        "Call this after every full analysis so future sessions can recall similar setups. "
        "Returns the stored document ID."
    ),
)
async def store_analysis(
    ctx: Context,
    symbol:             str   = Field(description="Asset symbol, e.g. BTC"),
    action:             str   = Field(description="Recommended action, e.g. buy, sell, hold"),
    rationale:          str   = Field(description="Full rationale string from trade_recommendation"),
    confidence:         float = Field(description="Final confidence score 0.0–1.0"),
    market_conditions:  str   = Field(description="Brief description of market conditions at time of analysis"),
    got_generations:    int   = Field(default=1, description="Number of GoT generations used"),
    attacks_survived:   int   = Field(default=0, description="Number of adversarial attacks survived"),
    session_id:         str   = Field(default="", description="Session ID for cross-reference with approval DB"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    client = _get_qdrant()
    doc_id = str(uuid.uuid4())
    ts     = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build the text we embed — designed for semantic recall
    text_to_embed = (
        f"Symbol: {symbol}. Action: {action}. "
        f"Market conditions: {market_conditions}. "
        f"Rationale: {rationale}. "
        f"Confidence: {confidence:.0%}. "
        f"GoT generations: {got_generations}. "
        f"Adversarial attacks survived: {attacks_survived}."
    )
    vector = _embed(text_to_embed)

    payload = {
        "doc_id":           doc_id,
        "symbol":           symbol.upper(),
        "action":           action,
        "rationale":        rationale,
        "confidence":       confidence,
        "market_conditions":market_conditions,
        "got_generations":  got_generations,
        "attacks_survived": attacks_survived,
        "session_id":       session_id,
        "stored_at":        ts,
        "text_embedded":    text_to_embed,
    }

    client.upsert(
        collection_name=COLLECTION_ANALYSES,
        points=[PointStruct(id=doc_id, vector=vector, payload=payload)],
    )
    return json.dumps({"doc_id": doc_id, "stored_at": ts, "status": "ok"})


# ── Tool 2: Semantic Search ───────────────────────────────────────────────────

@mcp.tool(
    title="Search Similar Analyses",
    description=(
        "Semantic search: given a description of current market conditions, "
        "find the most similar past analyses stored in memory. "
        "Returns top-k results ranked by vector similarity."
    ),
)
async def search_similar(
    ctx: Context,
    query: str  = Field(description="Description of current conditions, e.g. 'BTC RSI 72 declining volume bearish divergence'"),
    limit: int  = Field(default=5, description="Number of similar analyses to return"),
    symbol: str = Field(default="", description="Optional: filter results to a specific symbol"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    client = _get_qdrant()
    vector = _embed(query)

    query_filter = None
    if symbol:
        query_filter = Filter(
            must=[FieldCondition(key="symbol", match=MatchValue(value=symbol.upper()))]
        )

    results = client.search(
        collection_name=COLLECTION_ANALYSES,
        query_vector=vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    hits = [
        {
            "score":            round(r.score, 4),
            "symbol":           r.payload.get("symbol"),
            "action":           r.payload.get("action"),
            "confidence":       r.payload.get("confidence"),
            "market_conditions":r.payload.get("market_conditions"),
            "rationale":        r.payload.get("rationale", "")[:300],
            "attacks_survived": r.payload.get("attacks_survived"),
            "got_generations":  r.payload.get("got_generations"),
            "stored_at":        r.payload.get("stored_at"),
        }
        for r in results
    ]
    return json.dumps({
        "query":   query,
        "symbol":  symbol or "all",
        "results": hits,
    })


# ── Tool 3: Symbol History ────────────────────────────────────────────────────

@mcp.tool(
    title="Get Symbol History",
    description="Retrieve all stored analyses for a specific symbol, sorted newest first.",
)
async def get_symbol_history(
    ctx: Context,
    symbol: str = Field(description="Asset symbol, e.g. ETH"),
    limit:  int = Field(default=10, description="Maximum number of records to return"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    client  = _get_qdrant()
    results = client.scroll(
        collection_name=COLLECTION_ANALYSES,
        scroll_filter=Filter(
            must=[FieldCondition(key="symbol", match=MatchValue(value=symbol.upper()))]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    points = sorted(
        results[0],
        key=lambda p: p.payload.get("stored_at", ""),
        reverse=True,
    )
    history = [
        {
            "action":          p.payload.get("action"),
            "confidence":      p.payload.get("confidence"),
            "rationale":       p.payload.get("rationale", "")[:200],
            "attacks_survived":p.payload.get("attacks_survived"),
            "stored_at":       p.payload.get("stored_at"),
        }
        for p in points
    ]
    return json.dumps({"symbol": symbol.upper(), "history": history})


# ── Tool 4: Store Rejection (Feedback Loop) ───────────────────────────────────

@mcp.tool(
    title="Store Rejection",
    description=(
        "Record a human rejection of an agent recommendation, with the reviewer's "
        "reasoning. This feeds the pattern analyser — over time the agents learn "
        "which types of recommendations you consistently reject and why."
    ),
)
async def store_rejection(
    ctx: Context,
    symbol:     str   = Field(description="Asset symbol"),
    action:     str   = Field(description="The rejected action"),
    reason:     str   = Field(description="Reviewer's rejection reason (from approval dashboard)"),
    confidence: float = Field(description="Agent's confidence at time of rejection"),
    rec_id:     str   = Field(default="", description="Approval DB recommendation ID"),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    client = _get_qdrant()
    doc_id = str(uuid.uuid4())
    ts     = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    text_to_embed = (
        f"REJECTED: {symbol} {action}. "
        f"Reviewer reason: {reason}. "
        f"Agent confidence was {confidence:.0%}."
    )
    vector  = _embed(text_to_embed)
    payload = {
        "doc_id":    doc_id,
        "symbol":    symbol.upper(),
        "action":    action,
        "reason":    reason,
        "confidence":confidence,
        "rec_id":    rec_id,
        "stored_at": ts,
    }
    client.upsert(
        collection_name=COLLECTION_REJECTIONS,
        points=[PointStruct(id=doc_id, vector=vector, payload=payload)],
    )
    return json.dumps({"doc_id": doc_id, "stored_at": ts, "status": "ok"})


# ── Tool 5: Get Rejection Patterns ────────────────────────────────────────────

@mcp.tool(
    title="Get Rejection Patterns",
    description=(
        "Retrieve rejection history for a symbol or across all symbols. "
        "The orchestrator should call this before finalising recommendations "
        "so agents can self-calibrate based on known human disagreement patterns."
    ),
)
async def get_rejections(
    ctx: Context,
    symbol: str = Field(default="", description="Filter by symbol, or empty for all"),
    limit:  int = Field(default=10),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    client      = _get_qdrant()
    filt        = None
    if symbol:
        filt = Filter(must=[FieldCondition(key="symbol", match=MatchValue(value=symbol.upper()))])

    results, _ = client.scroll(
        collection_name=COLLECTION_REJECTIONS,
        scroll_filter=filt,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    rejections = [
        {
            "symbol":    p.payload.get("symbol"),
            "action":    p.payload.get("action"),
            "reason":    p.payload.get("reason"),
            "confidence":p.payload.get("confidence"),
            "stored_at": p.payload.get("stored_at"),
        }
        for p in sorted(results, key=lambda p: p.payload.get("stored_at",""), reverse=True)
    ]
    return json.dumps({"symbol": symbol or "all", "rejections": rejections})


# ── Tool 6: Store ThoughtNode ─────────────────────────────────────────────────

@mcp.tool(
    title="Store GoT Thought Node",
    description=(
        "Persist a GoT ThoughtNode with its adversarial critique record. "
        "Enables future searches like 'find past flash-crash attack scenarios on SOL longs'."
    ),
)
async def store_thought_node(
    ctx: Context,
    symbol:           str   = Field(description="Asset symbol"),
    agent_origin:     str   = Field(description="STINKMEANER or SAMUEL"),
    thesis:           str   = Field(description="The core hypothesis thesis"),
    direction:        str   = Field(description="long, short, or neutral"),
    survived_critique:bool  = Field(description="Did this thought survive adversarial critique?"),
    confidence_final: float = Field(description="Post-attack confidence score"),
    attack_types:     str   = Field(default="", description="Comma-separated attack types applied"),
    session_id:       str   = Field(default=""),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    client = _get_qdrant()
    doc_id = str(uuid.uuid4())
    ts     = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    text = (
        f"{agent_origin} hypothesis on {symbol} ({direction}): {thesis}. "
        f"Attacks applied: {attack_types}. "
        f"Survived: {survived_critique}. Final confidence: {confidence_final:.0%}."
    )
    vector  = _embed(text)
    payload = {
        "doc_id":           doc_id, "symbol": symbol.upper(),
        "agent_origin":     agent_origin, "thesis": thesis,
        "direction":        direction,
        "survived_critique":survived_critique,
        "confidence_final": confidence_final,
        "attack_types":     attack_types,
        "session_id":       session_id,
        "stored_at":        ts,
    }
    client.upsert(
        collection_name=COLLECTION_THOUGHTS,
        points=[PointStruct(id=doc_id, vector=vector, payload=payload)],
    )
    return json.dumps({"doc_id": doc_id, "stored_at": ts})


# ── Tool 7: Search Attack Patterns ───────────────────────────────────────────

@mcp.tool(
    title="Search Attack Patterns",
    description=(
        "Search past GoT ThoughtNodes by attack type. "
        "E.g. 'find all flash_crash attacks on BTC long theses and whether they survived.' "
        "Helps GRANDDAD calibrate stress test intensity based on historical outcomes."
    ),
)
async def search_patterns(
    ctx: Context,
    query:       str = Field(description="Natural language query, e.g. 'flash crash attacks on SOL longs'"),
    attack_type: str = Field(default="", description="Filter by attack type: flash_crash, volume_spoof, macro_shock, liquidity_drain"),
    limit:       int = Field(default=8),
) -> str:
    if not _auth(ctx):
        raise ValueError("Invalid API key")

    client = _get_qdrant()
    vector = _embed(query)

    filt = None
    if attack_type:
        filt = Filter(must=[FieldCondition(key="attack_types", match=MatchValue(value=attack_type))])

    results = client.search(
        collection_name=COLLECTION_THOUGHTS,
        query_vector=vector,
        limit=limit,
        query_filter=filt,
        with_payload=True,
    )
    hits = [
        {
            "score":           round(r.score, 4),
            "symbol":          r.payload.get("symbol"),
            "agent_origin":    r.payload.get("agent_origin"),
            "direction":       r.payload.get("direction"),
            "thesis":          r.payload.get("thesis", "")[:200],
            "attack_types":    r.payload.get("attack_types"),
            "survived":        r.payload.get("survived_critique"),
            "confidence_final":r.payload.get("confidence_final"),
            "stored_at":       r.payload.get("stored_at"),
        }
        for r in results
    ]
    return json.dumps({"query": query, "attack_type": attack_type or "all", "results": hits})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8002)
