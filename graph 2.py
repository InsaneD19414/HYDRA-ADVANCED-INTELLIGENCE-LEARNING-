"""
graph.py — Hydra Sentinel-X agent graph with Graph of Thoughts layer.

ARCHITECTURE (updated):

    START
      │
      ▼
  orchestrator_parse
      │
      ├──► market_analyst      ┐
      ├──► technical_analyst   │  Phase 1: parallel specialist fan-out
      ├──► risk_manager        │  (unchanged from original architecture)
      └──► strategy_advisor    ┘
                │
                ▼
      got_init   ← initialises GoT fields, sets retry budget
          │
          ▼
      generate_hypotheses  ◄──────────────────────────────┐
          │                                                │
          ▼                                                │
      adversarial_critique                                 │
          │                                                │
     ┌────┴──────────────────┐                             │
     │ survivors?            │ no survivors + retries left │
     │                       │                             │
     ▼                       ▼                             │
  aggregate_and_improve   increment_generation ────────────┘
          │
          ▼
  orchestrator_synthesise
          │
         END

TWO GRAPH VARIANTS:
  build_graph()      — original flat pipeline, fully backwards compatible.
  build_got_graph()  — GoT-enhanced pipeline for production.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from state import AgentState, GoTState
from agents.orchestrator import orchestrator_parse_node, orchestrator_synthesise_node
from agents.specialists import (
    market_analyst_node,
    technical_analyst_node,
    risk_manager_node,
    strategy_advisor_node,
)
from agents.got_nodes import (
    generate_hypotheses_node,
    adversarial_critique_node,
    aggregate_and_improve_node,
    route_after_critique,
    increment_generation,
)


# ── Shared routing ───────────────────────────────────────────────────────────

def route_to_specialists(state: AgentState) -> list[str]:
    active = state.get("active_agents", [
        "market_analyst", "technical_analyst", "risk_manager", "strategy_advisor"
    ])
    valid = {"market_analyst", "technical_analyst", "risk_manager", "strategy_advisor"}
    return [a for a in active if a in valid] or list(valid)


# ── GoT initialisation node ──────────────────────────────────────────────────

def got_init_node(state: GoTState) -> dict:
    """
    Initialise GoT fields and set the retry budget based on risk_tolerance.
    Conservative = 1 pass (fail fast → HOLD).
    Moderate     = 2 passes.
    Aggressive   = 3 passes (full retry budget).
    """
    risk = state.get("risk_tolerance", "moderate")
    max_gen = {"conservative": 1, "moderate": 2, "aggressive": 3}.get(risk, 2)
    return {
        "thought_graph":       [],
        "adversarial_attacks": [],
        "got_generation":      1,
        "got_max_generations": max_gen,
        "consensus_reached":   False,
        "surviving_thoughts":  [],
        "current_step":        "got_initialised",
    }


# ── Sync wrappers for async GoT nodes ────────────────────────────────────────
# LangGraph's synchronous runner needs synchronous callables.
# These wrappers handle event-loop compatibility across environments.

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def generate_hypotheses_sync(state: GoTState) -> dict:
    return _run_async(generate_hypotheses_node(state))

def adversarial_critique_sync(state: GoTState) -> dict:
    return _run_async(adversarial_critique_node(state))

def aggregate_and_improve_sync(state: GoTState) -> dict:
    return _run_async(aggregate_and_improve_node(state))


# ── Original graph (backwards compatible) ────────────────────────────────────

def build_graph(memory_db_path: str = "./memory/langgraph.db"):
    """Original flat pipeline — no GoT layer. Unchanged from v1."""
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator_parse",      orchestrator_parse_node)
    builder.add_node("market_analyst",          market_analyst_node)
    builder.add_node("technical_analyst",       technical_analyst_node)
    builder.add_node("risk_manager",            risk_manager_node)
    builder.add_node("strategy_advisor",        strategy_advisor_node)
    builder.add_node("orchestrator_synthesise", orchestrator_synthesise_node)

    builder.add_edge(START, "orchestrator_parse")
    builder.add_conditional_edges(
        "orchestrator_parse", route_to_specialists,
        {s: s for s in ["market_analyst","technical_analyst","risk_manager","strategy_advisor"]},
    )
    for s in ["market_analyst","technical_analyst","risk_manager","strategy_advisor"]:
        builder.add_edge(s, "orchestrator_synthesise")
    builder.add_edge("orchestrator_synthesise", END)

    Path(memory_db_path).parent.mkdir(parents=True, exist_ok=True)
    return builder.compile(checkpointer=SqliteSaver.from_conn_string(memory_db_path))


# ── GoT-enhanced graph ────────────────────────────────────────────────────────

def build_got_graph(memory_db_path: str = "./memory/langgraph.db"):
    """
    Full Graph of Thoughts pipeline.

    Phase 1 (specialists) runs identically to the original graph.
    Phase 2 (GoT) reads their structured output, generates competing
    hypotheses, stress-tests them adversarially, loops if needed, and
    writes a battle-tested recommendation back to trade_recommendation.
    The orchestrator_synthesise node then builds the final user-facing
    response from that enriched state — including the audit trail of
    which attacks the surviving thesis withstood.
    """
    builder = StateGraph(GoTState)

    # Phase 1 — specialist agents
    builder.add_node("orchestrator_parse",      orchestrator_parse_node)
    builder.add_node("market_analyst",          market_analyst_node)
    builder.add_node("technical_analyst",       technical_analyst_node)
    builder.add_node("risk_manager",            risk_manager_node)
    builder.add_node("strategy_advisor",        strategy_advisor_node)

    # Phase 2 — GoT reasoning
    builder.add_node("got_init",                got_init_node)
    builder.add_node("generate_hypotheses",     generate_hypotheses_sync)
    builder.add_node("adversarial_critique",    adversarial_critique_sync)
    builder.add_node("increment_generation",    increment_generation)
    builder.add_node("aggregate_and_improve",   aggregate_and_improve_sync)

    # Synthesis
    builder.add_node("orchestrator_synthesise", orchestrator_synthesise_node)

    # Phase 1 edges
    builder.add_edge(START, "orchestrator_parse")
    builder.add_conditional_edges(
        "orchestrator_parse", route_to_specialists,
        {s: s for s in ["market_analyst","technical_analyst","risk_manager","strategy_advisor"]},
    )
    # All four specialists converge at got_init
    for s in ["market_analyst","technical_analyst","risk_manager","strategy_advisor"]:
        builder.add_edge(s, "got_init")

    # Phase 2 edges
    builder.add_edge("got_init",            "generate_hypotheses")
    builder.add_edge("generate_hypotheses", "adversarial_critique")

    # The GoT loop: survivors → aggregate; all pruned → increment → generate
    builder.add_conditional_edges(
        "adversarial_critique",
        route_after_critique,
        {
            "aggregate": "aggregate_and_improve",
            "generate":  "increment_generation",
        },
    )
    builder.add_edge("increment_generation",  "generate_hypotheses")
    builder.add_edge("aggregate_and_improve", "orchestrator_synthesise")
    builder.add_edge("orchestrator_synthesise", END)

    Path(memory_db_path).parent.mkdir(parents=True, exist_ok=True)
    return builder.compile(checkpointer=SqliteSaver.from_conn_string(memory_db_path))
