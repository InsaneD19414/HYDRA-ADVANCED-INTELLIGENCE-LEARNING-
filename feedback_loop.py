"""
integrations/feedback_loop.py
──────────────────────────────
Rejection Pattern Analyser — closes the feedback loop between human
decisions in the approval dashboard and the agents' future reasoning.

THE PROBLEM IT SOLVES:
  Every time a human rejects a recommendation, they write a reason:
  "RSI too high, macro too risky", "volume looks spoofed", "I don't
  trust this breakout pattern on low volume." That reasoning sits in
  approvals.db and never influences future agent behaviour. The agents
  have no memory of your disagreements.

  This module changes that. It reads the rejection history (from both
  the SQLite approval DB and the Qdrant memory server), extracts
  patterns, and produces a "calibration context" string that the
  orchestrator prepends to every analysis session.

HOW IT WORKS:
  1. fetch_rejection_patterns()  — pulls recent rejections from both
     sources and groups them by symbol, action, and recurring themes.
  2. analyse_patterns()           — uses an LLM pass to distil the raw
     rejection list into specific, actionable calibration signals.
  3. build_calibration_context()  — formats the signals as a concise
     context block the orchestrator injects into agent system prompts.
  4. run_feedback_loop()          — orchestrates all three steps and
     returns the final context string. Called once per session start.

WHAT AGENTS RECEIVE:
  A short block like:

    HUMAN CALIBRATION SIGNALS (from approval history):
    • BTC BUY: Consistently rejected when RSI > 72 (5 rejections).
      Human reasoning pattern: "macro too risky despite technicals."
      → Increase caution for BUY signals when RSI > 70 in current conditions.
    • ETH LONG: 2 rejections citing "volume looks spoofed."
      → GRANDDAD should increase weight on VOLUME_SPOOF attack for ETH.
    • General: Human approved 3/3 HOLD recommendations in past 2 weeks.
      → HOLD is currently the preferred action; raise conviction bar for BUY.

  This is injected into the orchestrator's system prompt as additional
  context, not as hard constraints. The agents reason over it — they
  can disagree with it if the current data is sufficiently strong.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ── SQLite approval DB schema (mirrors approval/workflow.py) ─────────────────
_APPROVAL_SCHEMA_FIELDS = [
    "id", "session_id", "symbol", "action", "confidence",
    "rationale", "key_risks", "status", "reviewer_reason",
    "submitted_at", "decided_at",
]


class FeedbackLoopAnalyser:
    """
    Analyses the human approval history and produces calibration context
    for the agent system prompts.

    Args:
        approval_db_path: Path to the SQLite approvals.db from workflow.py
        lookback_days:    How many days of history to analyse (default: 30)
        min_rejections:   Minimum rejections before a pattern is surfaced (default: 2)
    """

    def __init__(
        self,
        approval_db_path: str = "./memory/approvals.db",
        lookback_days:    int = 30,
        min_rejections:   int = 2,
    ):
        self.db_path        = approval_db_path
        self.lookback_days  = lookback_days
        self.min_rejections = min_rejections
        self._llm           = ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0,
            max_tokens=1024,
        )

    # ── Step 1: Fetch raw rejection data ─────────────────────────────────────

    def fetch_rejection_patterns(self) -> dict[str, Any]:
        """
        Pull recent decisions from the approval DB and aggregate into
        symbol/action groups with their reviewer reasons.

        Returns a dict with:
          rejected:  list of {symbol, action, reason, confidence, decided_at}
          approved:  {symbol: count} — how often each symbol was approved
          stats:     {total_decided, total_rejected, rejection_rate}
        """
        if not Path(self.db_path).exists():
            logger.warning(f"Approval DB not found at {self.db_path} — skipping feedback analysis")
            return {"rejected": [], "approved": {}, "stats": {}}

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Pull recent decided recommendations
            rows = conn.execute("""
                SELECT symbol, action, confidence, rationale,
                       reviewer_reason, status, decided_at
                FROM trade_recommendations
                WHERE status IN ('rejected', 'approved', 'executed')
                  AND decided_at > datetime('now', ?)
                ORDER BY decided_at DESC
                LIMIT 200
            """, (f"-{self.lookback_days} days",)).fetchall()
            conn.close()

            rejected  = []
            approved  = Counter()
            total     = len(rows)
            total_rej = 0

            for row in rows:
                if row["status"] == "rejected":
                    rejected.append({
                        "symbol":     row["symbol"],
                        "action":     row["action"],
                        "confidence": row["confidence"],
                        "reason":     row["reviewer_reason"] or "",
                        "decided_at": row["decided_at"],
                    })
                    total_rej += 1
                else:
                    approved[row["symbol"]] += 1

            return {
                "rejected": rejected,
                "approved": dict(approved),
                "stats": {
                    "total_decided":  total,
                    "total_rejected": total_rej,
                    "rejection_rate": round(total_rej / total * 100, 1) if total else 0,
                },
            }

        except Exception as e:
            logger.error(f"Failed to fetch rejection patterns: {e}")
            return {"rejected": [], "approved": {}, "stats": {}}

    # ── Step 2: LLM analysis pass ─────────────────────────────────────────────

    def analyse_patterns(self, raw: dict[str, Any]) -> list[dict[str, str]]:
        """
        Use an LLM pass to distil the raw rejection list into specific,
        actionable calibration signals.

        Returns a list of signal dicts:
          [{signal, symbol, action, frequency, implication}, ...]
        """
        rejected = raw.get("rejected", [])
        stats    = raw.get("stats", {})
        approved = raw.get("approved", {})

        if not rejected:
            return []

        # Compute per-symbol, per-action rejection frequencies
        by_symbol_action = defaultdict(list)
        for r in rejected:
            key = f"{r['symbol']}:{r['action']}"
            by_symbol_action[key].append(r["reason"])

        # Only surface patterns that exceed min_rejections threshold
        significant = {
            k: v for k, v in by_symbol_action.items()
            if len(v) >= self.min_rejections
        }

        if not significant:
            logger.info("Feedback loop: no patterns exceed threshold, skipping LLM analysis")
            return []

        prompt_data = {
            "rejection_patterns": {
                k: {"count": len(v), "reasons": v[:5]}   # cap at 5 reasons per pattern
                for k, v in significant.items()
            },
            "approval_counts":    approved,
            "overall_stats":      stats,
        }

        messages = [
            SystemMessage(content="""You are analysing a human trader's approval/rejection history
to extract calibration signals for AI trading agents.

For each significant rejection pattern, produce ONE calibration signal that tells
the agents what to do differently. Signals must be:
  - Specific (mention RSI levels, price levels, conditions if pattern implies them)
  - Actionable (tell the agent what to change in its reasoning)
  - Honest about frequency (don't overweight single rejections)

Output ONLY a JSON array inside ```json ... ```:
[
  {
    "signal": "one-sentence calibration instruction for the agent",
    "symbol": "BTC" or "ALL" for cross-symbol patterns,
    "action": "buy" or "sell" or "all",
    "frequency": int (number of rejections driving this signal),
    "implication": "which agent should adjust: STINKMEANER, SAMUEL, GRANDDAD, or ORCHESTRATOR"
  }
]
"""),
            HumanMessage(content=f"Analyse this rejection data:\n{json.dumps(prompt_data, indent=2)}"),
        ]

        try:
            response = self._llm.invoke(messages)
            content  = response.content if hasattr(response, "content") else str(response)
            import re
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                signals = json.loads(match.group(1))
                return signals if isinstance(signals, list) else []
        except Exception as e:
            logger.error(f"LLM pattern analysis failed: {e}")

        return []

    # ── Step 3: Build calibration context block ───────────────────────────────

    def build_calibration_context(
        self,
        signals: list[dict[str, str]],
        stats:   dict[str, Any],
    ) -> str:
        """
        Format extracted signals as a concise context block to inject
        into agent system prompts.
        """
        if not signals:
            return ""

        lines = [
            "═══════════════════════════════════════════════",
            "HUMAN CALIBRATION SIGNALS (from approval history)",
            f"  Based on last {self.lookback_days} days | "
            f"Rejection rate: {stats.get('rejection_rate', 0):.1f}%",
            "═══════════════════════════════════════════════",
        ]

        for s in signals:
            symbol = s.get("symbol", "ALL")
            action = s.get("action", "all").upper()
            freq   = s.get("frequency", 0)
            target = s.get("implication", "ORCHESTRATOR")

            lines.append(
                f"• [{symbol}/{action}] (×{freq} rejections) → "
                f"{s.get('signal', '')} [Adjust: {target}]"
            )

        lines.append("═══════════════════════════════════════════════")
        lines.append(
            "These are patterns from past human decisions. Treat as priors, "
            "not hard constraints — strong data can override them."
        )
        return "\n".join(lines)

    # ── Step 4: Orchestrate ───────────────────────────────────────────────────

    def run_feedback_loop(self) -> str:
        """
        Full pipeline: fetch → analyse → format.
        Returns a calibration context string ready to inject into agent prompts.
        Call once per session start from the orchestrator.
        """
        try:
            raw     = self.fetch_rejection_patterns()
            signals = self.analyse_patterns(raw)
            context = self.build_calibration_context(signals, raw.get("stats", {}))
            logger.info(
                f"Feedback loop: {len(signals)} calibration signals extracted "
                f"from {raw['stats'].get('total_rejected', 0)} rejections"
            )
            return context
        except Exception as e:
            logger.error(f"Feedback loop failed: {e}")
            return ""


# ── Convenience function for the orchestrator node ───────────────────────────

_analyser: FeedbackLoopAnalyser | None = None


def get_feedback_context(
    approval_db_path: str = "./memory/approvals.db",
    lookback_days:    int = 30,
) -> str:
    """
    Module-level convenience function. Returns the calibration context string.
    Lazily initialises the analyser singleton.
    """
    global _analyser
    if _analyser is None or _analyser.db_path != approval_db_path:
        _analyser = FeedbackLoopAnalyser(
            approval_db_path=approval_db_path,
            lookback_days=lookback_days,
        )
    return _analyser.run_feedback_loop()
