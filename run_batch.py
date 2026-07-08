#!/usr/bin/env python3
"""
TradingAgents — batch analysis runner.

Accepts a comma-separated list of tickers and a date. Runs multi-agent
analysis for each, prints a JSON array of results to stdout.

Usage:
    uv run python run_batch.py TICKER1,TICKER2 YYYY-MM-DD

Output (stdout): JSON array
[
  {
    "ticker": "NVDA",
    "date": "2025-01-10",
    "signal": "BUY",
    "decision": "Full decision text...",
    "research_verdict": "Research manager verdict...",
    "risk_verdict": "Portfolio manager verdict...",
    "error": null,
    "duration_seconds": 45
  },
  ...
]

Progress/errors go to stderr so stdout stays clean JSON.
"""
import json
import os
import sys
import time
import traceback

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env next to this script (no external deps).

    Real environment variables take precedence over .env values.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

MAC_MINI_URL = os.environ.get("TRADING_MAC_MINI_URL", "http://localhost:8082")
MAC_MINI_MODEL = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

GAMING_PC_URL = os.environ.get("TRADING_GAMING_PC_URL", "http://localhost:1234/v1")
GAMING_PC_MODEL = "google/gemma-4-e4b"

# Override via env vars for comparison runs:
#   TRADING_BACKEND_URL=http://... TRADING_MODEL=model-name uv run python run_batch.py ...
_BACKEND_URL = os.environ.get("TRADING_BACKEND_URL", MAC_MINI_URL)
_MODEL = os.environ.get("TRADING_MODEL", MAC_MINI_MODEL)


def build_graph() -> TradingAgentsGraph:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "ollama"
    config["backend_url"] = _BACKEND_URL
    config["quick_think_llm"] = _MODEL
    config["deep_think_llm"] = _MODEL
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["checkpoint_enabled"] = False
    return TradingAgentsGraph(debug=False, config=config)


def analyze(ticker: str, date: str, ta: TradingAgentsGraph) -> dict:
    start = time.time()
    try:
        final_state, signal = ta.propagate(ticker, date)
        invest = final_state.get("investment_debate_state") or {}
        risk = final_state.get("risk_debate_state") or {}
        return {
            "ticker": ticker,
            "date": date,
            "signal": signal,
            "decision": final_state.get("final_trade_decision", ""),
            "research_verdict": invest.get("judge_decision", ""),
            "risk_verdict": risk.get("judge_decision", ""),
            "error": None,
            "duration_seconds": round(time.time() - start, 1),
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "date": date,
            "signal": "ERROR",
            "decision": "",
            "research_verdict": "",
            "risk_verdict": "",
            "error": traceback.format_exc(),
            "duration_seconds": round(time.time() - start, 1),
        }


def main():
    if len(sys.argv) < 3:
        print("Usage: run_batch.py TICKER1,TICKER2 YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    tickers = [t.strip().upper() for t in sys.argv[1].split(",") if t.strip()]
    date = sys.argv[2]

    print(f"Initializing TradingAgents — {len(tickers)} ticker(s), date {date}", file=sys.stderr)
    ta = build_graph()

    results = []
    for ticker in tickers:
        print(f"Analyzing {ticker}...", file=sys.stderr, flush=True)
        result = analyze(ticker, date, ta)
        results.append(result)
        status = result["signal"] if not result["error"] else f"ERROR: {str(result['error'])[:80]}"
        print(f"  {ticker}: {status} ({result['duration_seconds']}s)", file=sys.stderr)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
