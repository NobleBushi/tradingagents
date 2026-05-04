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
import sys
import time
import traceback

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

MAC_MINI_URL = "http://localhost:8082"
MAC_MINI_MODEL = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"


def build_graph() -> TradingAgentsGraph:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "ollama"
    config["backend_url"] = MAC_MINI_URL
    config["quick_think_llm"] = MAC_MINI_MODEL
    config["deep_think_llm"] = MAC_MINI_MODEL
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["checkpoint_enabled"] = True
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
