#!/usr/bin/env python3
"""
TradingAgents — local Gemma 4 run.

Quick roles  (analysts, researchers, trader, debaters): gemma-4-e4b on gaming PC
Deep roles   (research manager, portfolio manager):     gemma-4-26B on Mac mini

Usage:
    uv run python run_local.py [TICKER] [DATE]

    TICKER  Stock symbol (default: NVDA)
    DATE    Analysis date YYYY-MM-DD (default: 2025-01-10)

Examples:
    uv run python run_local.py
    uv run python run_local.py AAPL 2025-03-07
"""
import os
import sys
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Endpoints — override for your own inference hosts
GAMING_PC_URL  = os.environ.get("TRADING_GAMING_PC_URL", "http://localhost:1234")
MAC_MINI_URL   = os.environ.get("TRADING_MAC_MINI_URL", "http://localhost:8082")
GAMING_PC_MODEL = "google/gemma-4-e4b"
MAC_MINI_MODEL  = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
date   = sys.argv[2] if len(sys.argv) > 2 else "2025-01-10"

# --- Phase 1: all-local (Mac mini 26B for both tiers) ---
# When ANTHROPIC_API_KEY is available, swap deep_think to anthropic/claude-sonnet-4-6.
# For now both tiers use the Mac mini 26B model.
config = DEFAULT_CONFIG.copy()
config["llm_provider"]     = "ollama"
config["backend_url"]      = MAC_MINI_URL
config["quick_think_llm"]  = MAC_MINI_MODEL
config["deep_think_llm"]   = MAC_MINI_MODEL
config["max_debate_rounds"]      = 1
config["max_risk_discuss_rounds"] = 1
config["checkpoint_enabled"]     = True  # resume from last good step if it crashes

print(f"\n{'='*60}")
print(f"  TradingAgents — Local Run")
print(f"  Ticker: {ticker}  |  Date: {date}")
print(f"  Quick/Deep: {MAC_MINI_MODEL} @ Mac mini")
print(f"{'='*60}\n")

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate(ticker, date)

print(f"\n{'='*60}")
print("  FINAL DECISION")
print(f"{'='*60}")
print(decision)
