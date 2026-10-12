"""
Coinbase Self-Improving Paper Trading Research Agent
---------------------------------------------------
Paper trading only. This program NEVER submits live orders.
Install:
    py -m pip install streamlit pandas numpy requests
Run:
    py -m streamlit run coinbase_intraday_paper_agent.py
Notes:
- Uses public Coinbase Exchange REST endpoints only.
- Strategy discovery is systematic hypothesis testing, not unrestricted AI.
- Historical performance is not a promise of future profit.
"""
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
import streamlit as st

API = "https://api.exchange.coinbase.com"
DATA_DIR = Path(__file__).resolve().parent
MEMORY_FILE = DATA_DIR / "strategy_memory.json"
STATE_FILE = DATA_DIR / "paper_state.json"
STARTING_CASH_GBP = 100.0
FEE_RATE = 0.006  # conservative placeholder; adjust to your actual Coinbase fee tier
# Portfolio allocation is adaptive: no fixed £20 tranches and no fixed number of positions.
# The agent allocates a larger share only to stronger, independently tested candidates.
MAX_POSITION_FRACTION = 1.00
MAX_OPEN_POSITIONS = 12
MIN_POSITION_GBP = 2.0
CANDLE_GRANULARITY = 900  # 15 minutes
CANDLE_LIMIT = 300
REQUEST_TIMEOUT = 12
QUOTE_CURRENCIES = {"GBP", "USD"}

# ------------------------------- Public data -------------------------------
def api_get(path: str, params: dict | None = None) -> Any:
    response = requests.get(
        f"{API}{path}",
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "Coinbase-Paper-Research-Agent/1.0"},
    )
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=20, show_spinner=False)
def get_products() -> pd.DataFrame:
    raw = api_get("/products")
    rows = []
    for item in raw:
        if item.get("quote_currency") not in QUOTE_CURRENCIES:
            continue
        if item.get("status") != "online" or item.get("trading_disabled", False):
            continue
        rows.append({
            "product_id": item.get("id"),
            "base": item.get("base_currency"),
            "quote": item.get("quote_currency"),
            "base_min_size": float(item.get("base_min_size") or 0),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=5, show_spinner=False)
def get_ticker(product_id: str) -> 
