"""
Coinbase Self-Improving Paper Trading Research Agent
---------------------------------------------------
Paper trading only. This program NEVER submits live orders.
Install:
    pip install streamlit pandas numpy requests
Run:
    streamlit run coinbase_agent.py
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
FEE_RATE = 0.006
MAX_POSITION_FRACTION = 1.00
MAX_OPEN_POSITIONS = 12
MIN_POSITION_GBP = 2.0
CANDLE_GRANULARITY = 900
CANDLE_LIMIT = 300
REQUEST_TIMEOUT = 12
QUOTE_CURRENCIES = {"GBP", "USD"}

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
def get_ticker(product_id: str) -> dict:
    return api_get(f"/products/{product_id}/ticker")

@st.cache_data(ttl=30, show_spinner=False)
def get_candles(product_id: str, granularity: int = CANDLE_GRANULARITY) -> pd.DataFrame:
    raw = api_get(
        f"/products/{product_id}/candles",
        {"granularity": granularity},
    )
    if not raw:
        return pd.DataFrame()
    frame = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    for col in ["low", "high", "open", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("time").drop_duplicates("time").tail(CANDLE_LIMIT).reset_index(drop=True)

@st.cache_data(ttl=300, show_spinner=False)
def get_gbp_usd_rate() -> float:
    try:
        payload = requests.get(
            "https://api.coinbase.com/v2/exchange-rates",
            params={"currency": "GBP"},
            timeout=REQUEST_TIMEOUT,
        )
        payload.raise_for_status()
        return float(payload.json()["data"]["rates"]["USD"])
    except Exception:
        return 1.27

def to_gbp(amount: float, quote: str, gbp_usd: float) -> float:
    if quote == "GBP":
        return amount
    if quote == "USD":
        return amount / gbp_usd if gbp_usd > 0 else amount / 1.27
    return amount

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["ema_fast"] = close.ewm(span=9, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=21, adjust=False).mean()
    out["sma20"] = close.rolling(20).mean()
    out["std20"] = close.rolling(20).std()
    out["upper"] = out["sma20"] + 2 * out["std20"]
    out["lower"] = out["sma20"] - 2 * out["std20"]
    out["vol_mean"] = out["volume"].rolling(20).mean()
    out["ret"] = close.pct_change()
    return out

def strategy_signal(frame: pd.DataFrame, strategy: str) -> pd.Series:
    d = add_indicators(frame)
    if strategy == "EMA momentum":
        buy = 
