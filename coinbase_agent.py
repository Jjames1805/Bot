# Coinbase Self-Improving Paper Trading Research Agent
# Paper trading only. This program NEVER submits live orders.
# Install: pip install streamlit pandas numpy requests
# Run: streamlit run coinbase_agent.py

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
MAX_OPEN_POSITIONS = 8
MIN_POSITION_GBP = 5.0
CANDLE_GRANULARITY = 900
CANDLE_LIMIT = 200
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
    raw = api_get(f"/products/{product_id}/candles", {"granularity": granularity})
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

def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

def default_state() -> dict:
    return {
        "cash_gbp": STARTING_CASH_GBP,
        "positions": {},
        "closed_trades": [],
        "last_prices": {},
        "starting_cash_gbp": STARTING_CASH_GBP,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

def load_state() -> dict:
    return read_json(STATE_FILE, default_state())

def save_state(state: dict) -> None:
    write_json(STATE_FILE, state)

def load_memory() -> list:
    return read_json(MEMORY_FILE, [])

def save_experiment(row: dict) -> None:
    memory = load_memory()
    memory.append({**row, "tested_at": datetime.now(timezone.utc).isoformat()})
    write_json(MEMORY_FILE, memory[-2000:])

def mark_positions(state: dict, products: pd.DataFrame, gbp_usd: float) -> tuple[float, list[str]]:
    errors = []
    for product_id, pos in list(state["positions"].items()):
        try:
            ticker = get_ticker(product_id)
            price = float(ticker["price"])
            if price > 0:
                pos["last_price"] = price
                pos["last_price_at"] = datetime.now(timezone.utc).isoformat()
                state["last_prices"][product_id] = price
        except Exception as exc:
            errors.append(f"{product_id}: {exc}")
    value = float(state["cash_gbp"])
    for product_id, pos in state["positions"].items():
        quote = pos["quote"]
        value += to_gbp(float(pos["units"]) * float(pos.get("last_price", pos["entry_price"])), quote, gbp_usd)
    save_state(state)
    return value, errors

def paper_buy(state: dict, product_id: str, quote: str, price: float, gbp_usd: float, strategy: str, allocation_gbp: float | None = None) -> str:
    if product_id in state["positions"]:
        return "Already holding this market."
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return f"Safety limit reached ({MAX_OPEN_POSITIONS} simultaneous positions)."
    available = float(state["cash_gbp"])
    if available < MIN_POSITION_GBP:
        return "Not enough paper cash."
    if allocation_gbp is None:
        allocation_gbp = available
    budget_gbp = max(0.0, min(float(allocation_gbp), available))
    if budget_gbp < MIN_POSITION_GBP:
        return f"Allocation must be at least £{MIN_POSITION_GBP:.2f}."
    budget_quote = budget_gbp if quote == "GBP" else budget_gbp * gbp_usd
    spend_quote = budget_quote
