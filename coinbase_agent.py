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
    spend_quote = budget_quote / (1 + FEE_RATE)
    units = spend_quote / price
    state["cash_gbp"] -= budget_gbp
    state["positions"][product_id] = {
        "product_id": product_id,
        "quote": quote,
        "units": units,
        "entry_price": price,
        "last_price": price,
        "cost_gbp": budget_gbp,
        "strategy": strategy,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "entry_fee_rate": FEE_RATE,
    }
    save_state(state)
    return f"Paper buy recorded: £{budget_gbp:.2f} allocated to {product_id}."

def paper_sell(state: dict, product_id: str, price: float, gbp_usd: float, reason: str = "manual paper exit") -> str:
    pos = state["positions"].get(product_id)
    if not pos:
        return "No open paper position."
    gross_quote = float(pos["units"]) * price
    net_quote = gross_quote * (1 - FEE_RATE)
    proceeds_gbp = to_gbp(net_quote, pos["quote"], gbp_usd)
    pnl_gbp = proceeds_gbp - float(pos["cost_gbp"])
    state["cash_gbp"] += proceeds_gbp
    state["closed_trades"].append({
        **pos,
        "exit_price": price,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "proceeds_gbp": proceeds_gbp,
        "pnl_gbp": pnl_gbp,
        "points": pnl_gbp,
        "exit_reason": reason,
    })
    del state["positions"][product_id]
    save_state(state)
    return f"Paper sell recorded: {product_id}; net P&L £{pnl_gbp:.2f} ({pnl_gbp:+.2f} points)."

def aggressive_momentum_cycle(state: dict, products: pd.DataFrame, gbp_usd: float):
    activity, errors = [], []
    
    # 1. Manage existing positions: take profit if +3% or cut losses if -2.5%
    for product_id, pos in list(state["positions"].items()):
        try:
            ticker = get_ticker(product_id)
            price = float(ticker["price"])
            pos["last_price"] = price
            entry = float(pos["entry_price"])
            pnl_pct = (price / entry - 1) * 100
            
            if pnl_pct >= 3.0:
                msg = paper_sell(state, product_id, price, gbp_usd, reason="Target profit hit (+3%)")
                activity.append({"Market": product_id, "Action": "TAKE PROFIT", "Reason": f"Gain: {pnl_pct:.2f}%", "Details": msg})
            elif pnl_pct <= -2.5:
                msg = paper_sell(state, product_id, price, gbp_usd, reason="Stop loss triggered (-2.5%)")
                activity.append({"Market": product_id, "Action": "STOP LOSS", "Reason": f"Loss: {pnl_pct:.2f}%", "Details": msg})
            else:
                activity.append({"Market": product_id, "Action": "HOLD", "Reason": f"P&L: {pnl_pct:+.2f}%", "Details": f"Price: {price:g}"})
        except Exception as exc:
            errors.append(f"{product_id}: {type(exc).__name__}: {exc}")

    # 2. Scan ALL markets for aggressive momentum breakout entries
    if len(state["positions"]) < MAX_OPEN_POSITIONS and state["cash_gbp"] >= MIN_POSITION_GBP:
        candidates = []
        for _, product in products.iterrows():
            product_id = str(product["product_id"])
            if product_id in state["positions"]:
                continue
            try:
                candles = get_candles(product_id)
                if len(candles) < 15:
                    continue
                close = candles["close"]
                volume = candles["volume"]
                
                # Check recent 1-hour and 4-hour performance + volume surge
                ret_1h = (close.iloc[-1] / close.iloc[-4] - 1) * 100
                ret_4h = (close.iloc[-1] / close.iloc[-16] - 1) * 100 if len(close) >= 16 else ret_1h
                vol_recent = volume.tail(3).mean()
                vol_baseline = volume.iloc[-20:-3].mean() if len(volume) >= 20 else vol_recent
                vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1.0
                
                # High-momentum criteria: Positive momentum with a volume expansion
                if ret_1h > 0.8 and ret_4h > 1.5 and vol_ratio > 1.3:
                    score = ret_1h + (vol_ratio * 2)
                    candidates.append({
                        "product_id": product_id,
                        "quote": product["quote"],
                        "score": score,
                        "ret_1h": ret_1h,
                        "vol_ratio": vol_ratio
                    })
            except Exception:
                continue
            time.sleep(0.02)
            
        # Sort candidates by best momentum score
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        
        for cand in candidates:
            if len(state["positions"]) >= MAX_OPEN_POSITIONS or state["cash_gbp"] < MIN_POSITION_GBP:
                break
            pid = cand["product_id"]
            try:
                ticker = get_ticker(pid)
                price = float(ticker["price"])
                cash = float(state["cash_gbp"])
                # Allocate a healthy fraction (up to 30% of remaining cash per trade)
                allocation = min(cash, max(MIN_POSITION_GBP, cash * 0.30))
                msg = paper_buy(state, pid, cand["quote"], price, gbp_usd, "Momentum Breakout", allocation_gbp=allocation)
                activity.append({
                    "Market": pid, 
                    "Action": "MOMENTUM BUY", 
                    "Reason": f"1h move: +{cand['ret_1h']:.2f}%, Vol ratio: {cand['vol_ratio']:.1f}x", 
                    "Details": msg
                })
            except Exception as exc:
                errors.append(f"{pid}: {type(exc).__name__}: {exc}")

    save_state(state)
    if not activity:
        activity.append({"Market": "—", "Action": "SCANNING", "Reason": "Searching for high-momentum breakouts", "Details": ""})
    return activity, errors

st.set_page_config(page_title="Coinbase Momentum Trading Agent", page_icon="🚀", layout="wide")
st.title("🚀 Coinbase High-Momentum Paper Trading Agent")
st.caption("Scanning ALL GBP/USD assets • Aggressive intraday momentum capture • Simulated orders only")

if "state" not in st.session_state:
    st.session_state.state = load_state()
state = st.session_state.state

try:
    products = get_products()
    gbp_usd = get_gbp_usd_rate()
    st.caption(f"Total active exchange markets: {len(products)} · GBP/USD rate: £1 ≈ ${gbp_usd:.4f}")
except Exception as exc:
    st.error(f"Could not load Coinbase public market data: {exc}")
    st.stop()

st.subheader("Live momentum execution engine")
st.write("Continuously scans the entire exchange for volume spikes and upward breakouts every 30 seconds.")

@st.fragment(run_every=30)
def continuous_trader():
    current_state = st.session_state.state
    with st.spinner("Executing high-momentum trading cycle…"):
        activity, cycle_errors = aggressive_momentum_cycle(current_state, products, gbp_usd)
        portfolio_value, price_errors_now = mark_positions(current_state, products, gbp_usd)
    net_pnl = portfolio_value - float(current_state["starting_cash_gbp"])
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Paper portfolio", f"£{portfolio_value:.2f}", f"{net_pnl:+.2f} vs start")
    m2.metric("Available cash", f"£{float(current_state['cash_gbp']):.2f}")
    m3.metric("Open positions", str(len(current_state["positions"])))
    
    if price_errors_now:
        st.warning("Some price requests failed. " + " | ".join(price_errors_now[:4]))
    st.session_state.last_auto_activity = pd.DataFrame(activity)
    st.session_state.last_auto_errors = cycle_errors + price_errors_now
    st.caption(f"Last cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if "last_auto_activity" in st.session_state:
        st.dataframe(st.session_state.last_auto_activity, use_container_width=True, hide_index=True)

continuous_trader()

tab_positions, tab_memory = st.tabs(["Open positions & manual controls", "Performance ledger"])

with tab_positions:
    st.subheader("Active paper positions")
    if st.button("Reset paper account to £100", type="secondary"):
        state = default_state()
        st.session_state.state = state
        save_state(state)
        st.success("Paper account reset to £100.")
        st.rerun()
    if state["positions"]:
        pos_rows = []
        for pid, pos in state["positions"].items():
            px = float(pos.get("last_price", pos["entry_price"]))
            current_gbp = to_gbp(float(pos["units"]) * px * (1 - FEE_RATE), pos["quote"], gbp_usd)
            pnl = current_gbp - float(pos["cost_gbp"])
            pos_rows.append({
                "Market": pid,
                "Strategy": pos["strategy"],
                "Entry price": pos["entry_price"],
                "Latest price": px,
                "Cost £": pos["cost_gbp"],
                "Est. P&L £": round(pnl, 2),
                "Opened": pos["opened_at"],
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
        selected = st.selectbox("Position to manually close", list(state["positions"].keys()))
        if st.button("Close selected position"):
            try:
                current_price = float(get_ticker(selected)["price"])
                msg = paper_sell(state, selected, current_price, gbp_usd, reason="Manual exit")
                st.success(msg)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not close position: {exc}")
    else:
        st.info("No open paper positions currently.")

with tab_memory:
    st.subheader("Closed trades history")
    closed = state.get("closed_trades", [])
    if closed:
        st.dataframe(pd.DataFrame(closed), use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades yet.")

st.divider()
st.caption("Paper trading research agent only. Does not place real orders.")
