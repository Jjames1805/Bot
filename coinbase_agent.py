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
def get_ticker(product_id: str) -> dict:
    return api_get(f"/products/{product_id}/ticker")


@st.cache_data(ttl=30, show_spinner=False)
def get_candles(product_id: str, granularity: int = CANDLE_GRANULARITY) -> pd.DataFrame:
    raw = api_get(
        f"/products/{product_id}/candles",
        {"granularity": granularity},
    )
    # Coinbase rows: [time, low, high, open, close, volume]
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
        return 1.27  # fallback only for display conversion; clearly disclosed in UI


def to_gbp(amount: float, quote: str, gbp_usd: float) -> float:
    if quote == "GBP":
        return amount
    if quote == "USD":
        return amount / gbp_usd if gbp_usd > 0 else amount / 1.27
    return amount


# ---------------------------- Strategy laboratory ---------------------------

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
        buy = (d["ema_fast"] > d["ema_slow"]) & (d["ema_fast"].shift(1) <= d["ema_slow"].shift(1))
        sell = (d["ema_fast"] < d["ema_slow"]) & (d["ema_fast"].shift(1) >= d["ema_slow"].shift(1))
    elif strategy == "RSI reversal":
        buy = (d["rsi"] < 30) & (d["rsi"].shift(1) >= 30)
        sell = (d["rsi"] > 55) & (d["rsi"].shift(1) <= 55)
    elif strategy == "Bollinger rebound":
        buy = (d["close"] < d["lower"]) & (d["close"].shift(1) >= d["lower"].shift(1))
        sell = (d["close"] > d["sma20"])
    elif strategy == "Volume breakout":
        buy = (d["close"] > d["high"].shift(1).rolling(20).max()) & (d["volume"] > 1.5 * d["vol_mean"])
        sell = (d["close"] < d["ema_fast"])
    elif strategy == "Trend pullback":
        buy = (d["ema_fast"] > d["ema_slow"]) & (d["close"] > d["ema_slow"]) & (d["rsi"].between(40, 55))
        sell = (d["ema_fast"] < d["ema_slow"]) | (d["rsi"] > 72)
    else:
        buy = pd.Series(False, index=d.index)
        sell = pd.Series(False, index=d.index)
    signal = pd.Series(0, index=d.index, dtype=int)
    signal.loc[buy.fillna(False)] = 1
    signal.loc[sell.fillna(False)] = -1
    return signal


STRATEGIES = [
    "EMA momentum",
    "RSI reversal",
    "Bollinger rebound",
    "Volume breakout",
    "Trend pullback",
]


def backtest(frame: pd.DataFrame, strategy: str, fee_rate: float = FEE_RATE) -> dict:
    """Long-only, next-bar execution approximation; includes fee on entry and exit."""
    if len(frame) < 80:
        return {"trades": 0, "return_pct": 0.0, "win_rate": 0.0, "max_drawdown_pct": 0.0, "score": -999.0}
    d = frame.copy().reset_index(drop=True)
    signals = strategy_signal(d, strategy)
    cash = 1.0
    units = 0.0
    entry_value = 0.0
    trades = []
    equity = []
    for i in range(1, len(d)):
        price = float(d.loc[i, "open"])  # signal from previous candle, execution at next open
        sig = int(signals.iloc[i - 1])
        if sig == 1 and units == 0 and price > 0:
            spend = cash * (1 - fee_rate)
            units = spend / price
            entry_value = cash
            cash = 0.0
        elif sig == -1 and units > 0 and price > 0:
            proceeds = units * price * (1 - fee_rate)
            trades.append((proceeds / entry_value) - 1)
            cash = proceeds
            units = 0.0
            entry_value = 0.0
        equity.append(cash + units * float(d.loc[i, "close"]))
    if units > 0:
        proceeds = units * float(d.iloc[-1]["close"]) * (1 - fee_rate)
        trades.append((proceeds / entry_value) - 1)
        equity[-1] = proceeds
    if not equity:
        equity = [1.0]
    eq = pd.Series(equity, dtype=float)
    peak = eq.cummax()
    drawdown = (eq / peak - 1) * 100
    total_return = (float(eq.iloc[-1]) - 1) * 100
    win_rate = 100 * sum(x > 0 for x in trades) / len(trades) if trades else 0.0
    # Penalize drawdown and tiny sample sizes; this is a screening score, not a forecast.
    score = total_return - 0.7 * abs(float(drawdown.min())) + min(len(trades), 10) * 0.25
    if len(trades) < 4:
        score -= (4 - len(trades)) * 2
    return {
        "trades": len(trades),
        "return_pct": round(total_return, 3),
        "win_rate": round(win_rate, 1),
        "max_drawdown_pct": round(float(drawdown.min()), 3),
        "score": round(score, 3),
    }


def split_test(frame: pd.DataFrame, strategy: str) -> dict:
    """Compare an in-sample section with a later holdout section."""
    cut = int(len(frame) * 0.70)
    train = backtest(frame.iloc[:cut].reset_index(drop=True), strategy)
    test = backtest(frame.iloc[cut:].reset_index(drop=True), strategy)
    return {
        "strategy": strategy,
        "train_return_pct": train["return_pct"],
        "test_return_pct": test["return_pct"],
        "test_trades": test["trades"],
        "test_win_rate_pct": test["win_rate"],
        "test_max_drawdown_pct": test["max_drawdown_pct"],
        "research_score": round(test["score"], 3),
        "verdict": (
            "Candidate for paper trial"
            if test["trades"] >= 4 and test["return_pct"] > 0 and test["max_drawdown_pct"] > -12
            else "Reject / gather more data"
        ),
    }


def discover_strategies(product_id: str) -> pd.DataFrame:
    """Test each strategy independently and always return a consistently shaped table."""
    frame = get_candles(product_id)
    if frame.empty or len(frame) < 100:
        return pd.DataFrame(columns=[
            "strategy", "train_return_pct", "test_return_pct", "test_trades",
            "test_win_rate_pct", "test_max_drawdown_pct", "research_score", "verdict"
        ])

    rows = []
    for strategy in STRATEGIES:
        try:
            row = split_test(frame, strategy)
            rows.append(row)
        except Exception as exc:
            # Keep a failed strategy visible without breaking the entire research run.
            rows.append({
                "strategy": strategy,
                "train_return_pct": np.nan,
                "test_return_pct": np.nan,
                "test_trades": 0,
                "test_win_rate_pct": np.nan,
                "test_max_drawdown_pct": np.nan,
                "research_score": np.nan,
                "verdict": f"Test error: {type(exc).__name__}: {exc}",
            })

    result = pd.DataFrame(rows)
    # Defensive schema: pandas cannot sort by a column if every strategy failed
    # before returning its normal result, so ensure the expected columns exist.
    expected_columns = [
        "strategy", "train_return_pct", "test_return_pct", "test_trades",
        "test_win_rate_pct", "test_max_drawdown_pct", "research_score", "verdict"
    ]
    for column in expected_columns:
        if column not in result.columns:
            result[column] = np.nan
    return result.sort_values(
        "research_score", ascending=False, na_position="last"
    ).reset_index(drop=True)


def allocate_candidate_budgets(candidates: pd.DataFrame, cash_gbp: float) -> dict[str, float]:
    """Allocate paper cash proportionally to positive research scores, with diversification caps.

    This is a simple heuristic, not a claim that the highest backtest score predicts future profit.
    Only candidates that pass holdout screening are eligible.
    """
    if candidates.empty or cash_gbp < MIN_POSITION_GBP:
        return {}
    eligible = candidates[candidates["verdict"] == "Candidate for paper trial"].copy()
    if eligible.empty:
        return {}
    eligible = eligible.drop_duplicates(subset=["market"], keep="first")
    eligible["weight"] = pd.to_numeric(eligible["research_score"], errors="coerce").fillna(0).clip(lower=0)
    if eligible["weight"].sum() <= 0:
        eligible["weight"] = 1.0
    # Diversify: no single new position can consume more than 40% of currently available cash.
    max_per_position = cash_gbp * 0.40
    budgets = {}
    remaining = cash_gbp
    rows = eligible.sort_values("weight", ascending=False).to_dict("records")
    weights_total = sum(max(float(row["weight"]), 0.01) for row in rows)
    for row in rows:
        if remaining < MIN_POSITION_GBP:
            break
        weight = max(float(row["weight"]), 0.01)
        desired = cash_gbp * weight / weights_total
        amount = min(desired, max_per_position, remaining)
        if amount >= MIN_POSITION_GBP:
            budgets[str(row["market"])] = round(amount, 2)
            remaining -= amount
    return budgets


# ----------------------------- Persistent memory ----------------------------

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


# ----------------------------- Paper portfolio ------------------------------

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


def paper_buy(
    state: dict,
    product_id: str,
    quote: str,
    price: float,
    gbp_usd: float,
    strategy: str,
    allocation_gbp: float | None = None,
) -> str:
    """Record a simulated buy using a chosen GBP allocation, not a fixed tranche."""
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
        "points": pnl_gbp,  # one point per £1 net profit/loss
        "exit_reason": reason,
    })
    del state["positions"][product_id]
    save_state(state)
    return f"Paper sell recorded: {product_id}; net P&L £{pnl_gbp:.2f} ({pnl_gbp:+.2f} points)."


# -------------------------- Automatic paper trader --------------------------

def latest_strategy_action(frame: pd.DataFrame, strategy: str) -> tuple[str, str]:
    """Return BUY/SELL/WAIT from the most recently completed candle signal."""
    if frame.empty or len(frame) < 80:
        return "WAIT", "Not enough candle history"
    try:
        signals = strategy_signal(frame, strategy)
        signal = int(signals.iloc[-2])  # previous candle, avoiding incomplete-candle decisions
        if signal == 1:
            return "BUY", "Buy signal on latest completed candle"
        if signal == -1:
            return "SELL", "Sell signal on latest completed candle"
        return "WAIT", "No entry/exit signal on latest completed candle"
    except Exception as exc:
        return "WAIT", f"Signal calculation failed: {type(exc).__name__}: {exc}"


def auto_paper_cycle(state: dict, products: pd.DataFrame, gbp_usd: float, market_limit: int = 10):
    """Manage paper positions and consider new entries. Never sends live orders."""
    activity, errors = [], []

    for product_id, pos in list(state["positions"].items()):
        try:
            frame = get_candles(product_id)
            action, reason = latest_strategy_action(frame, pos.get("strategy", "EMA momentum"))
            price = float(get_ticker(product_id)["price"])
            pos["last_price"] = price
            pos["last_price_at"] = datetime.now(timezone.utc).isoformat()
            state["last_prices"][product_id] = price
            if action == "SELL":
                message = paper_sell(state, product_id, price, gbp_usd, reason)
                activity.append({"Market": product_id, "Action": "PAPER SELL", "Reason": reason, "Details": message})
            else:
                activity.append({"Market": product_id, "Action": "HOLD", "Reason": reason, "Details": f"Latest price: {price:g}"})
        except Exception as exc:
            errors.append(f"{product_id}: {type(exc).__name__}: {exc}")

    if len(state["positions"]) < MAX_OPEN_POSITIONS and state["cash_gbp"] >= MIN_POSITION_GBP:
        for _, product in products.head(market_limit).iterrows():
            if len(state["positions"]) >= MAX_OPEN_POSITIONS or state["cash_gbp"] < MIN_POSITION_GBP:
                break
            product_id = str(product["product_id"])
            if product_id in state["positions"]:
                continue
            try:
                frame = get_candles(product_id)
                if frame.empty or len(frame) < 100:
                    activity.append({"Market": product_id, "Action": "SKIP", "Reason": "Not enough candle history", "Details": ""})
                    continue
                tested = []
                strategy_errors = []
                for strategy in STRATEGIES:
                    try:
                        result = split_test(frame, strategy)
                        result["strategy"] = strategy
                        tested.append(result)
                    except Exception as exc:
                        error_text = f"{strategy}: {type(exc).__name__}: {exc}"
                        strategy_errors.append(error_text)
                        errors.append(f"{product_id}/{error_text}")
                if not tested:
                    activity.append({
                        "Market": product_id, "Action": "SKIP",
                        "Reason": "All strategy tests failed",
                        "Details": "; ".join(strategy_errors) or "No strategy returned a result",
                    })
                    continue
                df = pd.DataFrame(tested)
                # Rank every strategy, even when the conservative holdout gate rejects it.
                # Holdout results remain visible in research, but do not permanently block
                # the paper agent from trying a current signal.
                eligible = df.copy()
                eligible["research_score"] = pd.to_numeric(eligible["research_score"], errors="coerce").fillna(-999)
                eligible = eligible.sort_values("research_score", ascending=False)

                # Check every strategy's latest completed-candle signal, not only the
                # strategies that passed the historical holdout filter.
                active_candidates = []
                signal_reasons = []
                for _, candidate in eligible.iterrows():
                    candidate_strategy = str(candidate["strategy"])
                    candidate_action, candidate_reason = latest_strategy_action(frame, candidate_strategy)
                    signal_reasons.append(f"{candidate_strategy}: {candidate_action}")
                    if candidate_action == "BUY":
                        active_candidates.append((candidate, candidate_strategy, candidate_reason))
                if not active_candidates:
                    # More frequent, transparent fallback entry: use the existing
                    # Trend pullback family when the completed candle shows a healthy
                    # uptrend and RSI is not already overheated. This is still paper-only.
                    try:
                        indicators = add_indicators(frame)
                        last = indicators.iloc[-2]  # most recently completed candle
                        if (
                            pd.notna(last["ema_fast"]) and pd.notna(last["ema_slow"])
                            and pd.notna(last["rsi"])
                            and last["close"] >= last["ema_slow"]
                            and last["ema_fast"] >= last["ema_slow"]
                            and 30 <= float(last["rsi"]) <= 70
                        ):
                            fallback_row = eligible[eligible["strategy"] == "Trend pullback"]
                            best = fallback_row.iloc[0] if not fallback_row.empty else eligible.iloc[0]
                            strategy = "Trend pullback"
                            reason = "Fallback entry: completed candle confirms uptrend and RSI is 30–70"
                            active_candidates.append((best, strategy, reason))
                    except Exception as exc:
                        errors.append(f"{product_id}/fallback signal: {type(exc).__name__}: {exc}")

                if not active_candidates:
                    activity.append({
                        "Market": product_id, "Action": "WAIT",
                        "Reason": "No current buy signal; waiting for market conditions",
                        "Details": "; ".join(signal_reasons),
                    })
                    continue
                best, strategy, reason = active_candidates[0]

                cash = float(state["cash_gbp"])
                recent_vol = float(frame["close"].pct_change().tail(24).std() or 0)
                base_fraction = min(0.40, max(0.10, 0.10 + min(float(best["research_score"]), 10) * 0.02))
                volatility_factor = min(1.0, 0.02 / max(recent_vol, 0.0001))
                allocation = min(cash * 0.40, cash * base_fraction * volatility_factor)
                if allocation < MIN_POSITION_GBP:
                    activity.append({"Market": product_id, "Action": "SKIP", "Reason": "Calculated allocation below minimum", "Details": ""})
                    continue
                price = float(get_ticker(product_id)["price"])
                message = paper_buy(state, product_id, str(product["quote"]), price, gbp_usd, strategy, allocation_gbp=allocation)
                activity.append({"Market": product_id, "Action": "PAPER BUY", "Reason": reason, "Details": f"Strategy: {strategy}. {message}"})
            except Exception as exc:
                errors.append(f"{product_id}: {type(exc).__name__}: {exc}")

    save_state(state)
    if not activity:
        activity.append({"Market": "—", "Action": "NO ACTION", "Reason": "No available signal or position action", "Details": "Check the errors and market history."})
    return activity, errors


# ----------------------------------- UI --------------------------------------

st.set_page_config(page_title="Coinbase Paper Research Agent", page_icon="🧪", layout="wide")
st.title("🧪 Coinbase Self-Improving Paper Trading Agent")
st.caption("Public market data • GBP/USD markets • Simulated trades only • No live orders")

if "state" not in st.session_state:
    st.session_state.state = load_state()
state = st.session_state.state

# Automatic paper trading is always enabled; no sidebar controls are required.
auto_scan = True
max_markets = 30
auto_market_limit = 30
st.caption("Continuous paper trading is always enabled while this app is open. Public Coinbase prices only; no live orders.")

try:
    products = get_products()
    gbp_usd = get_gbp_usd_rate()
    st.caption(f"Markets found: {len(products)} · GBP/USD reference rate: £1 ≈ ${gbp_usd:.4f}")
except Exception as exc:
    st.error(f"Could not load Coinbase public market data: {exc}")
    st.stop()

# Portfolio totals are rendered inside the automatic fragment below so they refresh
# together with each cycle instead of leaving the top-line metrics stale.

st.subheader("Continuous automatic paper trading")
st.write("Automatically checks all configured strategies across up to 30 markets, manages open paper positions, and updates price marks every 30 seconds while this app remains open. Signals use completed candles, so a trade will not necessarily happen every cycle.")
@st.fragment(run_every=30)
def continuous_paper_trader():
    current_state = st.session_state.state
    with st.spinner("Running automatic paper-trading cycle…"):
        activity, cycle_errors = auto_paper_cycle(current_state, products, gbp_usd, auto_market_limit)
        # Update latest price marks on each cycle, including positions that didn't generate a sell signal.
        portfolio_value, price_errors_now = mark_positions(current_state, products, gbp_usd)
    # These metrics are inside the fragment so cash, positions, and portfolio value
    # update together every cycle rather than showing the previous full-page render.
    net_pnl = portfolio_value - float(current_state["starting_cash_gbp"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Paper portfolio", f"£{portfolio_value:.2f}", f"{net_pnl:+.2f} vs start")
    m2.metric("Available paper cash", f"£{float(current_state['cash_gbp']):.2f}")
    m3.metric("Open positions", str(len(current_state["positions"])))
    m4.metric("Score", f"{net_pnl:+.2f} points", "1 point per £1 net P&L")
    if price_errors_now:
        st.warning("Some live price requests failed; affected positions keep their last known price. " + " | ".join(price_errors_now[:4]))
    st.session_state.last_auto_activity = pd.DataFrame(activity)
    st.session_state.last_auto_errors = cycle_errors + price_errors_now
    st.caption(f"Last cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} · refresh every 30 seconds · portfolio £{portfolio_value:.2f}")
    if "last_auto_activity" in st.session_state:
        st.dataframe(st.session_state.last_auto_activity, use_container_width=True, hide_index=True)
    if st.session_state.get("last_auto_errors"):
        with st.expander("Automatic cycle errors"):
            st.code("\\n".join(st.session_state.last_auto_errors[:30]))

continuous_paper_trader()

tab_market, tab_research, tab_positions, tab_memory = st.tabs([
    "Market opportunities", "Strategy discovery lab", "Paper positions", "Experiment memory"
])

with tab_market:
    st.subheader("Market scanner")
    st.write("Looks for unusual hourly/four-hour movement and basic momentum. These are observations, not guaranteed trade signals.")
    if st.button("Scan market opportunities", type="primary") or auto_scan:
        rows = []
        with st.spinner("Fetching public Coinbase prices and candles…"):
            for _, product in products.head(max_markets).iterrows():
                pid = product["product_id"]
                try:
                    ticker = get_ticker(pid)
                    price = float(ticker["price"])
                    candles = get_candles(pid)
                    if len(candles) < 20:
                        continue
                    close = candles["close"]
                    change_1h = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else np.nan
                    change_4h = (close.iloc[-1] / close.iloc[-17] - 1) * 100 if len(close) >= 17 else np.nan
                    vol_now = float(candles["volume"].tail(4).mean())
                    vol_prior = float(candles["volume"].iloc[-24:-4].mean()) if len(candles) >= 24 else np.nan
                    volume_ratio = vol_now / vol_prior if vol_prior and vol_prior > 0 else np.nan
                    if abs(change_1h) >= 1.5 or abs(change_4h) >= 3 or (pd.notna(volume_ratio) and volume_ratio >= 1.8):
                        reason = []
                        if abs(change_1h) >= 1.5: reason.append("1h move")
                        if abs(change_4h) >= 3: reason.append("4h move")
                        if pd.notna(volume_ratio) and volume_ratio >= 1.8: reason.append("volume spike")
                        rows.append({
                            "Market": pid, "Price": price, "1h change %": round(change_1h, 2),
                            "4h change %": round(change_4h, 2), "Volume ratio": round(volume_ratio, 2) if pd.notna(volume_ratio) else None,
                            "Observation": ", ".join(reason),
                        })
                except Exception:
                    continue
                time.sleep(0.05)
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("4h change %", key=lambda s: s.abs(), ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No markets crossed the scanner thresholds, or public data was unavailable.")
    else:
        st.info("Press **Scan market opportunities** to look for unusual moves.")

with tab_research:
    st.subheader("Strategy discovery laboratory")
    st.write("The agent compares several generated rule families on recent candles, then scores the later 30% as a holdout. It can reject all candidates.")
    chosen_market = st.selectbox("Market to research", products["product_id"].tolist() if not products.empty else [])
    if st.button("Generate and test strategy ideas", type="primary", disabled=not bool(chosen_market)):
        with st.spinner(f"Testing strategy hypotheses for {chosen_market}…"):
            results = discover_strategies(chosen_market)
        if results.empty:
            st.warning("Not enough candle history returned to test this market. Try again later.")
        else:
            st.session_state.last_research = results
            for _, row in results.iterrows():
                save_experiment({"market": chosen_market, **row.to_dict()})
            results = results.copy()
            results["market"] = chosen_market
            st.dataframe(results, use_container_width=True, hide_index=True)
            eligible = results[results["verdict"] == "Candidate for paper trial"]
            if not eligible.empty:
                st.success("Candidate found for additional paper trial—not proof of future profitability.")
                st.caption("Research scores are only a screening heuristic. They do not establish future profitability.")
                if st.button("Allocate paper cash to best passing strategy", key="allocate_research_candidate"):
                    best = eligible.sort_values("research_score", ascending=False).iloc[0]
                    try:
                        product = products.loc[products["product_id"] == chosen_market].iloc[0]
                        price = float(get_ticker(chosen_market)["price"])
                        budget = min(float(state["cash_gbp"]), float(state["cash_gbp"]) * 0.40)
                        message = paper_buy(
                            state, chosen_market, product["quote"], price, gbp_usd,
                            str(best["strategy"]), allocation_gbp=budget
                        )
                        st.success(message)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not create paper position: {exc}")
            else:
                st.info("No strategy passed the current screening rules. Staying out is a valid outcome.")
    if "last_research" in st.session_state:
        st.caption("Most recent research results")
        st.dataframe(st.session_state.last_research, use_container_width=True, hide_index=True)

with tab_positions:
    st.subheader("Open paper positions")
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
                "Market": pid, "Strategy": pos["strategy"], "Entry price": pos["entry_price"],
                "Latest price": px, "Units": pos["units"], "Cost £": pos["cost_gbp"],
                "Est. net P&L £": round(pnl, 2), "Opened": pos["opened_at"],
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
        selected = st.selectbox("Position to close in paper account", list(state["positions"].keys()))
        if st.button("Close selected paper position"):
            try:
                current_price = float(get_ticker(selected)["price"])
                msg = paper_sell(state, selected, current_price, gbp_usd)
                st.success(msg)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not get a fresh price; position was not closed: {exc}")
    else:
        st.info("No open paper positions.")
    st.divider()
    st.subheader("Record a simulated buy")
    st.warning("Manual paper buys are simulated and do not send orders to Coinbase.")
    buy_market = st.selectbox("Market", products["product_id"].tolist(), key="buy_market")
    strategy_name = st.selectbox("Reason / strategy label", STRATEGIES + ["Research candidate", "Manual experiment"])
    max_alloc = max(float(state["cash_gbp"]), MIN_POSITION_GBP)
    default_alloc = min(float(state["cash_gbp"]) * 0.25, float(state["cash_gbp"]))
    allocation = st.number_input(
        "Paper cash to allocate (£)",
        min_value=MIN_POSITION_GBP,
        max_value=max_alloc,
        value=max(MIN_POSITION_GBP, default_alloc),
        step=1.0,
        help="Choose any allocation up to available cash. This does not place a real order.",
    )
    if st.button("Buy with paper funds"):
        try:
            product = products.loc[products["product_id"] == buy_market].iloc[0]
            px = float(get_ticker(buy_market)["price"])
            msg = paper_buy(state, buy_market, product["quote"], px, gbp_usd, strategy_name, allocation_gbp=allocation)
            st.success(msg)
            st.rerun()
        except Exception as exc:
            st.error(f"Paper buy could not be recorded: {exc}")

with tab_memory:
    st.subheader("Experiment memory")
    memory = load_memory()
    st.write("Research runs are stored locally in `strategy_memory.json`; the paper account is stored in `paper_state.json`.")
    if memory:
        mem_df = pd.DataFrame(memory)
        if "research_score" in mem_df.columns:
            st.dataframe(mem_df.sort_values("research_score", ascending=False, na_position="last").head(100), use_container_width=True, hide_index=True)
        else:
            st.dataframe(mem_df.tail(100), use_container_width=True, hide_index=True)
    else:
        st.info("No experiments saved yet. Run the strategy discovery lab to start building research history.")
    st.caption(f"Closed paper trades recorded: {len(state['closed_trades'])} · Realised P&L: £{realised:+.2f} · Estimated unrealised movement: £{unrealised:+.2f}")

st.divider()
st.caption(
    "Important: this is an experimental research and paper-trading tool, not financial advice. "
    "The strategy search is limited to programmed rule families; it does not guarantee profit. "
    "Historical candles, API availability, fees, slippage, spread, and execution assumptions can make results misleading. "
    "Paper fills are simplified and should not be treated as equivalent to real trading."
)
