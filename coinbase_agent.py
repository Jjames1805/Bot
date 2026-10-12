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
        price = float(d.loc[i, "open"])
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
    for column in ["strategy", "train_return_pct", "test_return_pct", "test_trades",
                   "test_win_rate_pct", "test_max_drawdown_pct", "research_score", "verdict"]:
        if column not in result.columns:
            result[column] = np.nan
    return result.sort_values("research_score", ascending=False, na_position="last").reset_index(drop=True)

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

def latest_strategy_action(frame: pd.DataFrame, strategy: str) -> tuple[str, str]:
    if frame.empty or len(frame) < 80:
        return "WAIT", "Not enough candle history"
    try:
        signals = strategy_signal(frame, strategy)
        signal = int(signals.iloc[-2])
        if signal == 1:
            return "BUY", "Buy signal on latest completed candle"
        if signal == -1:
            return "SELL", "Sell signal on latest completed candle"
        return "WAIT", "No entry/exit signal on latest completed candle"
    except Exception as exc:
        return "WAIT", f"Signal calculation failed: {type(exc).__name__}: {exc}"

def auto_paper_cycle(state: dict, products: pd.DataFrame, gbp_usd: float, market_limit: int = 10):
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
                for strategy in STRATEGIES:
                    try:
                        result = split_test(frame, strategy)
                        result["strategy"] = strategy
                        tested.append(result)
                    except Exception as exc:
                        errors.append(f"{product_id}/{strategy}: {type(exc).__name__}: {exc}")
                if not tested:
                    activity.append({"Market": product_id, "Action": "SKIP", "Reason": "All strategy tests failed", "Details": ""})
                    continue
                df = pd.DataFrame(tested)
                eligible = df.copy()
                eligible["research_score"] = pd.to_numeric(eligible["research_score"], errors="coerce").fillna(-999)
                eligible = eligible.sort_values("research_score", ascending=False)
                active_candidates = []
                signal_reasons = []
                for _, candidate in eligible.iterrows():
                    candidate_strategy = str(candidate["strategy"])
                    candidate_action, candidate_reason = latest_strategy_action(frame, candidate_strategy)
                    signal_reasons.append(f"{candidate_strategy}: {candidate_action}")
                    if candidate_action == "BUY":
                        active_candidates.append((candidate, candidate_strategy, candidate_reason))
                if not active_candidates:
                    try:
                        indicators = add_indicators(frame)
                        last = indicators.iloc[-2]
                        if (pd.notna(last["ema_fast"]) and pd.notna(last["ema_slow"]) and pd.notna(last["rsi"])
                            and last["close"] >= last["ema_slow"] and last["ema_fast"] >= last["ema_slow"]
                            and 30 <= float(last["rsi"]) <= 70):
                            fallback_row = eligible[eligible["strategy"] == "Trend pullback"]
                            best = fallback_row.iloc[0] if not fallback_row.empty else eligible.iloc[0]
                            strategy = "Trend pullback"
                            reason = "Fallback entry: completed candle confirms uptrend and RSI is 30-70"
                            active_candidates.append((best, strategy, reason))
                    except Exception as exc:
                        errors.append(f"{product_id}/fallback: {type(exc).__name__}: {exc}")
                if not active_candidates:
                    activity.append({
                        "Market": product_id, "Action": "WAIT",
                        "Reason": "No current buy signal",
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
                    activity.append({"Market": product_id, "Action": "SKIP", "Reason": "Allocation too small", "Details": ""})
                    continue
                price = float(get_ticker(product_id)["price"])
                message = paper_buy(state, product_id, str(product["quote"]), price, gbp_usd, strategy, allocation_gbp=allocation)
                activity.append({"Market": product_id, "Action": "PAPER BUY", "Reason": reason, "Details": f"Strategy: {strategy}. {message}"})
            except Exception as exc:
                errors.append(f"{product_id}: {type(exc).__name__}: {exc}")
    save_state(state)
    if not activity:
        activity.append({"Market": "—", "Action": "NO ACTION", "Reason": "No signal", "Details": ""})
    return activity, errors

st.set_page_config(page_title="Coinbase Paper Research Agent", page_icon="🧪", layout="wide")
st.title("🧪 Coinbase Self-Improving Paper Trading Agent")
st.caption("Public market data • GBP/USD markets • Simulated trades only • No live orders")

if "state" not in st.session_state:
    st.session_state.state = load_state()
state = st.session_state.state

auto_scan = True
max_markets = 30
auto_market_limit = 30

st.caption("Continuous paper trading is always enabled while this app is open.")

try:
    products = get_products()
    gbp_usd = get_gbp_usd_rate()
    st.caption(f"Markets found: {len(products)} · GBP/USD rate: £1 ≈ ${gbp_usd:.4f}")
except Exception as exc:
    st.error(f"Could not load Coinbase public market data: {exc}")
    st.stop()

st.subheader("Continuous automatic paper trading")
st.write("Checks strategies across markets and manages paper positions every 30 seconds.")

@st.fragment(run_every=30)
def continuous_paper_trader():
    current_state = st.session_state.state
    with st.spinner("Running automatic paper-trading cycle…"):
        activity, cycle_errors = auto_paper_cycle(current_state, products, gbp_usd, auto_market_limit)
        portfolio_value, price_errors_now = mark_positions(current_state, products, gbp_usd)
    net_pnl = portfolio_value - float(current_state["starting_cash_gbp"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Paper portfolio", f"£{portfolio_value:.2f}", f"{net_pnl:+.2f} vs start")
    m2.metric("Available paper cash", f"£{float(current_state['cash_gbp']):.2f}")
    m3.metric("Open positions", str(len(current_state["positions"])))
    m4.metric("Score", f"{net_pnl:+.2f} points", "1 point per £1 net P&L")
    if price_errors_now:
        st.warning("Some price requests failed. " + " | ".join(price_errors_now[:4]))
    st.session_state.last_auto_activity = pd.DataFrame(activity)
    st.session_state.last_auto_errors = cycle_errors + price_errors_now
    st.caption(f"Last cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if "last_auto_activity" in st.session_state:
        st.dataframe(st.session_state.last_auto_activity, use_container_width=True, hide_index=True)
    if st.session_state.get("last_auto_errors"):
        with st.expander("Automatic cycle errors"):
            st.code("\n".join(st.session_state.last_auto_errors[:30]))

continuous_paper_trader()

tab_market, tab_research, tab_positions, tab_memory = st.tabs([
    "Market opportunities", "Strategy discovery lab", "Paper positions", "Experiment memory"
])

with tab_market:
    st.subheader("Market scanner")
    if st.button("Scan market opportunities", type="primary") or auto_scan:
        rows = []
        with st.spinner("Fetching public Coinbase prices…"):
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
                        if abs(change_1h) >= 1.5:
                            reason.append("1h move")
                        if abs(change_4h) >= 3:
                            reason.append("4h move")
                        if pd.notna(volume_ratio) and volume_ratio >= 1.8:
                            reason.append("volume spike")
                        rows.append({
                            "Market": pid,
                            "Price": price,
                            "1h change %": round(change_1h, 2),
                            "4h change %": round(change_4h, 2),
                            "Volume ratio": round(volume_ratio, 2) if pd.notna(volume_ratio) else None,
                            "Observation": ", ".join(reason),
                        })
                except Exception:
                    continue
                time.sleep(0.05)
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("4h change %", key=lambda s: s.abs(), ascending=False),
                       use_container_width=True, hide_index=True)
        else:
            st.info("No markets crossed the scanner thresholds.")
    else:
        st.info("Press Scan market opportunities.")

with tab_research:
    st.subheader("Strategy discovery laboratory")
    chosen_market = st.selectbox("Market to research", products["product_id"].tolist() if not products.empty else [])
    if st.button("Generate and test strategy ideas", type="primary", disabled=not bool(chosen_market)):
        with st.spinner(f"Testing strategies for {chosen_market}…"):
            results = discover_strategies(chosen_market)
        if results.empty:
            st.warning("Not enough candle history.")
        else:
            st.session_state.last_research = results
            for _, row in results.iterrows():
                save_experiment({"market": chosen_market, **row.to_dict()})
            results = results.copy()
            results["market"] = chosen_market
            st.dataframe(results, use_container_width=True, hide_index=True)
            eligible = results[results["verdict"] == "Candidate for paper trial"]
            if not eligible.empty:
                st.success("Candidate found for paper trial.")
                if st.button("Allocate paper cash to best strategy", key="allocate_research_candidate"):
                    best = eligible.sort_values("research_score", ascending=False).iloc[0]
                    try:
                        product = products.loc[products["product_id"] == chosen_market].iloc[0]
                        price = float(get_ticker(chosen_market)["price"])
                        budget = min(float(state["cash_gbp"]), float(state["cash_gbp"]) * 0.40)
                        message = paper_buy(state, chosen_market, product["quote"], price, gbp_usd,
                                            str(best["strategy"]), allocation_gbp=budget)
                        st.success(message)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not create paper position: {exc}")
            else:
                st.info("No strategy passed the screening rules.")
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
                "Market": pid,
                "Strategy": pos["strategy"],
                "Entry price": pos["entry_price"],
                "Latest price": px,
                "Units": pos["units"],
                "Cost £": pos["cost_gbp"],
                "Est. net P&L £": round(pnl, 2),
                "Opened": pos["opened_at"],
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
        selected = st.selectbox("Position to close", list(state["positions"].keys()))
        if st.button("Close selected paper position"):
            try:
                current_price = float(get_ticker(selected)["price"])
                msg = paper_sell(state, selected, current_price, gbp_usd)
                st.success(msg)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not close position: {exc}")
    else:
        st.info("No open paper positions.")
    st.divider()
    st.subheader("Record a simulated buy")
    st.warning("This is paper trading only – no real orders are sent.")
    buy_market = st.selectbox("Market", products["product_id"].tolist(), key="buy_market")
    strategy_name = st.selectbox("Strategy label", STRATEGIES + ["Research candidate", "Manual experiment"])
    max_alloc = max(float(state["cash_gbp"]), MIN_POSITION_GBP)
    default_alloc = min(float(state["cash_gbp"]) * 0.25, float(state["cash_gbp"]))
    allocation = st.number_input(
        "Paper cash to allocate (£)",
        min_value=MIN_POSITION_GBP,
        max_value=max_alloc,
        value=max(MIN_POSITION_GBP, default_alloc),
        step=1.0,
    )
    if st.button("Buy with paper funds"):
        try:
            product = products.loc[products["product_id"] == buy_market].iloc[0]
            px = float(get_ticker(buy_market)["price"])
            msg = paper_buy(state, buy_market, product["quote"], px, gbp_usd, strategy_name, allocation_gbp=allocation)
            st.success(msg)
            st.rerun()
        except Exception as exc:
            st.error(f"Paper buy failed: {exc}")

with tab_memory:
    st.subheader("Experiment memory")
    memory = load_memory()
    st.write("Research history is stored in strategy_memory.json")
    if memory:
        mem_df = pd.DataFrame(memory)
        if "research_score" in mem_df.columns:
            st.dataframe(mem_df.sort_values("research_score", ascending=False, na_position="last").head(100),
                       use_container_width=True, hide_index=True)
        else:
            st.dataframe(mem_df.tail(100), use_container_width=True, hide_index=True)
    else:
        st.info("No experiments saved yet.")

    realised = sum(float(t.get("pnl_gbp", 0)) for t in state.get("closed_trades", []))
    unrealised = 0.0
    for pid, pos in state.get("positions", {}).items():
        px = float(pos.get("last_price", pos["entry_price"]))
        current_gbp = to_gbp(float(pos["units"]) * px * (1 - FEE_RATE), pos["quote"], gbp_usd)
        unrealised += current_gbp - float(pos["cost_gbp"])

    st.caption(
        f"Closed trades: {len(state['closed_trades'])} · "
        f"Realised P&L: £{realised:+.2f} · Unrealised: £{unrealised:+.2f}"
    )

st.divider()
st.caption(
    "This is an experimental paper-trading tool only. "
    "It does not place real orders and is not financial advice."
)
