"""
V26.2 Strategy Backtest — Coiled EMA Breakout + Micro Fib Grid
Mirrors live_mt5_bot.py signal logic, risk, and partial-exit rules.
Uses Yahoo Finance GC=F (COMEX gold futures) as XAUUSD proxy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ─── STRATEGY CONSTANTS (match live_mt5_bot.py) ─────────────────────────────
STARTING_CAP = 500.0
MARGIN_PER_TRADE = 200.0
LEVERAGE = 50
TRADE_POWER = MARGIN_PER_TRADE * LEVERAGE  # $10,000

RISK_PERCENTAGE = 0.02
MAX_RISK_PER_TRADE = 20.0
PARTIAL_CLOSE_FRACTION = 0.5
FINAL_RISK_REWARD = 2.0

HARD_DD_PCT = 0.15
HALT_EQUITY = STARTING_CAP * (1 - HARD_DD_PCT)

ATR_MIN_RATIO = 0.70
COILED_EMA_BLOCKED = {"London (08-13 UTC)", "NY Only (17-22 UTC)"}

OUT_DIR = Path("backtest_results")


def get_session(hour: int) -> str:
    if 0 <= hour < 8:
        return "Asian (00-08 UTC)"
    if 8 <= hour < 13:
        return "London (08-13 UTC)"
    if 13 <= hour < 17:
        return "NY Overlap (13-17 UTC)"
    if 17 <= hour < 22:
        return "NY Only (17-22 UTC)"
    return "Late / Sydney (22-00 UTC)"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 15m + daily gold futures bars (UTC)."""
    raw_15m = yf.download(
        "GC=F", interval="15m", period="60d", progress=False, auto_adjust=True
    )
    raw_1d = yf.download(
        "GC=F", interval="1d", period="1y", progress=False, auto_adjust=True
    )
    if isinstance(raw_15m.columns, pd.MultiIndex):
        raw_15m.columns = raw_15m.columns.get_level_values(0)
    if isinstance(raw_1d.columns, pd.MultiIndex):
        raw_1d.columns = raw_1d.columns.get_level_values(0)

    df_15m = raw_15m.rename(
        columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close"}
    )[["Open", "High", "Low", "Close"]].dropna()
    df_1d = raw_1d.rename(
        columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close"}
    )[["Open", "High", "Low", "Close"]].dropna()

    # Normalize to UTC tz-naive for joins
    if df_15m.index.tz is not None:
        df_15m.index = df_15m.index.tz_convert("UTC").tz_localize(None)
    if df_1d.index.tz is not None:
        df_1d.index = df_1d.index.tz_convert("UTC").tz_localize(None)

    return df_15m, df_1d


def compute_indicators(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    df["EMA_25"] = df["Close"].ewm(span=25, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50)

    hl = df["High"] - df["Low"]
    hcp = (df["High"] - df["Close"].shift(1)).abs()
    lcp = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    df["ATR_14"] = tr.ewm(com=13, adjust=False).mean()
    df["ATR_MA20"] = df["ATR_14"].rolling(20).mean()
    df["ATR_ratio"] = (df["ATR_14"] / df["ATR_MA20"]).fillna(1.0)
    return df


def compute_daily_levels(df_1d: pd.DataFrame) -> pd.DataFrame:
    df = df_1d.copy()
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Range"] = df["Prev_High"] - df["Prev_Low"]
    df["Prev_Green"] = df["Close"].shift(1) > df["Open"].shift(1)

    df["Fib_1618_Bull"] = df["Prev_High"] + (1.618 * df["Prev_Range"])
    df["Fib_0382_Bull"] = df["Prev_High"] - (0.382 * df["Prev_Range"])
    df["Fib_0618_Bull"] = df["Prev_High"] - (0.618 * df["Prev_Range"])

    df["Fib_1618_Bear"] = df["Prev_Low"] - (1.618 * df["Prev_Range"])
    df["Fib_0382_Bear"] = df["Prev_Low"] + (0.382 * df["Prev_Range"])
    df["Fib_0618_Bear"] = df["Prev_Low"] + (0.618 * df["Prev_Range"])
    return df


def signal_on_bar(df_15m: pd.DataFrame, daily_row: pd.Series, i: int) -> dict | None:
    """Evaluate entry on completed bar i (same conditions as live bot idx=-2)."""
    row = df_15m.iloc[i]
    c1 = float(row["Close"])
    c2 = float(df_15m["Close"].iloc[i - 1])
    ema_25 = float(row["EMA_25"])
    ema_50 = float(row["EMA_50"])
    ema_200 = float(row["EMA_200"])
    rsi = float(row["RSI_14"])
    atr_ratio = float(row["ATR_ratio"])

    if np.isnan(atr_ratio) or atr_ratio < ATR_MIN_RATIO:
        return None
    if np.isnan(ema_200) or np.isnan(daily_row["Prev_High"]):
        return None

    session = get_session(df_15m.index[i].hour)
    ema_bull = ema_25 > ema_50
    ema_bear = ema_25 < ema_50
    ema_coil = abs(ema_25 - ema_50) <= 0.50
    bullish_stack = ema_25 > ema_50 and ema_50 > ema_200
    bearish_stack = ema_25 < ema_50 and ema_50 < ema_200

    is_green = bool(daily_row["Prev_Green"])
    prev_high = float(daily_row["Prev_High"])
    prev_low = float(daily_row["Prev_Low"])
    fib_1618_bull = float(daily_row["Fib_1618_Bull"])
    fib_0382_bull = float(daily_row["Fib_0382_Bull"])
    fib_0618_bull = float(daily_row["Fib_0618_Bull"])
    fib_1618_bear = float(daily_row["Fib_1618_Bear"])
    fib_0382_bear = float(daily_row["Fib_0382_Bear"])
    fib_0618_bear = float(daily_row["Fib_0618_Bear"])

    direction = ""
    setup_name = ""

    if ema_coil and session not in COILED_EMA_BLOCKED:
        if bullish_stack and (c2 > ema_25) and (c1 > ema_25) and rsi > 50:
            direction, setup_name = "LONG", "Coiled_EMA_Breakout"
        elif bearish_stack and (c2 < ema_25) and (c1 < ema_25) and rsi < 50:
            direction, setup_name = "SHORT", "Coiled_EMA_Breakout"

    if not direction:
        bull_levels = [
            (fib_1618_bull, "1D_Fib_1.618"),
            (prev_high, "1D_Fib_1.0"),
            (fib_0382_bull, "1D_Fib_0.382"),
            (fib_0618_bull, "1D_Fib_0.618"),
        ]
        bear_levels = [
            (fib_1618_bear, "1D_Fib_1.618"),
            (prev_low, "1D_Fib_1.0"),
            (fib_0382_bear, "1D_Fib_0.382"),
            (fib_0618_bear, "1D_Fib_0.618"),
        ]
        if is_green:
            for level, name in bull_levels:
                if c2 > level and c1 > level and ema_bull and rsi > 50:
                    direction, setup_name = "LONG", name
                    break
        else:
            for level, name in bear_levels:
                if c2 < level and c1 < level and ema_bear and rsi < 50:
                    direction, setup_name = "SHORT", name
                    break

        if not direction:
            if is_green:
                for level, name in bull_levels:
                    if c2 < level and c1 < level and ema_bear and rsi < 50:
                        direction, setup_name = "SHORT", name + " (Correction)"
                        break
            else:
                for level, name in bear_levels:
                    if c2 > level and c1 > level and ema_bull and rsi > 50:
                        direction, setup_name = "LONG", name + " (Correction)"
                        break

    if not direction:
        return None

    entry = c1  # closed-bar fill approximation (live uses live tick)
    return {
        "direction": direction,
        "entry": entry,
        "setup_name": setup_name,
        "session": session,
        "signal_time": df_15m.index[i],
    }


@dataclass
class Trade:
    setup: str
    direction: str
    session: str
    entry_time: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    qty: float
    risk: float
    exit_time: str
    exit_price: float
    exit_reason: str
    pnl: float
    equity_after: float
    bars_held: int
    tp1_hit: bool


def run_backtest(df_15m: pd.DataFrame, df_1d: pd.DataFrame) -> tuple[list[Trade], dict]:
    df_15m = compute_indicators(df_15m)
    df_1d = compute_daily_levels(df_1d)

    # Map each 15m bar to the latest completed daily row (prior calendar day levels)
    daily_dates = df_1d.index.normalize()
    df_1d = df_1d.copy()
    df_1d["_date"] = daily_dates

    capital = STARTING_CAP
    peak = STARTING_CAP
    max_dd = 0.0
    equity_curve = []
    trades: list[Trade] = []

    i = 200  # warm-up for EMA200
    n = len(df_15m)

    while i < n - 1:
        ts = df_15m.index[i]
        equity_curve.append({"time": str(ts), "equity": capital})

        if capital < HALT_EQUITY:
            i += 1
            continue

        # Use most recent completed daily bar prior to this 15m bar's date
        bar_date = ts.normalize()
        prior_daily = df_1d[df_1d["_date"] < bar_date]
        if prior_daily.empty:
            i += 1
            continue
        daily_row = prior_daily.iloc[-1]

        sig = signal_on_bar(df_15m, daily_row, i)
        if not sig:
            i += 1
            continue

        entry = float(sig["entry"])
        direction = sig["direction"]
        risk = min(MAX_RISK_PER_TRADE, capital * RISK_PERCENTAGE)
        qty = TRADE_POWER / entry
        sl_dist = risk / qty
        tp1_dist = sl_dist
        tp2_dist = sl_dist * FINAL_RISK_REWARD

        if direction == "LONG":
            sl = entry - sl_dist
            tp1 = entry + tp1_dist
            tp2 = entry + tp2_dist
        else:
            sl = entry + sl_dist
            tp1 = entry - tp1_dist
            tp2 = entry - tp2_dist

        # Simulate forward bar-by-bar from next bar
        tp1_hit = False
        qty_remaining = qty
        realized = 0.0
        sl_price = sl
        exit_price = entry
        exit_reason = "EOD"
        exit_time = df_15m.index[-1]
        bars_held = 0
        j = i + 1

        while j < n:
            bar = df_15m.iloc[j]
            high, low = float(bar["High"]), float(bar["Low"])
            bars_held += 1

            if direction == "LONG":
                # Conservative: if SL and TP both touch same bar, count SL first
                if (not tp1_hit) and low <= sl_price:
                    pnl = (sl_price - entry) * qty_remaining
                    realized += pnl
                    exit_price, exit_reason, exit_time = sl_price, "Hard SL", df_15m.index[j]
                    qty_remaining = 0
                    break
                if (not tp1_hit) and high >= tp1:
                    close_qty = qty * PARTIAL_CLOSE_FRACTION
                    realized += (tp1 - entry) * close_qty
                    qty_remaining -= close_qty
                    tp1_hit = True
                    sl_price = entry
                    # continue same bar for possible TP2 / BE
                if tp1_hit and low <= sl_price:
                    realized += (sl_price - entry) * qty_remaining
                    exit_price, exit_reason, exit_time = sl_price, "BE Stop", df_15m.index[j]
                    qty_remaining = 0
                    break
                if high >= tp2:
                    realized += (tp2 - entry) * qty_remaining
                    exit_price, exit_reason, exit_time = tp2, "TP2", df_15m.index[j]
                    qty_remaining = 0
                    break
            else:  # SHORT
                if (not tp1_hit) and high >= sl_price:
                    pnl = (entry - sl_price) * qty_remaining
                    realized += pnl
                    exit_price, exit_reason, exit_time = sl_price, "Hard SL", df_15m.index[j]
                    qty_remaining = 0
                    break
                if (not tp1_hit) and low <= tp1:
                    close_qty = qty * PARTIAL_CLOSE_FRACTION
                    realized += (entry - tp1) * close_qty
                    qty_remaining -= close_qty
                    tp1_hit = True
                    sl_price = entry
                if tp1_hit and high >= sl_price:
                    realized += (entry - sl_price) * qty_remaining
                    exit_price, exit_reason, exit_time = sl_price, "BE Stop", df_15m.index[j]
                    qty_remaining = 0
                    break
                if low <= tp2:
                    realized += (entry - tp2) * qty_remaining
                    exit_price, exit_reason, exit_time = tp2, "TP2", df_15m.index[j]
                    qty_remaining = 0
                    break
            j += 1

        if qty_remaining > 0:
            # Force flat at last available close
            last_close = float(df_15m.iloc[-1]["Close"])
            if direction == "LONG":
                realized += (last_close - entry) * qty_remaining
            else:
                realized += (entry - last_close) * qty_remaining
            exit_price, exit_reason, exit_time = last_close, "Data End", df_15m.index[-1]
            qty_remaining = 0

        capital += realized
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0.0)

        trades.append(
            Trade(
                setup=sig["setup_name"],
                direction=direction,
                session=sig["session"],
                entry_time=str(sig["signal_time"]),
                entry=round(entry, 2),
                sl=round(sl, 2),
                tp1=round(tp1, 2),
                tp2=round(tp2, 2),
                qty=round(qty, 4),
                risk=round(risk, 2),
                exit_time=str(exit_time),
                exit_price=round(exit_price, 2),
                exit_reason=exit_reason,
                pnl=round(realized, 2),
                equity_after=round(capital, 2),
                bars_held=bars_held,
                tp1_hit=tp1_hit,
            )
        )

        # Resume after exit bar (one position at a time, like live bot)
        i = j + 1 if j < n else n

    stats = summarize(trades, capital, max_dd, df_15m)
    return trades, {"stats": stats, "equity_curve": equity_curve}


def summarize(trades: list[Trade], final_equity: float, max_dd: float, df: pd.DataFrame) -> dict:
    if not trades:
        return {
            "trades": 0,
            "final_equity": final_equity,
            "net_pnl": final_equity - STARTING_CAP,
            "return_pct": 0.0,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "period_start": str(df.index[0]),
            "period_end": str(df.index[-1]),
            "bars": len(df),
            "symbol": "GC=F (gold futures proxy for XAUUSD)",
            "timeframe": "15m",
        }

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    by_setup: dict[str, dict] = {}
    by_reason: dict[str, int] = {}
    by_dir: dict[str, dict] = {}

    for t in trades:
        s = by_setup.setdefault(t.setup, {"n": 0, "pnl": 0.0, "wins": 0})
        s["n"] += 1
        s["pnl"] += t.pnl
        s["wins"] += int(t.pnl > 0)
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        d = by_dir.setdefault(t.direction, {"n": 0, "pnl": 0.0, "wins": 0})
        d["n"] += 1
        d["pnl"] += t.pnl
        d["wins"] += int(t.pnl > 0)

    for s in by_setup.values():
        s["pnl"] = round(s["pnl"], 2)
        s["win_rate"] = round(100 * s["wins"] / s["n"], 1) if s["n"] else 0
    for d in by_dir.values():
        d["pnl"] = round(d["pnl"], 2)
        d["win_rate"] = round(100 * d["wins"] / d["n"], 1) if d["n"] else 0

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "symbol": "GC=F (gold futures proxy for XAUUSD)",
        "timeframe": "15m",
        "period_start": str(df.index[0]),
        "period_end": str(df.index[-1]),
        "bars": len(df),
        "starting_capital": STARTING_CAP,
        "final_equity": round(final_equity, 2),
        "net_pnl": round(final_equity - STARTING_CAP, 2),
        "return_pct": round(100 * (final_equity - STARTING_CAP) / STARTING_CAP, 2),
        "trades": len(trades),
        "wins": int((pnls > 0).sum()),
        "losses": int((pnls <= 0).sum()),
        "win_rate_pct": round(100 * float((pnls > 0).mean()), 1),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "largest_win": round(float(pnls.max()), 2),
        "largest_loss": round(float(pnls.min()), 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "expectancy": round(float(pnls.mean()), 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 1),
        "tp1_hit_rate_pct": round(100 * float(np.mean([t.tp1_hit for t in trades])), 1),
        "halt_equity": HALT_EQUITY,
        "hit_hard_dd_halt": final_equity < HALT_EQUITY,
        "by_setup": by_setup,
        "by_direction": by_dir,
        "by_exit_reason": by_reason,
        "assumptions": [
            "Data: Yahoo Finance GC=F 15m (~60d available), not broker XAUUSD spot",
            "Entry fill: signal bar close (live bot uses live bid/ask)",
            "Intrabar: if SL and TP1 both touch same bar before TP1 hit, SL wins (conservative)",
            "One position at a time; no overnight gap model beyond bar OHLC",
            "No spreads/commissions modeled",
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("Downloading GC=F data...")
    df_15m, df_1d = load_data()
    print(f"15m bars: {len(df_15m)} | {df_15m.index[0]} → {df_15m.index[-1]}")
    print(f"1d bars:  {len(df_1d)} | {df_1d.index[0]} → {df_1d.index[-1]}")

    trades, payload = run_backtest(df_15m, df_1d)
    stats = payload["stats"]

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    trades_path = OUT_DIR / "trades.csv"
    stats_path = OUT_DIR / "summary.json"
    equity_path = OUT_DIR / "equity_curve.csv"

    trades_df.to_csv(trades_path, index=False)
    pd.DataFrame(payload["equity_curve"]).to_csv(equity_path, index=False)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n========== V26.2 BACKTEST RESULTS ==========")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {trades_path}, {stats_path}, {equity_path}")


if __name__ == "__main__":
    main()
