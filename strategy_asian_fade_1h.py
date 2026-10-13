#!/usr/bin/env python3
"""
PRIMARY high-winrate strategy: Asian-Range Fade on 1h gold.

Backtested ~91% winrate over 2024-2026 on GC=F (see search_results/STRATEGY.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START_CAP = 500.0
RISK_PCT = 0.02
MAX_RISK = 20.0
HALT = START_CAP * 0.85
LOOK = 40
RSI_LEN = 7
OS, OB = 25, 70
SL_ATR, TP_ATR = 2.0, 0.3
MAX_HOLD = 12
OUT = Path("search_results")


def load_1h() -> pd.DataFrame:
    df = yf.download("GC=F", interval="1h", period="730d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    delta = c.diff()
    ag = delta.clip(lower=0).ewm(com=RSI_LEN - 1, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=RSI_LEN - 1, adjust=False).mean()
    d["RSI"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - c.shift(1)).abs(),
            (d["Low"] - c.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.ewm(com=13, adjust=False).mean()
    d["range_hi"] = d["High"].rolling(LOOK).max().shift(1)
    d["range_lo"] = d["Low"].rolling(LOOK).min().shift(1)
    d["hour"] = d.index.hour
    d["overlap"] = (d["hour"] >= 13) & (d["hour"] < 17)
    # Require wick into extreme + close back inside range (matches strategy_search gen_signals)
    d["long"] = (
        d["overlap"]
        & (d["Low"] <= d["range_lo"])
        & (c > d["range_lo"])
        & (d["RSI"] < OS)
    )
    d["short"] = (
        d["overlap"]
        & (d["High"] >= d["range_hi"])
        & (c < d["range_hi"])
        & (d["RSI"] > OB)
    )
    return d


def backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Strict causality: signal on bar i close → fill at open[i+1]."""
    opens = df["Open"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    atrs = df["ATR"].to_numpy(float)
    long_e = df["long"].fillna(False).to_numpy()
    short_e = df["short"].fillna(False).to_numpy()
    capital = START_CAP
    peak = START_CAP
    max_dd = 0.0
    trades = []
    i = 250
    n = len(df)
    while i < n - 2:
        if capital < HALT:
            break
        gl, gs = bool(long_e[i]), bool(short_e[i])
        if gl == gs:
            i += 1
            continue
        direction = "LONG" if gl else "SHORT"
        atr = atrs[i]  # known at signal close
        entry = opens[i + 1]  # earliest realistic fill
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
            i += 1
            continue
        risk = min(MAX_RISK, capital * RISK_PCT)
        sl_dist, tp_dist = atr * SL_ATR, atr * TP_ATR
        qty = risk / sl_dist
        sl = entry - sl_dist if direction == "LONG" else entry + sl_dist
        tp = entry + tp_dist if direction == "LONG" else entry - tp_dist
        j, bars, exit_price, reason, hit = i + 1, 0, entry, "time", False
        while j < n and bars < MAX_HOLD:
            hi, lo = highs[j], lows[j]
            bars += 1
            if direction == "LONG":
                if lo <= sl:
                    exit_price, reason, hit = sl, "SL", True
                    break
                if hi >= tp:
                    exit_price, reason, hit = tp, "TP", True
                    break
            else:
                if hi >= sl:
                    exit_price, reason, hit = sl, "SL", True
                    break
                if lo <= tp:
                    exit_price, reason, hit = tp, "TP", True
                    break
            j += 1
        if not hit:
            exit_price = closes[min(j, n - 1)]
        pnl = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0)
        trades.append(
            {
                "entry_time": str(df.index[i]),
                "direction": direction,
                "entry": round(float(entry), 2),
                "sl": round(float(sl), 2),
                "tp": round(float(tp), 2),
                "exit": round(float(exit_price), 2),
                "reason": reason,
                "bars": bars,
                "pnl": round(float(pnl), 2),
                "equity": round(float(capital), 2),
            }
        )
        i = j + 1 if j > i else i + 1

    tdf = pd.DataFrame(trades)
    arr = tdf["pnl"].to_numpy()
    wins, losses = arr[arr > 0], arr[arr <= 0]
    gw, gl = float(wins.sum()) if len(wins) else 0.0, float(abs(losses.sum())) if len(losses) else 0.0
    summary = {
        "strategy": "Asian-Range Fade 1h",
        "period": f"{df.index[0]} → {df.index[-1]}",
        "trades": int(len(arr)),
        "winrate_pct": round(float((arr > 0).mean()) * 100, 2),
        "profit_factor": round(gw / gl, 3) if gl else None,
        "net_pnl": round(float(arr.sum()), 2),
        "final_equity": round(float(capital), 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "tp_exits": int((tdf["reason"] == "TP").sum()),
        "sl_exits": int((tdf["reason"] == "SL").sum()),
    }
    return tdf, summary


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = prepare(load_1h())
    trades, summary = backtest(df)
    trades.to_csv(OUT / "primary_asian_fade_1h_trades.csv", index=False)
    (OUT / "primary_asian_fade_1h_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
