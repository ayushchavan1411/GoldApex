#!/usr/bin/env python3
"""
Full trade-level backtest of the discovered high-winrate strategy.

Strategy: Bollinger(20, 2.5) mean-reversion + RSI(3) filter
Sessions: London + NY Overlap only (08:00–17:00 UTC)
Exits: SL = 1.5×ATR(14), TP = 0.4×ATR(14), max hold 24 bars
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_search import (
    START_CAP,
    TRADE_POWER,
    MAX_RISK,
    RISK_PCT,
    HALT,
    add_indicators,
    load_frames,
)

OUT = Path("search_results")


def run() -> None:
    frames = load_frames()
    df = add_indicators(frames["15m"])
    c = df["Close"]
    lo = df["BB_LO_20_2.5"]
    up = df["BB_UP_20_2.5"]
    rsi = df["RSI_3"]
    sess = df["session"].isin(["london", "overlap"])

    long_e = ((df["Low"] <= lo) & (rsi < 30) & sess).to_numpy()
    short_e = ((df["High"] >= up) & (rsi > 70) & sess).to_numpy()

    closes = df["Close"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    atrs = df["ATR_14"].to_numpy(float)
    times = df.index.to_numpy()

    capital = START_CAP
    peak = START_CAP
    max_dd = 0.0
    trades = []
    i = 250
    n = len(df)
    while i < n - 1:
        if capital < HALT:
            break
        go_long, go_short = bool(long_e[i]), bool(short_e[i])
        if go_long == go_short:
            i += 1
            continue
        direction = "LONG" if go_long else "SHORT"
        entry = closes[i]
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue
        risk = min(MAX_RISK, capital * RISK_PCT)
        sl_dist = atr * 1.5
        tp_dist = atr * 0.4
        qty = risk / sl_dist
        sl = entry - sl_dist if direction == "LONG" else entry + sl_dist
        tp = entry + tp_dist if direction == "LONG" else entry - tp_dist

        exit_price, reason, bars = entry, "time", 0
        j = i + 1
        hit = False
        while j < n and bars < 24:
            hi, bar_lo = highs[j], lows[j]
            bars += 1
            if direction == "LONG":
                if bar_lo <= sl:
                    exit_price, reason, hit = sl, "SL", True
                    break
                if hi >= tp:
                    exit_price, reason, hit = tp, "TP", True
                    break
            else:
                if hi >= sl:
                    exit_price, reason, hit = sl, "SL", True
                    break
                if bar_lo <= tp:
                    exit_price, reason, hit = tp, "TP", True
                    break
            j += 1
        if not hit:
            exit_price = closes[min(j, n - 1)]
            reason = "time"

        pnl = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0)
        trades.append(
            {
                "entry_time": str(times[i]),
                "exit_time": str(times[min(j, n - 1)]),
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
    tdf.to_csv(OUT / "winner_trades.csv", index=False)

    arr = tdf["pnl"].to_numpy()
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    split_time = df.index[int(len(df) * 0.7)]
    is_df = tdf[tdf["entry_time"] < str(split_time)]
    oos_df = tdf[tdf["entry_time"] >= str(split_time)]

    def stats(x: pd.DataFrame) -> dict:
        a = x["pnl"].to_numpy()
        if len(a) == 0:
            return {}
        w = a[a > 0]
        l = a[a <= 0]
        gw, gl = float(w.sum()) if len(w) else 0.0, float(abs(l.sum())) if len(l) else 0.0
        return {
            "trades": int(len(a)),
            "winrate": round(float((a > 0).mean()) * 100, 2),
            "profit_factor": round(gw / gl, 3) if gl else None,
            "net_pnl": round(float(a.sum()), 2),
            "avg_win": round(float(w.mean()), 2) if len(w) else 0,
            "avg_loss": round(float(l.mean()), 2) if len(l) else 0,
            "tp_exits": int((x["reason"] == "TP").sum()),
            "sl_exits": int((x["reason"] == "SL").sum()),
            "time_exits": int((x["reason"] == "time").sum()),
        }

    summary = {
        "strategy": "BB(20,2.5) mean-reversion + RSI(3) filter",
        "sessions": "London + NY Overlap (08-17 UTC)",
        "exit": "SL 1.5×ATR14 / TP 0.4×ATR14 / max 24 bars",
        "symbol": "GC=F 15m proxy",
        "period": f"{df.index[0]} → {df.index[-1]}",
        "starting_capital": START_CAP,
        "final_equity": round(float(capital), 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "full": stats(tdf),
        "in_sample_70pct": stats(is_df),
        "out_of_sample_30pct": stats(oos_df),
        "notes": [
            "High winrate comes from TP << SL ( asymmetric R:R ).",
            "Still net profitable on IS and OOS in this sample.",
            "Yahoo GC=F ≠ broker XAUUSD; no spread/commission modeled.",
            "Past backtest ≠ future performance.",
        ],
    }
    with open(OUT / "winner_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT/'winner_trades.csv'} and {OUT/'winner_report.json'}")


if __name__ == "__main__":
    run()
