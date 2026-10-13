#!/usr/bin/env python3
"""
Validate Asian-Range Fade (1h) on REAL broker XAUUSD from MetaTrader 5.

Run this on a Windows machine where MT5 terminal is installed & logged in:

    pip install MetaTrader5 pandas numpy
    python mt5_validate_asian_fade.py

Optional login (if terminal is not already authorized):
    set MT5_LOGIN=123456
    set MT5_PASSWORD=yourpass
    set MT5_SERVER=YourBroker-Server
    set MT5_PATH=C:\\Program Files\\MetaTrader 5\\terminal64.exe

Or validate from an MT5-exported CSV (no live terminal needed):
    python mt5_validate_asian_fade.py --csv XAUUSD_H1.csv

CSV columns required (MT5 export or similar):
    time,open,high,low,close
    (time like 2024.05.21 13:00 or ISO)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Same constants as strategy_asian_fade_1h.py
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
SYMBOL_CANDIDATES = [
    "XAUUSD",
    "GOLD",
    "XAUUSD.pro",
    "XAUUSD.",
    "XAUUSD.m",
    "XAUUSDm",
    "XAUUSD.a",
]


def load_from_mt5(bars: int = 20000) -> tuple[pd.DataFrame, str]:
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise SystemExit(
            "MetaTrader5 package not available on this OS.\n"
            "Install/run on Windows with MT5, or use --csv exported from MT5.\n"
            f"Detail: {e}"
        )

    path = os.getenv("MT5_PATH")
    ok = mt5.initialize(path=path) if path else mt5.initialize()
    if not ok:
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    if login and password and server:
        if not mt5.login(int(login), password=password, server=server):
            err = mt5.last_error()
            mt5.shutdown()
            raise SystemExit(f"MT5 login failed: {err}")

    symbol = None
    for name in SYMBOL_CANDIDATES:
        info = mt5.symbol_info(name)
        if info is not None:
            mt5.symbol_select(name, True)
            symbol = name
            break
    if not symbol:
        mt5.shutdown()
        raise SystemExit(f"No gold symbol found. Tried: {SYMBOL_CANDIDATES}")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
    account = mt5.account_info()
    terminal = mt5.terminal_info()
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise SystemExit("No H1 rates returned from MT5")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df.set_index("time")[["Open", "High", "Low", "Close"]]
    meta = {
        "source": "mt5",
        "symbol": symbol,
        "account": getattr(account, "login", None),
        "server": getattr(account, "server", None),
        "company": getattr(account, "company", None),
        "terminal": getattr(terminal, "name", None),
        "bars": len(df),
    }
    print(json.dumps(meta, indent=2))
    return df, symbol


def load_from_csv(path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    # normalize names
    mapping = {}
    for want in ("time", "date", "datetime", "open", "high", "low", "close"):
        if want in cols:
            mapping[want] = cols[want]
    if "time" not in mapping and "datetime" in mapping:
        mapping["time"] = mapping["datetime"]
    if "time" not in mapping and "date" in mapping:
        mapping["time"] = mapping["date"]
    need = ["time", "open", "high", "low", "close"]
    missing = [k for k in need if k not in mapping]
    if missing:
        raise SystemExit(f"CSV missing columns {missing}. Have: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "Open": df[mapping["open"]].astype(float),
            "High": df[mapping["high"]].astype(float),
            "Low": df[mapping["low"]].astype(float),
            "Close": df[mapping["close"]].astype(float),
        }
    )
    t = pd.to_datetime(df[mapping["time"]], utc=True, errors="coerce")
    if t.isna().all():
        t = pd.to_datetime(df[mapping["time"]], format="%Y.%m.%d %H:%M", utc=True, errors="coerce")
    if t.isna().any():
        raise SystemExit("Could not parse time column")
    out.index = t.dt.tz_localize(None)
    out = out.sort_index().dropna()
    print(json.dumps({"source": "csv", "path": path, "bars": len(out)}, indent=2))
    return out, Path(path).stem


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
    hour = d.index.hour
    d["overlap"] = (hour >= 13) & (hour < 17)
    d["long"] = d["overlap"] & (d["Low"] <= d["range_lo"]) & (c > d["range_lo"]) & (d["RSI"] < OS)
    d["short"] = d["overlap"] & (d["High"] >= d["range_hi"]) & (c < d["range_hi"]) & (d["RSI"] > OB)
    return d


def backtest(df: pd.DataFrame, spread_points: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Strict causality: signal close[i] → fill open[i+1]. spread_points = absolute price spread."""
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
        atr = atrs[i]
        entry = opens[i + 1]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
            i += 1
            continue
        # pay half-spread on entry
        if direction == "LONG":
            entry += spread_points / 2
        else:
            entry -= spread_points / 2
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
        # pay half-spread on exit
        if direction == "LONG":
            exit_price -= spread_points / 2
        else:
            exit_price += spread_points / 2
        pnl = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0)
        trades.append(
            {
                "entry_time": str(df.index[i]),
                "direction": direction,
                "entry": round(float(entry), 3),
                "exit": round(float(exit_price), 3),
                "reason": reason,
                "bars": bars,
                "pnl": round(float(pnl), 2),
                "equity": round(float(capital), 2),
            }
        )
        i = j + 1 if j > i else i + 1

    tdf = pd.DataFrame(trades)
    arr = tdf["pnl"].to_numpy() if len(tdf) else np.array([])
    if len(arr) == 0:
        return tdf, {"trades": 0, "error": "no trades"}
    wins, losses = arr[arr > 0], arr[arr <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    by_day = tdf.groupby(pd.to_datetime(tdf["entry_time"]).dt.date).size()
    summary = {
        "strategy": "Asian-Range Fade 1h (MT5/XAUUSD validation)",
        "period": f"{df.index[0]} → {df.index[-1]}",
        "bars": len(df),
        "spread_points": spread_points,
        "trades": int(len(arr)),
        "winrate_pct": round(float((arr > 0).mean()) * 100, 2),
        "profit_factor": round(gw / gl, 3) if gl else None,
        "net_pnl": round(float(arr.sum()), 2),
        "final_equity": round(float(capital), 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "tp_exits": int((tdf["reason"] == "TP").sum()),
        "sl_exits": int((tdf["reason"] == "SL").sum()),
        "avg_trades_per_active_day": round(float(by_day.mean()), 2),
        "days_with_trades": int(len(by_day)),
        "avg_trades_per_calendar_day": round(
            len(arr) / max(1, (df.index[-1] - df.index[0]).days), 3
        ),
    }
    return tdf, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="MT5-exported H1 CSV instead of live MT5 API")
    ap.add_argument("--bars", type=int, default=20000)
    ap.add_argument(
        "--spread",
        type=float,
        default=0.25,
        help="Round-trip spread in price points (default 0.25 = $0.25)",
    )
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.csv:
        df, symbol = load_from_csv(args.csv)
    else:
        df, symbol = load_from_mt5(args.bars)

    prepared = prepare(df)
    # zero-spread and with-spread
    for spread, tag in [(0.0, "raw"), (args.spread, "spread")]:
        trades, summary = backtest(prepared, spread_points=spread)
        summary["symbol"] = symbol
        out_json = OUT / f"mt5_asian_fade_1h_{tag}.json"
        out_csv = OUT / f"mt5_asian_fade_1h_{tag}_trades.csv"
        out_json.write_text(json.dumps(summary, indent=2))
        trades.to_csv(out_csv, index=False)
        print(f"\n=== {tag.upper()} (spread={spread}) ===")
        print(json.dumps(summary, indent=2))
        print(f"Wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
