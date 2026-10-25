#!/usr/bin/env python3
"""
High-frequency high-WR alert/backtest bot (CAUSAL).

Default strategy = dense daily trader from search:
  RSI(5) cross into extremes on 15m
  Sessions: London + Overlap + NY (08–22 UTC)
  LONG when RSI crosses below 25; SHORT when RSI crosses above 80
  Fill: next bar open after signal close
  SL 2.0×ATR · TP 0.2×ATR · max hold 6

Backtest (GC=F 15m ~60d): ~94.4% WR · ~3.4 trades/day · PF ~1.83

  python3 live_hf_wr_bot.py --backtest
  python3 live_hf_wr_bot.py --live          # Windows + MT5 + Telegram
  python3 live_hf_wr_bot.py --backtest --strategy stoch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("hf_search_results")
START_CAP = 500.0
RISK_PCT = 0.02
MAX_RISK = 20.0
HALT = 425.0

STRATEGIES = {
    "rsi": {
        "name": "RSI(5) cross fade 15m London+Overlap+NY",
        "rsi": 5,
        "os": 25,
        "ob": 80,
        "sessions": {"london", "overlap", "ny"},
        "sl_atr": 2.0,
        "tp_atr": 0.2,
        "max_hold": 6,
        "kind": "rsi",
    },
    "stoch": {
        "name": "Stoch(5) cross fade 15m London+Overlap",
        "stoch": 5,
        "os": 20,
        "ob": 80,
        "sessions": {"london", "overlap"},
        "sl_atr": 2.0,
        "tp_atr": 0.2,
        "max_hold": 12,
        "kind": "stoch",
    },
}


def session_name(hour: int) -> str:
    if 0 <= hour < 8:
        return "asian"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 17:
        return "overlap"
    if 17 <= hour < 22:
        return "ny"
    return "late"


def prepare(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(com=13, adjust=False).mean()
    d["session"] = [session_name(ts.hour) for ts in d.index]
    sess = d["session"].isin(cfg["sessions"])

    if cfg["kind"] == "rsi":
        period = cfg["rsi"]
        delta = c.diff()
        ag = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        al = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rsi = (100 - (100 / (1 + ag / al.replace(0, np.nan)))).fillna(50)
        d["RSI"] = rsi
        d["long"] = (rsi < cfg["os"]) & (rsi.shift(1) >= cfg["os"]) & sess
        d["short"] = (rsi > cfg["ob"]) & (rsi.shift(1) <= cfg["ob"]) & sess
    else:
        period = cfg["stoch"]
        ll = l.rolling(period).min()
        hh = h.rolling(period).max()
        k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        dline = k.rolling(3).mean()
        d["STOCH_K"], d["STOCH_D"] = k, dline
        d["long"] = (k < cfg["os"]) & (k > dline) & (k.shift(1) <= dline.shift(1)) & sess
        d["short"] = (k > cfg["ob"]) & (k < dline) & (k.shift(1) >= dline.shift(1)) & sess
    return d


def backtest(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    d = prepare(df, cfg)
    opens = d["Open"].to_numpy(float)
    highs = d["High"].to_numpy(float)
    lows = d["Low"].to_numpy(float)
    closes = d["Close"].to_numpy(float)
    atrs = d["ATR"].to_numpy(float)
    long_e = d["long"].fillna(False).to_numpy()
    short_e = d["short"].fillna(False).to_numpy()
    capital = START_CAP
    trades = []
    i = 250
    n = len(d)
    while i < n - 2:
        if capital < HALT:
            break
        gl, gs = bool(long_e[i]), bool(short_e[i])
        if gl == gs:
            i += 1
            continue
        direction = "LONG" if gl else "SHORT"
        atr, entry = atrs[i], opens[i + 1]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
            i += 1
            continue
        risk = min(MAX_RISK, capital * RISK_PCT)
        sl_dist, tp_dist = atr * cfg["sl_atr"], atr * cfg["tp_atr"]
        qty = risk / sl_dist
        sl = entry - sl_dist if direction == "LONG" else entry + sl_dist
        tp = entry + tp_dist if direction == "LONG" else entry - tp_dist
        j, bars, exit_p, reason, hit = i + 1, 0, entry, "time", False
        while j < n and bars < cfg["max_hold"]:
            hi, lo = highs[j], lows[j]
            bars += 1
            if direction == "LONG":
                if lo <= sl:
                    exit_p, reason, hit = sl, "SL", True
                    break
                if hi >= tp:
                    exit_p, reason, hit = tp, "TP", True
                    break
            else:
                if hi >= sl:
                    exit_p, reason, hit = sl, "SL", True
                    break
                if lo <= tp:
                    exit_p, reason, hit = tp, "TP", True
                    break
            j += 1
        if not hit:
            exit_p = closes[min(j, n - 1)]
        pnl = (exit_p - entry) * qty if direction == "LONG" else (entry - exit_p) * qty
        capital += pnl
        trades.append(
            {
                "signal_time": str(d.index[i]),
                "entry_time": str(d.index[i + 1]),
                "direction": direction,
                "entry": round(float(entry), 2),
                "sl": round(float(sl), 2),
                "tp": round(float(tp), 2),
                "exit": round(float(exit_p), 2),
                "reason": reason,
                "pnl": round(float(pnl), 2),
                "equity": round(float(capital), 2),
            }
        )
        i = j + 1 if j > i else i + 1

    tdf = pd.DataFrame(trades)
    arr = tdf["pnl"].to_numpy()
    by = tdf.groupby(pd.to_datetime(tdf["signal_time"]).dt.date).size()
    bar_days = len(set(d.index[250:].date))
    wins, losses = arr[arr > 0], arr[arr <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    summary = {
        "strategy": cfg["name"],
        "period": f"{d.index[0]} → {d.index[-1]}",
        "trades": int(len(arr)),
        "winrate_pct": round(float((arr > 0).mean()) * 100, 2),
        "profit_factor": round(gw / gl, 3) if gl else None,
        "net_pnl": round(float(arr.sum()), 2),
        "final_equity": round(float(capital), 2),
        "trades_per_bar_day": round(len(arr) / max(bar_days, 1), 3),
        "day_coverage": round(len(by) / max(bar_days, 1), 3),
        "avg_trades_on_active_days": round(float(by.mean()), 2),
        "max_trades_one_day": int(by.max()) if len(by) else 0,
        "tp_exits": int((tdf["reason"] == "TP").sum()),
        "sl_exits": int((tdf["reason"] == "SL").sum()),
        "causality": "signal at close → fill next open",
    }
    return tdf, summary


def run_backtest(strategy: str) -> None:
    import yfinance as yf

    cfg = STRATEGIES[strategy]
    raw = yf.download("GC=F", interval="15m", period="60d", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    trades, summary = backtest(df, cfg)
    OUT.mkdir(exist_ok=True)
    trades.to_csv(OUT / f"bot_{strategy}_trades.csv", index=False)
    (OUT / f"bot_{strategy}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(text)
        return
    import requests

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "Markdown"},
            timeout=8,
        )
    except Exception as e:
        print("telegram error", e)


def run_live(strategy: str) -> None:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        sys.exit("MetaTrader5 requires Windows + MT5. Use --backtest on this machine.")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    cfg = STRATEGIES[strategy]
    path = os.getenv("MT5_PATH")
    ok = mt5.initialize(path=path) if path else mt5.initialize()
    if not ok:
        sys.exit(f"MT5 init failed: {mt5.last_error()}")
    symbol = None
    for name in ["XAUUSD", "GOLD", "XAUUSD.pro", "XAUUSDm"]:
        if mt5.symbol_info(name):
            mt5.symbol_select(name, True)
            symbol = name
            break
    if not symbol:
        sys.exit("gold symbol not found")

    send_telegram(
        f"🚀 *HF WR Bot ONLINE*\n{cfg['name']}\n"
        f"SL {cfg['sl_atr']}×ATR / TP {cfg['tp_atr']}×ATR · next-open causal"
    )
    last_signal_ts = None
    while True:
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 300)
            if rates is None:
                time.sleep(2)
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
            df = df.rename(
                columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
            )
            df = df.set_index("time")[["Open", "High", "Low", "Close"]]
            d = prepare(df, cfg)
            ts = d.index[-2]
            if last_signal_ts == ts:
                time.sleep(1)
                continue
            row = d.iloc[-2]
            if bool(row["long"]) or bool(row["short"]):
                last_signal_ts = ts
                direction = "LONG" if row["long"] else "SHORT"
                atr = float(row["ATR"])
                tick = mt5.symbol_info_tick(symbol)
                entry = float(tick.ask if direction == "LONG" else tick.bid)
                sl_dist, tp_dist = atr * cfg["sl_atr"], atr * cfg["tp_atr"]
                sl = entry - sl_dist if direction == "LONG" else entry + sl_dist
                tp = entry + tp_dist if direction == "LONG" else entry - tp_dist
                send_telegram(
                    f"{'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} *HF WR SIGNAL*\n"
                    f"{cfg['name']}\n"
                    f"Signal bar: `{ts}` UTC\n"
                    f"Live ref: `${entry:.2f}`\n"
                    f"SL: `${sl:.2f}` | TP: `${tp:.2f}`"
                )
            time.sleep(1)
        except Exception as e:
            send_telegram(f"⚠️ HF bot error: {e}")
            time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="rsi")
    args = ap.parse_args()
    if args.live:
        run_live(args.strategy)
    else:
        run_backtest(args.strategy)


if __name__ == "__main__":
    main()
