#!/usr/bin/env python3
"""
Aggressive multi-strategy gold searcher.

Keeps expanding strategy families, parameter grids, and filter combinations
until it finds candidates with:
  - winrate >= TARGET_WR
  - profit_factor >= TARGET_PF
  - trades >= MIN_TRADES
  - out-of-sample winrate >= OOS_WR
  - out-of-sample profit_factor >= OOS_PF

Designed to run for many hours. Writes progress continuously.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("search_results")
OUT.mkdir(exist_ok=True)
BEST_PATH = OUT / "best_candidates.jsonl"
PROGRESS_PATH = OUT / "progress.json"
WINNER_PATH = OUT / "WINNER.json"
LOG_PATH = OUT / "search.log"

TARGET_WR = 0.80
TARGET_PF = 1.25
MIN_TRADES = 40
OOS_WR = 0.78
OOS_PF = 1.15
OOS_MIN_TRADES = 15
START_CAP = 500.0
TRADE_POWER = 10_000.0
MAX_RISK = 20.0
RISK_PCT = 0.02
HALT = START_CAP * 0.85

RNG = random.Random(42)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    specs = [
        ("GC=F", "15m", "60d", "15m"),
        ("GC=F", "1h", "730d", "1h"),
        ("GC=F", "5m", "60d", "5m"),
    ]
    for ticker, interval, period, key in specs:
        try:
            df = yf.download(
                ticker, interval=interval, period=period, progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close"]].dropna()
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            if len(df) > 300:
                frames[key] = df
                log(f"Loaded {key}: {len(df)} bars {df.index[0]} → {df.index[-1]}")
            else:
                log(f"Skip {key}: only {len(df)} bars")
        except Exception as e:
            log(f"Failed {key}: {e}")
    if not frames:
        raise RuntimeError("No market data available")
    return frames


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    for span in (5, 8, 9, 12, 20, 21, 25, 34, 50, 55, 100, 200):
        d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
        d[f"SMA_{span}"] = c.rolling(span).mean()

    # RSI variants
    for period in (2, 3, 5, 7, 14, 21):
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ag = gain.ewm(com=period - 1, adjust=False).mean()
        al = loss.ewm(com=period - 1, adjust=False).mean()
        rs = ag / al.replace(0, np.nan)
        d[f"RSI_{period}"] = (100 - (100 / (1 + rs))).fillna(50)

    # ATR
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    for period in (7, 14, 21):
        d[f"ATR_{period}"] = tr.ewm(com=period - 1, adjust=False).mean()

    # Bollinger
    for period, k in ((20, 2.0), (20, 2.5), (10, 1.5), (50, 2.0)):
        mid = c.rolling(period).mean()
        std = c.rolling(period).std()
        d[f"BB_MID_{period}_{k}"] = mid
        d[f"BB_UP_{period}_{k}"] = mid + k * std
        d[f"BB_LO_{period}_{k}"] = mid - k * std
        d[f"BB_PCT_{period}_{k}"] = (c - (mid - k * std)) / (2 * k * std.replace(0, np.nan))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIG"]

    # Stochastic
    for period in (5, 14):
        ll = l.rolling(period).min()
        hh = h.rolling(period).max()
        d[f"STOCH_K_{period}"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        d[f"STOCH_D_{period}"] = d[f"STOCH_K_{period}"].rolling(3).mean()

    # Donchian / breakout levels
    for period in (10, 20, 40, 55):
        d[f"DON_HI_{period}"] = h.rolling(period).max().shift(1)
        d[f"DON_LO_{period}"] = l.rolling(period).min().shift(1)

    # Momentum / ROC
    for period in (3, 5, 10, 20):
        d[f"ROC_{period}"] = c.pct_change(period) * 100
        d[f"MOM_{period}"] = c - c.shift(period)

    # Session / calendar
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["session"] = np.select(
        [
            (d["hour"] >= 0) & (d["hour"] < 8),
            (d["hour"] >= 8) & (d["hour"] < 13),
            (d["hour"] >= 13) & (d["hour"] < 17),
            (d["hour"] >= 17) & (d["hour"] < 22),
        ],
        ["asian", "london", "overlap", "ny"],
        default="late",
    )

    # Candle features
    body = (c - d["Open"]).abs()
    rng = (h - l).replace(0, np.nan)
    d["body_pct"] = body / rng
    d["upper_wick"] = (h - np.maximum(c, d["Open"])) / rng
    d["lower_wick"] = (np.minimum(c, d["Open"]) - l) / rng
    d["bull"] = c > d["Open"]
    d["bear"] = c < d["Open"]
    d["range"] = h - l
    d["range_ma20"] = d["range"].rolling(20).mean()
    d["vol_ok"] = d["range"] > 0.5 * d["range_ma20"]

    # Prior day levels via resampling if enough days
    daily = d.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    daily["pdh"] = daily["High"].shift(1)
    daily["pdl"] = daily["Low"].shift(1)
    daily["pd_mid"] = (daily["pdh"] + daily["pdl"]) / 2
    daily["pd_green"] = daily["Close"].shift(1) > daily["Open"].shift(1)
    daily = daily[["pdh", "pdl", "pd_mid", "pd_green"]]
    d = d.join(daily, how="left")
    d[["pdh", "pdl", "pd_mid", "pd_green"]] = d[["pdh", "pdl", "pd_mid", "pd_green"]].ffill()

    return d


@dataclass
class Result:
    name: str
    timeframe: str
    winrate: float
    profit_factor: float
    trades: int
    net_pnl: float
    final_equity: float
    expectancy: float
    max_dd: float
    avg_win: float
    avg_loss: float
    params: dict
    oos_winrate: float | None = None
    oos_profit_factor: float | None = None
    oos_trades: int | None = None
    oos_net_pnl: float | None = None
    score: float = 0.0


def simulate(
    df: pd.DataFrame,
    long_entries: np.ndarray,
    short_entries: np.ndarray,
    sl_mult_atr: float,
    tp_mult_atr: float,
    atr_col: str = "ATR_14",
    use_be: bool = False,
    be_rr: float = 1.0,
    partial_frac: float = 0.0,
    max_hold: int = 48,
    allow_short: bool = True,
    allow_long: bool = True,
    start_i: int = 250,
    end_i: int | None = None,
) -> dict:
    """Bar-based simulator. Entries on bar i close; manage from i+1."""
    n = len(df)
    end_i = end_i if end_i is not None else n - 1
    capital = START_CAP
    peak = START_CAP
    max_dd = 0.0
    pnls: list[float] = []

    closes = df["Close"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    atrs = df[atr_col].to_numpy(dtype=float) if atr_col in df.columns else np.full(n, np.nan)

    i = start_i
    while i < end_i:
        if capital < HALT:
            break
        go_long = allow_long and bool(long_entries[i])
        go_short = allow_short and bool(short_entries[i])
        if not go_long and not go_short:
            i += 1
            continue
        if go_long and go_short:
            i += 1
            continue

        direction = "LONG" if go_long else "SHORT"
        entry = closes[i]
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry) or entry <= 0:
            i += 1
            continue

        risk = min(MAX_RISK, capital * RISK_PCT)
        qty = TRADE_POWER / entry
        sl_dist = max(atr * sl_mult_atr, 1e-6)
        # Scale qty so dollar risk ≈ risk when possible
        qty = risk / sl_dist
        tp_dist = atr * tp_mult_atr

        if direction == "LONG":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        qty_rem = qty
        realized = 0.0
        tp1_hit = False
        j = i + 1
        bars = 0
        exit_price = entry
        exited = False

        while j < n and bars < max_hold:
            hi, lo = highs[j], lows[j]
            bars += 1
            if direction == "LONG":
                if (not tp1_hit) and lo <= sl:
                    realized += (sl - entry) * qty_rem
                    exit_price = sl
                    exited = True
                    break
                if use_be and (not tp1_hit) and hi >= entry + be_rr * sl_dist:
                    if partial_frac > 0:
                        close_q = qty * partial_frac
                        realized += (entry + be_rr * sl_dist - entry) * close_q
                        qty_rem -= close_q
                    tp1_hit = True
                    sl = entry
                if tp1_hit and lo <= sl:
                    realized += (sl - entry) * qty_rem
                    exit_price = sl
                    exited = True
                    break
                if hi >= tp:
                    realized += (tp - entry) * qty_rem
                    exit_price = tp
                    exited = True
                    break
            else:
                if (not tp1_hit) and hi >= sl:
                    realized += (entry - sl) * qty_rem
                    exit_price = sl
                    exited = True
                    break
                if use_be and (not tp1_hit) and lo <= entry - be_rr * sl_dist:
                    if partial_frac > 0:
                        close_q = qty * partial_frac
                        realized += (entry - (entry - be_rr * sl_dist)) * close_q
                        qty_rem -= close_q
                    tp1_hit = True
                    sl = entry
                if tp1_hit and hi >= sl:
                    realized += (entry - sl) * qty_rem
                    exit_price = sl
                    exited = True
                    break
                if lo <= tp:
                    realized += (entry - tp) * qty_rem
                    exit_price = tp
                    exited = True
                    break
            j += 1

        if not exited:
            # time exit at last managed close
            last = closes[min(j, n - 1)]
            if direction == "LONG":
                realized += (last - entry) * qty_rem
            else:
                realized += (entry - last) * qty_rem
            exit_price = last

        capital += realized
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0)
        pnls.append(realized)
        i = j + 1 if j > i else i + 1

    if not pnls:
        return {
            "trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "final_equity": capital,
            "expectancy": 0.0,
            "max_dd": max_dd,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    arr = np.array(pnls, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
    return {
        "trades": int(len(arr)),
        "winrate": float((arr > 0).mean()),
        "profit_factor": float(pf),
        "net_pnl": float(arr.sum()),
        "final_equity": float(capital),
        "expectancy": float(arr.mean()),
        "max_dd": float(max_dd),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def session_mask(df: pd.DataFrame, sessions: tuple[str, ...] | None) -> np.ndarray:
    if not sessions:
        return np.ones(len(df), dtype=bool)
    return df["session"].isin(sessions).to_numpy()


def gen_signals(df: pd.DataFrame, kind: str, p: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    long_e = np.zeros(n, dtype=bool)
    short_e = np.zeros(n, dtype=bool)
    c = df["Close"]
    sess = session_mask(df, p.get("sessions"))

    if kind == "rsi_mr":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (rsi < p["os"]) & (rsi.shift(1) >= p["os"]) & sess
        short_e = (rsi > p["ob"]) & (rsi.shift(1) <= p["ob"]) & sess

    elif kind == "rsi_mr_ema_filter":
        rsi = df[f"RSI_{p['rsi']}"]
        ema = df[f"EMA_{p['ema']}"]
        long_e = (rsi < p["os"]) & (c > ema) & sess
        short_e = (rsi > p["ob"]) & (c < ema) & sess

    elif kind == "bb_mr":
        up = df[f"BB_UP_{p['period']}_{p['k']}"]
        lo = df[f"BB_LO_{p['period']}_{p['k']}"]
        long_e = (df["Low"] <= lo) & (c > lo) & sess
        short_e = (df["High"] >= up) & (c < up) & sess

    elif kind == "bb_mr_rsi":
        up = df[f"BB_UP_{p['period']}_{p['k']}"]
        lo = df[f"BB_LO_{p['period']}_{p['k']}"]
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= lo) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= up) & (rsi > p["ob"]) & sess

    elif kind == "stoch_mr":
        kcol = df[f"STOCH_K_{p['period']}"]
        dcol = df[f"STOCH_D_{p['period']}"]
        long_e = (kcol < p["os"]) & (kcol > dcol) & (kcol.shift(1) <= dcol.shift(1)) & sess
        short_e = (kcol > p["ob"]) & (kcol < dcol) & (kcol.shift(1) >= dcol.shift(1)) & sess

    elif kind == "ema_pullback":
        e_fast = df[f"EMA_{p['fast']}"]
        e_slow = df[f"EMA_{p['slow']}"]
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (e_fast > e_slow) & (c > e_slow) & (df["Low"] <= e_fast) & (c > e_fast) & (rsi > p["rsi_min"]) & sess
        short_e = (e_fast < e_slow) & (c < e_slow) & (df["High"] >= e_fast) & (c < e_fast) & (rsi < p["rsi_max"]) & sess

    elif kind == "ema_stack_break":
        e1, e2, e3 = df[f"EMA_{p['a']}"], df[f"EMA_{p['b']}"], df[f"EMA_{p['c']}"]
        long_e = (e1 > e2) & (e2 > e3) & (c > e1) & (c.shift(1) <= e1.shift(1)) & sess
        short_e = (e1 < e2) & (e2 < e3) & (c < e1) & (c.shift(1) >= e1.shift(1)) & sess

    elif kind == "macd_cross":
        long_e = (df["MACD"] > df["MACD_SIG"]) & (df["MACD"].shift(1) <= df["MACD_SIG"].shift(1)) & (df["MACD_HIST"] > 0) & sess
        short_e = (df["MACD"] < df["MACD_SIG"]) & (df["MACD"].shift(1) >= df["MACD_SIG"].shift(1)) & (df["MACD_HIST"] < 0) & sess

    elif kind == "donchian_break":
        hi = df[f"DON_HI_{p['period']}"]
        lo = df[f"DON_LO_{p['period']}"]
        long_e = (c > hi) & (c.shift(1) <= hi.shift(1)) & sess
        short_e = (c < lo) & (c.shift(1) >= lo.shift(1)) & sess

    elif kind == "nr_breakout":
        # narrow range then break
        nr = df["range"] < df["range"].rolling(p["look"]).quantile(p["q"])
        long_e = nr.shift(1) & (c > df["High"].shift(1)) & sess
        short_e = nr.shift(1) & (c < df["Low"].shift(1)) & sess

    elif kind == "pdh_pdl_fade":
        long_e = (df["Low"] <= df["pdl"]) & (c > df["pdl"]) & (df[f"RSI_{p['rsi']}"] < p["os"]) & sess
        short_e = (df["High"] >= df["pdh"]) & (c < df["pdh"]) & (df[f"RSI_{p['rsi']}"] > p["ob"]) & sess

    elif kind == "pdh_pdl_break":
        long_e = (c > df["pdh"]) & (c.shift(1) <= df["pdh"].shift(1)) & (df[f"RSI_{p['rsi']}"] > 50) & sess
        short_e = (c < df["pdl"]) & (c.shift(1) >= df["pdl"].shift(1)) & (df[f"RSI_{p['rsi']}"] < 50) & sess

    elif kind == "engulf_trend":
        prev_bull = df["bull"].shift(1)
        prev_bear = df["bear"].shift(1)
        eng_bull = df["bull"] & prev_bear & (c >= df["Open"].shift(1)) & (df["Open"] <= c.shift(1))
        eng_bear = df["bear"] & prev_bull & (c <= df["Open"].shift(1)) & (df["Open"] >= c.shift(1))
        ema = df[f"EMA_{p['ema']}"]
        long_e = eng_bull & (c > ema) & sess
        short_e = eng_bear & (c < ema) & sess

    elif kind == "pinbar_mr":
        long_e = (df["lower_wick"] > p["wick"]) & (df["body_pct"] < p["body"]) & (df[f"RSI_{p['rsi']}"] < p["os"]) & sess
        short_e = (df["upper_wick"] > p["wick"]) & (df["body_pct"] < p["body"]) & (df[f"RSI_{p['rsi']}"] > p["ob"]) & sess

    elif kind == "triple_confirm_mr":
        rsi = df[f"RSI_{p['rsi']}"]
        lo = df[f"BB_LO_{p['bb']}_{p['k']}"]
        kcol = df[f"STOCH_K_{p['stoch']}"]
        up = df[f"BB_UP_{p['bb']}_{p['k']}"]
        long_e = (rsi < p["os"]) & (df["Low"] <= lo) & (kcol < p["os"]) & sess
        short_e = (rsi > p["ob"]) & (df["High"] >= up) & (kcol > p["ob"]) & sess

    elif kind == "quad_confirm_mr":
        rsi = df[f"RSI_{p['rsi']}"]
        lo = df[f"BB_LO_{p['bb']}_{p['k']}"]
        up = df[f"BB_UP_{p['bb']}_{p['k']}"]
        kcol = df[f"STOCH_K_{p['stoch']}"]
        ema = df[f"EMA_{p['ema']}"]
        long_e = (rsi < p["os"]) & (df["Low"] <= lo) & (kcol < p["os"]) & (c > ema) & sess
        short_e = (rsi > p["ob"]) & (df["High"] >= up) & (kcol > p["ob"]) & (c < ema) & sess

    elif kind == "asian_range_fade":
        # fade extremes during london using asian range proxy via rolling 32 bars (~8h on 15m)
        look = p["look"]
        a_hi = df["High"].rolling(look).max().shift(1)
        a_lo = df["Low"].rolling(look).min().shift(1)
        long_e = (df["Low"] <= a_lo) & (c > a_lo) & (df[f"RSI_{p['rsi']}"] < p["os"]) & sess
        short_e = (df["High"] >= a_hi) & (c < a_hi) & (df[f"RSI_{p['rsi']}"] > p["ob"]) & sess

    elif kind == "combo_or":
        # OR of two base masks passed in
        long_e = p["long_a"] | p["long_b"]
        short_e = p["short_a"] | p["short_b"]

    elif kind == "combo_and":
        long_e = p["long_a"] & p["long_b"]
        short_e = p["short_a"] & p["short_b"]

    else:
        raise ValueError(kind)

    # Optional volatility filter
    if p.get("vol_ok"):
        vok = df["vol_ok"].fillna(False).to_numpy()
        long_e = long_e & vok
        short_e = short_e & vok

    # Optional hour filter
    if p.get("hours"):
        hm = df["hour"].isin(p["hours"]).to_numpy()
        long_e = long_e & hm
        short_e = short_e & hm

    # Optional DOW filter
    if p.get("dows"):
        dm = df["dow"].isin(p["dows"]).to_numpy()
        long_e = long_e & dm
        short_e = short_e & dm

    return long_e.to_numpy() if hasattr(long_e, "to_numpy") else long_e, short_e.to_numpy() if hasattr(short_e, "to_numpy") else short_e


def score_result(m: dict, oos: dict | None = None) -> float:
    """Higher is better. Heavy weight on WR + PF + trade count + OOS."""
    if m["trades"] < 10:
        return -1e9
    s = (
        m["winrate"] * 100
        + m["profit_factor"] * 15
        + min(m["trades"], 200) * 0.05
        + (5 if m["net_pnl"] > 0 else -20)
        - m["max_dd"] * 30
    )
    if oos:
        if oos["trades"] < 5:
            s -= 50
        else:
            s += oos["winrate"] * 120
            s += oos["profit_factor"] * 20
            s += 10 if oos["net_pnl"] > 0 else -25
    return float(s)


def meets_target(is_m: dict, oos_m: dict) -> bool:
    return (
        is_m["trades"] >= MIN_TRADES
        and is_m["winrate"] >= TARGET_WR
        and is_m["profit_factor"] >= TARGET_PF
        and is_m["net_pnl"] > 0
        and oos_m["trades"] >= OOS_MIN_TRADES
        and oos_m["winrate"] >= OOS_WR
        and oos_m["profit_factor"] >= OOS_PF
        and oos_m["net_pnl"] > 0
    )


def evaluate(
    name: str,
    tf: str,
    df: pd.DataFrame,
    long_e: np.ndarray,
    short_e: np.ndarray,
    exit_p: dict,
    params: dict,
) -> Result | None:
    n = len(df)
    split = int(n * 0.7)
    common = dict(
        long_entries=long_e,
        short_entries=short_e,
        sl_mult_atr=exit_p["sl"],
        tp_mult_atr=exit_p["tp"],
        atr_col=exit_p.get("atr", "ATR_14"),
        use_be=exit_p.get("be", False),
        be_rr=exit_p.get("be_rr", 1.0),
        partial_frac=exit_p.get("partial", 0.0),
        max_hold=exit_p.get("max_hold", 48),
        allow_short=exit_p.get("allow_short", True),
        allow_long=exit_p.get("allow_long", True),
    )
    is_m = simulate(df, start_i=250, end_i=split, **common)
    oos_m = simulate(df, start_i=split, end_i=n - 1, **common)
    if is_m["trades"] < 10:
        return None
    sc = score_result(is_m, oos_m)
    return Result(
        name=name,
        timeframe=tf,
        winrate=is_m["winrate"],
        profit_factor=is_m["profit_factor"],
        trades=is_m["trades"],
        net_pnl=is_m["net_pnl"],
        final_equity=is_m["final_equity"],
        expectancy=is_m["expectancy"],
        max_dd=is_m["max_dd"],
        avg_win=is_m["avg_win"],
        avg_loss=is_m["avg_loss"],
        params={**params, "exit": exit_p},
        oos_winrate=oos_m["winrate"],
        oos_profit_factor=oos_m["profit_factor"],
        oos_trades=oos_m["trades"],
        oos_net_pnl=oos_m["net_pnl"],
        score=sc,
    )


def exit_grid(phase: int) -> list[dict]:
    grids = []
    sls = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    # High WR often comes from TP << SL (asymmetric), but we still require PF
    tps = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    holds = [4, 8, 12, 24, 48]
    if phase == 0:
        sls, tps, holds = [0.6, 1.0, 1.5], [0.4, 0.6, 1.0], [8, 24]
    elif phase == 1:
        sls, tps, holds = [0.5, 0.8, 1.2, 2.0], [0.3, 0.5, 0.8, 1.2], [6, 12, 24]
    for sl, tp, mh, be, sides in itertools.product(
        sls,
        tps,
        holds,
        [False, True] if phase >= 1 else [False],
        [
            {"allow_long": True, "allow_short": True},
            {"allow_long": True, "allow_short": False},
            {"allow_long": False, "allow_short": True},
        ]
        if phase >= 2
        else [{"allow_long": True, "allow_short": True}],
    ):
        if tp / sl > 3.5:  # skip absurd
            continue
        grids.append(
            {
                "sl": sl,
                "tp": tp,
                "max_hold": mh,
                "be": be,
                "be_rr": 0.8 if be else 1.0,
                "partial": 0.5 if be else 0.0,
                "atr": "ATR_14",
                **sides,
            }
        )
    return grids


def strategy_param_space(phase: int) -> list[tuple[str, dict]]:
    space: list[tuple[str, dict]] = []
    sessions_opts = [
        None,
        ("asian",),
        ("london",),
        ("overlap",),
        ("ny",),
        ("asian", "london"),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("asian", "overlap"),
    ]
    if phase == 0:
        sessions_opts = [None, ("london", "overlap"), ("asian",), ("overlap",)]

    # RSI mean reversion
    for rsi, os_, ob, sessions in itertools.product(
        [2, 3, 5, 7, 14] if phase else [2, 3, 7],
        [10, 15, 20, 25, 30],
        [70, 75, 80, 85, 90],
        sessions_opts,
    ):
        if os_ >= 40 or ob <= 60:
            continue
        space.append(("rsi_mr", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))

    # RSI + EMA filter
    for rsi, ema, os_, ob, sessions in itertools.product(
        [2, 3, 7, 14],
        [20, 50, 100, 200] if phase else [50, 200],
        [15, 20, 25, 30],
        [70, 75, 80, 85],
        sessions_opts[:6],
    ):
        space.append(
            (
                "rsi_mr_ema_filter",
                {"rsi": rsi, "ema": ema, "os": os_, "ob": ob, "sessions": sessions},
            )
        )

    # Bollinger MR
    for period, k, sessions in itertools.product(
        [10, 20, 50],
        [1.5, 2.0, 2.5],
        sessions_opts,
    ):
        # column keys use float formatting as created
        space.append(("bb_mr", {"period": period, "k": k, "sessions": sessions}))

    # BB + RSI
    for period, k, rsi, os_, ob, sessions in itertools.product(
        [10, 20],
        [1.5, 2.0, 2.5],
        [2, 3, 7, 14],
        [20, 25, 30],
        [70, 75, 80],
        sessions_opts[:5],
    ):
        space.append(
            (
                "bb_mr_rsi",
                {
                    "period": period,
                    "k": k,
                    "rsi": rsi,
                    "os": os_,
                    "ob": ob,
                    "sessions": sessions,
                },
            )
        )

    # Stochastic
    for period, os_, ob, sessions in itertools.product(
        [5, 14], [15, 20, 25], [75, 80, 85], sessions_opts[:5]
    ):
        space.append(("stoch_mr", {"period": period, "os": os_, "ob": ob, "sessions": sessions}))

    # EMA pullback
    for fast, slow, rsi, sessions in itertools.product(
        [8, 12, 21], [34, 50, 100], [7, 14], sessions_opts[:4]
    ):
        if fast >= slow:
            continue
        space.append(
            (
                "ema_pullback",
                {
                    "fast": fast,
                    "slow": slow,
                    "rsi": rsi,
                    "rsi_min": 45,
                    "rsi_max": 55,
                    "sessions": sessions,
                },
            )
        )

    # EMA stack break
    for a, b, c, sessions in itertools.product(
        [8, 12, 21], [21, 34, 50], [50, 100, 200], sessions_opts[:4]
    ):
        if not (a < b < c):
            continue
        space.append(("ema_stack_break", {"a": a, "b": b, "c": c, "sessions": sessions}))

    # MACD
    for sessions in sessions_opts[:5]:
        space.append(("macd_cross", {"sessions": sessions}))

    # Donchian
    for period, sessions in itertools.product([10, 20, 40, 55], sessions_opts[:4]):
        space.append(("donchian_break", {"period": period, "sessions": sessions}))

    # NR breakout
    for look, q, sessions in itertools.product([5, 10, 20], [0.2, 0.3], sessions_opts[:3]):
        space.append(("nr_breakout", {"look": look, "q": q, "sessions": sessions}))

    # PDH/PDL
    for rsi, os_, ob, sessions in itertools.product(
        [2, 3, 7, 14], [20, 25, 30], [70, 75, 80], sessions_opts[:4]
    ):
        space.append(("pdh_pdl_fade", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))
        space.append(("pdh_pdl_break", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))

    # Engulf / pinbar
    for ema, sessions in itertools.product([20, 50, 200], sessions_opts[:4]):
        space.append(("engulf_trend", {"ema": ema, "sessions": sessions}))
    for rsi, wick, body, os_, ob, sessions in itertools.product(
        [2, 3, 7], [0.55, 0.65], [0.35, 0.45], [25, 30], [70, 75], sessions_opts[:3]
    ):
        space.append(
            (
                "pinbar_mr",
                {
                    "rsi": rsi,
                    "wick": wick,
                    "body": body,
                    "os": os_,
                    "ob": ob,
                    "sessions": sessions,
                },
            )
        )

    # Triple / quad confirm (high-WR candidates)
    for rsi, bb, k, stoch, os_, ob, sessions in itertools.product(
        [2, 3, 7],
        [10, 20],
        [1.5, 2.0, 2.5],
        [5, 14],
        [20, 25, 30],
        [70, 75, 80],
        sessions_opts[:5],
    ):
        space.append(
            (
                "triple_confirm_mr",
                {
                    "rsi": rsi,
                    "bb": bb,
                    "k": k,
                    "stoch": stoch,
                    "os": os_,
                    "ob": ob,
                    "sessions": sessions,
                },
            )
        )
        if phase >= 1:
            for ema in (50, 100, 200):
                space.append(
                    (
                        "quad_confirm_mr",
                        {
                            "rsi": rsi,
                            "bb": bb,
                            "k": k,
                            "stoch": stoch,
                            "ema": ema,
                            "os": os_,
                            "ob": ob,
                            "sessions": sessions,
                        },
                    )
                )

    # Asian range fade
    for look, rsi, os_, ob, sessions in itertools.product(
        [20, 32, 40],
        [2, 3, 7],
        [20, 25, 30],
        [70, 75, 80],
        [("london",), ("london", "overlap"), ("overlap",), None],
    ):
        space.append(
            (
                "asian_range_fade",
                {"look": look, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions},
            )
        )

    # Extra filters via copies
    if phase >= 2:
        extras = []
        for kind, p in space:
            for hours in [None, (8, 9, 10, 11, 12, 13, 14, 15), (13, 14, 15, 16), (0, 1, 2, 3, 4, 5, 6, 7)]:
                for vol in [False, True]:
                    pp = dict(p)
                    if hours:
                        pp["hours"] = hours
                    if vol:
                        pp["vol_ok"] = True
                    extras.append((kind, pp))
        space.extend(extras)

    RNG.shuffle(space)
    return space


def save_progress(stats: dict, best: list[Result]) -> None:
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    # keep top 50
    best_sorted = sorted(best, key=lambda r: r.score, reverse=True)[:50]
    with open(OUT / "top50.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in best_sorted], f, indent=2)


def append_candidate(r: Result) -> None:
    with open(BEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(r)) + "\n")


def try_combo_phase(
    dfs: dict[str, pd.DataFrame],
    top: list[Result],
    tested: int,
    best: list[Result],
    stats: dict,
) -> tuple[int, Result | None]:
    """Combine top signal logics with AND/OR on same timeframe."""
    winner = None
    # rebuild signals for top names is hard without storing masks; instead combine by re-gen
    by_tf: dict[str, list[Result]] = {}
    for r in top:
        by_tf.setdefault(r.timeframe, []).append(r)

    for tf, items in by_tf.items():
        df = dfs[tf]
        items = items[:20]
        for a, b in itertools.combinations(items, 2):
            if a.name.split("|")[0] == b.name.split("|")[0]:
                continue
            # parse kind/params from stored params
            try:
                kind_a = a.params["kind"]
                kind_b = b.params["kind"]
                pa = {k: v for k, v in a.params.items() if k not in ("kind", "exit")}
                pb = {k: v for k, v in b.params.items() if k not in ("kind", "exit")}
                la, sa = gen_signals(df, kind_a, pa)
                lb, sb = gen_signals(df, kind_b, pb)
            except Exception:
                continue
            for mode in ("combo_and", "combo_or"):
                long_e = (la & lb) if mode == "combo_and" else (la | lb)
                short_e = (sa & sb) if mode == "combo_and" else (sa | sb)
                for exit_p in [a.params["exit"], b.params["exit"]] + exit_grid(0)[:12]:
                    tested += 1
                    name = f"{mode}|{a.name}++{b.name}"
                    res = evaluate(
                        name,
                        tf,
                        df,
                        long_e,
                        short_e,
                        exit_p,
                        {"kind": mode, "a": a.name, "b": b.name},
                    )
                    if not res:
                        continue
                    if res.score > 0:
                        best.append(res)
                    if (
                        res.winrate >= 0.75
                        and (res.oos_winrate or 0) >= 0.7
                        and res.profit_factor >= 1.1
                    ):
                        append_candidate(res)
                    if meets_target(
                        {
                            "trades": res.trades,
                            "winrate": res.winrate,
                            "profit_factor": res.profit_factor,
                            "net_pnl": res.net_pnl,
                        },
                        {
                            "trades": res.oos_trades or 0,
                            "winrate": res.oos_winrate or 0,
                            "profit_factor": res.oos_profit_factor or 0,
                            "net_pnl": res.oos_net_pnl or 0,
                        },
                    ):
                        winner = res
                        return tested, winner
            if tested % 200 == 0:
                stats.update({"tested": tested, "phase": "combo", "best_score": max((x.score for x in best), default=-1e9)})
                save_progress(stats, best)
                log(f"combo tested={tested} best_score={stats['best_score']:.2f}")
    return tested, winner


def main() -> None:
    log("=== STRATEGY SEARCH START ===")
    log(
        f"Targets: IS WR>={TARGET_WR:.0%} PF>={TARGET_PF} trades>={MIN_TRADES} | "
        f"OOS WR>={OOS_WR:.0%} PF>={OOS_PF} trades>={OOS_MIN_TRADES}"
    )
    frames_raw = load_frames()
    dfs = {k: add_indicators(v) for k, v in frames_raw.items()}
    for k, d in dfs.items():
        log(f"Indicators ready for {k}: {d.shape}")

    tested = 0
    best: list[Result] = []
    winner: Result | None = None
    stats = {
        "tested": 0,
        "phase": 0,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_wr": TARGET_WR,
        "best_score": -1e9,
        "best_wr": 0,
        "best_oos_wr": 0,
    }

    phase = 0
    while winner is None:
        log(f"----- PHASE {phase} -----")
        space = strategy_param_space(phase)
        exits = exit_grid(phase)
        log(f"Param space={len(space)} exit_grid={len(exits)} approx_evals~{len(space)*min(len(exits), 40)}")

        # For each strategy, sample exits (full grid gets huge); escalate with phase
        max_exits = 8 if phase == 0 else (16 if phase == 1 else 40)
        for idx, (kind, params) in enumerate(space):
            for tf, df in dfs.items():
                # Skip 5m for early phases to save time, expand later
                if phase == 0 and tf == "5m":
                    continue
                try:
                    long_e, short_e = gen_signals(df, kind, params)
                except Exception:
                    continue
                if long_e.sum() + short_e.sum() < 15:
                    continue

                # Rank a quick exit probe then deepen promising ones
                probe_exits = exits[:: max(1, len(exits) // max_exits)][:max_exits]
                local_best = None
                for exit_p in probe_exits:
                    tested += 1
                    name = f"{kind}|{tf}|p{phase}"
                    res = evaluate(
                        name,
                        tf,
                        df,
                        long_e,
                        short_e,
                        exit_p,
                        {"kind": kind, **params},
                    )
                    if not res:
                        continue
                    if local_best is None or res.score > local_best.score:
                        local_best = res
                    if res.score > stats["best_score"]:
                        stats["best_score"] = res.score
                        stats["best_wr"] = res.winrate
                        stats["best_oos_wr"] = res.oos_winrate or 0
                        log(
                            f"NEW BEST score={res.score:.1f} WR={res.winrate:.1%} "
                            f"OOS_WR={res.oos_winrate:.1%} PF={res.profit_factor:.2f} "
                            f"tr={res.trades}/{res.oos_trades} {res.name} {res.params}"
                        )
                        append_candidate(res)
                        best.append(res)
                        save_progress(stats, best)
                    elif res.winrate >= 0.75 and (res.oos_winrate or 0) >= 0.7:
                        append_candidate(res)
                        best.append(res)

                    if meets_target(
                        {
                            "trades": res.trades,
                            "winrate": res.winrate,
                            "profit_factor": res.profit_factor,
                            "net_pnl": res.net_pnl,
                        },
                        {
                            "trades": res.oos_trades or 0,
                            "winrate": res.oos_winrate or 0,
                            "profit_factor": res.oos_profit_factor or 0,
                            "net_pnl": res.oos_net_pnl or 0,
                        },
                    ):
                        winner = res
                        break
                if winner:
                    break

                # Deepen exits for promising local best
                if local_best and local_best.winrate >= 0.7 and (local_best.oos_winrate or 0) >= 0.65:
                    for exit_p in exits:
                        tested += 1
                        res = evaluate(
                            f"{kind}|{tf}|deep",
                            tf,
                            df,
                            long_e,
                            short_e,
                            exit_p,
                            {"kind": kind, **params},
                        )
                        if not res:
                            continue
                        if res.score > stats["best_score"]:
                            stats["best_score"] = res.score
                            stats["best_wr"] = res.winrate
                            stats["best_oos_wr"] = res.oos_winrate or 0
                            log(
                                f"NEW BEST score={res.score:.1f} WR={res.winrate:.1%} "
                                f"OOS_WR={res.oos_winrate:.1%} PF={res.profit_factor:.2f} "
                                f"tr={res.trades}/{res.oos_trades} {res.name}"
                            )
                            append_candidate(res)
                            best.append(res)
                        if meets_target(
                            {
                                "trades": res.trades,
                                "winrate": res.winrate,
                                "profit_factor": res.profit_factor,
                                "net_pnl": res.net_pnl,
                            },
                            {
                                "trades": res.oos_trades or 0,
                                "winrate": res.oos_winrate or 0,
                                "profit_factor": res.oos_profit_factor or 0,
                                "net_pnl": res.oos_net_pnl or 0,
                            },
                        ):
                            winner = res
                            break
                if winner:
                    break

            if winner:
                break
            if idx % 25 == 0:
                stats.update({"tested": tested, "phase": phase})
                save_progress(stats, best)
                top_wr = max((b.winrate for b in best), default=0)
                top_oos = max((b.oos_winrate or 0 for b in best), default=0)
                log(
                    f"phase={phase} idx={idx}/{len(space)} tested={tested} "
                    f"best_score={stats['best_score']:.1f} best_wr={top_wr:.1%} best_oos_wr={top_oos:.1%}"
                )

        if winner:
            break

        # Combination phase using top candidates
        log("Running combination search on top candidates...")
        top = sorted(best, key=lambda r: r.score, reverse=True)[:40]
        tested, winner = try_combo_phase(dfs, top, tested, best, stats)
        if winner:
            break

        # Tighten targets slightly? No — keep absolute. Expand search space.
        phase += 1
        # Mutate thresholds randomly in later phases via additional param noise
        if phase >= 3:
            log("Phase>=3: random mutation search around top candidates")
            for r in sorted(best, key=lambda x: x.score, reverse=True)[:30]:
                tf = r.timeframe
                df = dfs[tf]
                base = {k: v for k, v in r.params.items() if k not in ("kind", "exit", "a", "b")}
                kind = r.params.get("kind")
                if not kind or kind.startswith("combo"):
                    continue
                for _ in range(40):
                    mp = dict(base)
                    # jitter numeric thresholds
                    for key in list(mp.keys()):
                        if key in ("os",) and isinstance(mp[key], (int, float)):
                            mp[key] = int(min(35, max(5, mp[key] + RNG.randint(-5, 5))))
                        if key in ("ob",) and isinstance(mp[key], (int, float)):
                            mp[key] = int(min(95, max(65, mp[key] + RNG.randint(-5, 5))))
                    try:
                        long_e, short_e = gen_signals(df, kind, mp)
                    except Exception:
                        continue
                    for exit_p in exit_grid(2)[:20]:
                        tested += 1
                        res = evaluate(
                            f"mutate|{kind}|{tf}",
                            tf,
                            df,
                            long_e,
                            short_e,
                            exit_p,
                            {"kind": kind, **mp},
                        )
                        if not res:
                            continue
                        if res.score > stats["best_score"]:
                            stats["best_score"] = res.score
                            append_candidate(res)
                            best.append(res)
                            log(
                                f"NEW BEST mutate score={res.score:.1f} WR={res.winrate:.1%} "
                                f"OOS_WR={res.oos_winrate:.1%} PF={res.profit_factor:.2f}"
                            )
                        if meets_target(
                            {
                                "trades": res.trades,
                                "winrate": res.winrate,
                                "profit_factor": res.profit_factor,
                                "net_pnl": res.net_pnl,
                            },
                            {
                                "trades": res.oos_trades or 0,
                                "winrate": res.oos_winrate or 0,
                                "profit_factor": res.oos_profit_factor or 0,
                                "net_pnl": res.oos_net_pnl or 0,
                            },
                        ):
                            winner = res
                            break
                    if winner:
                        break
                if winner:
                    break
        if winner:
            break

        stats.update({"tested": tested, "phase": phase})
        save_progress(stats, best)
        log(f"Phase {phase-1} complete. tested={tested}. Continuing...")

        # Safety: never stop — loop forever until winner or process killed
        if phase > 100:
            # reset phase extras but keep searching with reshuffled random exits
            phase = 2

    assert winner is not None
    log("===== WINNER FOUND =====")
    log(json.dumps(asdict(winner), indent=2))
    with open(WINNER_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(winner), f, indent=2)
    stats.update({"tested": tested, "winner": asdict(winner), "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    save_progress(stats, best + [winner])
    log("Saved WINNER.json — search complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
