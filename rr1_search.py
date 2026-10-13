#!/usr/bin/env python3
"""
Never-stop CAUSAL searcher for:
  - Risk:Reward = 1:1  (tp_atr == sl_atr always)
  - Winrate >= 80% IS and OOS
  - >= 1 trade per gold trading day
  - Profit factor >= 1.2, net PnL > 0 both windows

Causality:
  - wick vs prior-bar BB/EMA/Stoch (.shift(1))
  - range levels .shift(1)
  - signal on close[i] → fill open[i+1]

Runs until killed. Writes rr1_search_results/ continuously.
"""

from __future__ import annotations

import itertools
import json
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("rr1_search_results")
OUT.mkdir(exist_ok=True)
LOG = OUT / "search.log"
CHAMPION = OUT / "CHAMPION.json"
CANDIDATES = OUT / "candidates.jsonl"
PROGRESS = OUT / "progress.json"
BOT_PATH = Path("live_rr1_bot.py")

START_CAP = 500.0
RISK_PCT = 0.02
MAX_RISK = 20.0
HALT = START_CAP * 0.85

TARGET_WR = 0.80
TARGET_PF = 1.20
MIN_TPD = 1.0
MIN_IS = 50
MIN_OOS = 25
OOS_WR = 0.80
OOS_PF = 1.15
OOS_TPD = 0.85

RNG = random.Random(11)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for key, interval, period in [("5m", "5m", "60d"), ("15m", "15m", "60d"), ("1h", "1h", "730d")]:
        df = yf.download("GC=F", interval=interval, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        if len(df) > 500:
            frames[key] = df
            log(f"Loaded {key}: {len(df)} {df.index[0]} → {df.index[-1]}")
    if not frames:
        raise RuntimeError("no data")
    return frames


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    for span in (5, 8, 9, 12, 20, 21, 34, 50, 100, 200):
        d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
        d[f"SMA_{span}"] = c.rolling(span).mean()
    for period in (2, 3, 5, 7, 9, 14, 21):
        delta = c.diff()
        ag = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        al = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        d[f"RSI_{period}"] = (100 - (100 / (1 + ag / al.replace(0, np.nan)))).fillna(50)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    d["ATR_7"] = tr.ewm(com=6, adjust=False).mean()
    d["ATR_14"] = tr.ewm(com=13, adjust=False).mean()
    for period, k in ((10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5), (50, 2.0)):
        mid = c.rolling(period).mean()
        std = c.rolling(period).std()
        d[f"BB_UP_{period}_{k}"] = mid + k * std
        d[f"BB_LO_{period}_{k}"] = mid - k * std
        d[f"BB_MID_{period}_{k}"] = mid
    for period in (5, 9, 14):
        ll = l.rolling(period).min()
        hh = h.rolling(period).max()
        d[f"STOCH_K_{period}"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        d[f"STOCH_D_{period}"] = d[f"STOCH_K_{period}"].rolling(3).mean()
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIG"]
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
    body = (c - d["Open"]).abs()
    rng = (h - l).replace(0, np.nan)
    d["lower_wick"] = (np.minimum(c, d["Open"]) - l) / rng
    d["upper_wick"] = (h - np.maximum(c, d["Open"])) / rng
    d["body_pct"] = body / rng
    d["bull"] = c > d["Open"]
    d["bear"] = c < d["Open"]
    d["range"] = h - l
    d["vol_ok"] = d["range"] > 0.4 * d["range"].rolling(20).mean()
    # Donchian prior
    for period in (10, 20, 40):
        d[f"DON_HI_{period}"] = h.rolling(period).max().shift(1)
        d[f"DON_LO_{period}"] = l.rolling(period).min().shift(1)
    return d


def trading_days(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 1.0
    return max(1.0, ((idx[-1] - idx[0]).days + 1) * 5.5 / 7.0)


def simulate(
    df: pd.DataFrame,
    long_e: np.ndarray,
    short_e: np.ndarray,
    rr_atr: float,
    max_hold: int,
    start_i: int,
    end_i: int,
    atr_col: str = "ATR_14",
    allow_long: bool = True,
    allow_short: bool = True,
) -> dict:
    """1:1 only: SL = TP = rr_atr * ATR."""
    n = len(df)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    atrs = df[atr_col].to_numpy(float)
    times = df.index

    capital = START_CAP
    peak = START_CAP
    max_dd = 0.0
    pnls: list[float] = []
    i = start_i
    while i < end_i and i + 1 < n:
        if capital < HALT:
            break
        gl = allow_long and bool(long_e[i])
        gs = allow_short and bool(short_e[i])
        if gl == gs:
            i += 1
            continue
        direction = "LONG" if gl else "SHORT"
        atr = atrs[i]
        entry = opens[i + 1]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
            i += 1
            continue
        risk = min(MAX_RISK, capital * RISK_PCT)
        dist = atr * rr_atr
        if dist <= 0:
            i += 1
            continue
        qty = risk / dist
        sl = entry - dist if direction == "LONG" else entry + dist
        tp = entry + dist if direction == "LONG" else entry - dist  # 1:1

        j = i + 1
        bars = 0
        exit_p = entry
        hit = False
        while j < n and bars < max_hold:
            hi, lo = highs[j], lows[j]
            bars += 1
            if direction == "LONG":
                # conservative: SL before TP if both
                if lo <= sl:
                    exit_p, hit = sl, True
                    break
                if hi >= tp:
                    exit_p, hit = tp, True
                    break
            else:
                if hi >= sl:
                    exit_p, hit = sl, True
                    break
                if lo <= tp:
                    exit_p, hit = tp, True
                    break
            j += 1
        if not hit:
            exit_p = closes[min(j, n - 1)]
        pnl = (exit_p - entry) * qty if direction == "LONG" else (entry - exit_p) * qty
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0)
        pnls.append(pnl)
        i = j + 1 if j > i else i + 1

    if not pnls:
        return {
            "trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "final_equity": capital,
            "max_dd": max_dd,
            "trades_per_day": 0.0,
            "expectancy": 0.0,
        }
    arr = np.array(pnls, float)
    wins, losses = arr[arr > 0], arr[arr <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
    tdays = trading_days(times[start_i:end_i])
    return {
        "trades": int(len(arr)),
        "winrate": float((arr > 0).mean()),
        "profit_factor": float(pf),
        "net_pnl": float(arr.sum()),
        "final_equity": float(capital),
        "max_dd": float(max_dd),
        "trades_per_day": float(len(arr) / tdays),
        "expectancy": float(arr.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def sess_mask(df, sessions):
    if not sessions:
        return np.ones(len(df), dtype=bool)
    return df["session"].isin(sessions).to_numpy()


def gen_signals(df: pd.DataFrame, kind: str, p: dict):
    c = df["Close"]
    sess = sess_mask(df, p.get("sessions"))

    if kind == "rsi_cross":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (rsi < p["os"]) & (rsi.shift(1) >= p["os"]) & sess
        short_e = (rsi > p["ob"]) & (rsi.shift(1) <= p["ob"]) & sess

    elif kind == "rsi_level":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (rsi < p["os"]) & (rsi.shift(1) >= p["os"] - p.get("band", 5)) & sess
        short_e = (rsi > p["ob"]) & (rsi.shift(1) <= p["ob"] + p.get("band", 5)) & sess

    elif kind == "bb_fade":
        lo = df[f"BB_LO_{p['period']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['period']}_{p['k']}"].shift(1)
        long_e = (df["Low"] <= lo) & (c > lo) & sess
        short_e = (df["High"] >= up) & (c < up) & sess

    elif kind == "bb_rsi_fade":
        lo = df[f"BB_LO_{p['period']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['period']}_{p['k']}"].shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= lo) & (c > lo) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= up) & (c < up) & (rsi > p["ob"]) & sess

    elif kind == "bb_mid_reject":
        # touch outer band then close back inside mid side
        lo = df[f"BB_LO_{p['period']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['period']}_{p['k']}"].shift(1)
        mid = df[f"BB_MID_{p['period']}_{p['k']}"].shift(1)
        long_e = (df["Low"] <= lo) & (c > mid) & sess
        short_e = (df["High"] >= up) & (c < mid) & sess

    elif kind == "stoch_cross":
        k = df[f"STOCH_K_{p['period']}"]
        d = df[f"STOCH_D_{p['period']}"]
        long_e = (k < p["os"]) & (k > d) & (k.shift(1) <= d.shift(1)) & sess
        short_e = (k > p["ob"]) & (k < d) & (k.shift(1) >= d.shift(1)) & sess

    elif kind == "triple_fade":
        lo = df[f"BB_LO_{p['bb']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['bb']}_{p['k']}"].shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        sk = df[f"STOCH_K_{p['stoch']}"].shift(1)
        long_e = (df["Low"] <= lo) & (c > lo) & (rsi < p["os"]) & (sk < p["os"]) & sess
        short_e = (df["High"] >= up) & (c < up) & (rsi > p["ob"]) & (sk > p["ob"]) & sess

    elif kind == "ema_wick":
        ema = df[f"EMA_{p['ema']}"].shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= ema) & (c > ema) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= ema) & (c < ema) & (rsi > p["ob"]) & sess

    elif kind == "ema_stack_pull":
        a = df[f"EMA_{p['a']}"].shift(1)
        b = df[f"EMA_{p['b']}"].shift(1)
        e = df[f"EMA_{p['c']}"].shift(1)
        long_e = (a > b) & (b > e) & (df["Low"] <= a) & (c > a) & sess
        short_e = (a < b) & (b < e) & (df["High"] >= a) & (c < a) & sess

    elif kind == "range_fade":
        look = p["look"]
        a_hi = df["High"].rolling(look).max().shift(1)
        a_lo = df["Low"].rolling(look).min().shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= a_lo) & (c > a_lo) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= a_hi) & (c < a_hi) & (rsi > p["ob"]) & sess

    elif kind == "micro_reclaim":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] < df["Low"].shift(1)) & (c > df["Low"].shift(1)) & (rsi < p["os"]) & sess
        short_e = (df["High"] > df["High"].shift(1)) & (c < df["High"].shift(1)) & (rsi > p["ob"]) & sess

    elif kind == "pin_rsi":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["lower_wick"] > p["wick"]) & (df["body_pct"] < p["body"]) & (rsi < p["os"]) & sess
        short_e = (df["upper_wick"] > p["wick"]) & (df["body_pct"] < p["body"]) & (rsi > p["ob"]) & sess

    elif kind == "engulf":
        ema = df[f"EMA_{p['ema']}"].shift(1)
        eng_bull = df["bull"] & df["bear"].shift(1) & (c >= df["Open"].shift(1)) & (df["Open"] <= c.shift(1))
        eng_bear = df["bear"] & df["bull"].shift(1) & (c <= df["Open"].shift(1)) & (df["Open"] >= c.shift(1))
        long_e = eng_bull & (c > ema) & sess
        short_e = eng_bear & (c < ema) & sess

    elif kind == "macd_cross":
        long_e = (df["MACD"] > df["MACD_SIG"]) & (df["MACD"].shift(1) <= df["MACD_SIG"].shift(1)) & sess
        short_e = (df["MACD"] < df["MACD_SIG"]) & (df["MACD"].shift(1) >= df["MACD_SIG"].shift(1)) & sess

    elif kind == "donch_fade":
        hi = df[f"DON_HI_{p['period']}"]
        lo = df[f"DON_LO_{p['period']}"]
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= lo) & (c > lo) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= hi) & (c < hi) & (rsi > p["ob"]) & sess

    elif kind == "combo_and":
        long_e = p["long_a"] & p["long_b"]
        short_e = p["short_a"] & p["short_b"]

    else:
        raise ValueError(kind)

    long_e = long_e.fillna(False).to_numpy(dtype=bool, copy=True) if hasattr(long_e, "fillna") else np.array(long_e, dtype=bool, copy=True)
    short_e = short_e.fillna(False).to_numpy(dtype=bool, copy=True) if hasattr(short_e, "fillna") else np.array(short_e, dtype=bool, copy=True)
    if p.get("vol_ok"):
        vok = df["vol_ok"].fillna(False).to_numpy()
        long_e &= vok
        short_e &= vok
    if p.get("hours"):
        hm = df["hour"].isin(p["hours"]).to_numpy()
        long_e &= hm
        short_e &= hm
    if p.get("dows"):
        dm = df["dow"].isin(p["dows"]).to_numpy()
        long_e &= dm
        short_e &= dm
    return long_e, short_e


@dataclass
class Result:
    name: str
    timeframe: str
    winrate: float
    profit_factor: float
    trades: int
    trades_per_day: float
    net_pnl: float
    max_dd: float
    params: dict
    oos_winrate: float
    oos_profit_factor: float
    oos_trades: int
    oos_trades_per_day: float
    oos_net_pnl: float
    score: float
    rr: float = 1.0


def qualifies(is_m, oos_m) -> bool:
    return (
        is_m["trades"] >= MIN_IS
        and oos_m["trades"] >= MIN_OOS
        and is_m["winrate"] >= TARGET_WR
        and oos_m["winrate"] >= OOS_WR
        and is_m["profit_factor"] >= TARGET_PF
        and oos_m["profit_factor"] >= OOS_PF
        and is_m["net_pnl"] > 0
        and oos_m["net_pnl"] > 0
        and is_m["trades_per_day"] >= MIN_TPD
        and oos_m["trades_per_day"] >= OOS_TPD
    )


def score_of(is_m, oos_m) -> float:
    if is_m["trades"] < 20:
        return -1e9
    # Must be near target or get crushed
    if is_m["winrate"] < 0.75 or is_m["trades_per_day"] < 0.8 or is_m["net_pnl"] <= 0:
        pen = -3000
    else:
        pen = 0
    s = (
        pen
        + is_m["winrate"] * 150
        + oos_m["winrate"] * 180
        + min(is_m["profit_factor"], 8) * 25
        + min(oos_m["profit_factor"], 8) * 30
        + min(is_m["trades_per_day"], 6) * 20
        + min(oos_m["trades_per_day"], 6) * 20
        - is_m["max_dd"] * 50
    )
    if qualifies(is_m, oos_m):
        s += 10000
    return float(s)


def evaluate(name, tf, df, long_e, short_e, exit_p, params) -> Result | None:
    n = len(df)
    split = int(n * 0.7)
    common = dict(
        long_e=long_e,
        short_e=short_e,
        rr_atr=exit_p["rr"],  # SL=TP=rr*ATR
        max_hold=exit_p["max_hold"],
        atr_col=exit_p.get("atr", "ATR_14"),
        allow_long=exit_p.get("allow_long", True),
        allow_short=exit_p.get("allow_short", True),
    )
    is_m = simulate(df, start_i=250, end_i=split, **common)
    oos_m = simulate(df, start_i=split, end_i=n - 2, **common)
    if is_m["trades"] < 20:
        return None
    return Result(
        name=name,
        timeframe=tf,
        winrate=is_m["winrate"],
        profit_factor=is_m["profit_factor"],
        trades=is_m["trades"],
        trades_per_day=is_m["trades_per_day"],
        net_pnl=is_m["net_pnl"],
        max_dd=is_m["max_dd"],
        params={**params, "exit": exit_p},
        oos_winrate=oos_m["winrate"],
        oos_profit_factor=oos_m["profit_factor"],
        oos_trades=oos_m["trades"],
        oos_trades_per_day=oos_m["trades_per_day"],
        oos_net_pnl=oos_m["net_pnl"],
        score=score_of(is_m, oos_m),
        rr=1.0,
    )


def param_space(phase: int):
    space = []
    sessions_opts = [
        None,
        ("asian",),
        ("london",),
        ("overlap",),
        ("ny",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
        ("asian", "london", "overlap"),
        ("asian", "london", "overlap", "ny"),
    ]
    for rsi, os_, ob, sessions in itertools.product(
        [2, 3, 5, 7, 9, 14],
        [10, 15, 20, 25, 30, 35],
        [65, 70, 75, 80, 85, 90],
        sessions_opts,
    ):
        if os_ >= 40 or ob <= 60:
            continue
        space.append(("rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))
        space.append(("rsi_level", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions, "band": 5}))

    for period, k, sessions in itertools.product([10, 20, 50], [1.5, 2.0, 2.5], sessions_opts):
        space.append(("bb_fade", {"period": period, "k": k, "sessions": sessions}))
        space.append(("bb_mid_reject", {"period": period, "k": k, "sessions": sessions}))

    for period, k, rsi, os_, ob, sessions in itertools.product(
        [10, 20], [1.5, 2.0, 2.5], [2, 3, 5, 7], [20, 25, 30, 35], [65, 70, 75, 80], sessions_opts[:8]
    ):
        space.append(
            ("bb_rsi_fade", {"period": period, "k": k, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions})
        )

    for period, os_, ob, sessions in itertools.product(
        [5, 9, 14], [15, 20, 25, 30], [70, 75, 80, 85], sessions_opts[:8]
    ):
        space.append(("stoch_cross", {"period": period, "os": os_, "ob": ob, "sessions": sessions}))

    for bb, k, rsi, stoch, os_, ob, sessions in itertools.product(
        [10, 20], [1.5, 2.0], [2, 3, 7], [5, 14], [20, 25, 30], [70, 75, 80], sessions_opts[:6]
    ):
        space.append(
            (
                "triple_fade",
                {"bb": bb, "k": k, "rsi": rsi, "stoch": stoch, "os": os_, "ob": ob, "sessions": sessions},
            )
        )

    for ema, rsi, os_, ob, sessions in itertools.product(
        [8, 12, 20, 50], [2, 3, 7], [25, 30, 35], [65, 70, 75], sessions_opts[:7]
    ):
        space.append(("ema_wick", {"ema": ema, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))

    for a, b, c, sessions in itertools.product([5, 8, 12], [20, 21, 34], [50, 100, 200], sessions_opts[:5]):
        if a < b < c:
            space.append(("ema_stack_pull", {"a": a, "b": b, "c": c, "sessions": sessions}))

    for look, rsi, os_, ob, sessions in itertools.product(
        [6, 12, 20, 32, 40], [2, 3, 5, 7], [20, 25, 30, 35], [65, 70, 75, 80], sessions_opts[:7]
    ):
        space.append(("range_fade", {"look": look, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))
        space.append(("donch_fade", {"period": look if look in (10, 20, 40) else 20, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))

    for rsi, os_, ob, sessions in itertools.product(
        [2, 3, 5, 7], [20, 25, 30, 35], [65, 70, 75, 80], sessions_opts[:7]
    ):
        space.append(("micro_reclaim", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))

    for rsi, wick, body, os_, ob, sessions in itertools.product(
        [2, 3, 7], [0.5, 0.6, 0.7], [0.3, 0.4, 0.5], [25, 30], [70, 75], sessions_opts[:5]
    ):
        space.append(
            ("pin_rsi", {"rsi": rsi, "wick": wick, "body": body, "os": os_, "ob": ob, "sessions": sessions})
        )

    for ema, sessions in itertools.product([20, 50, 200], sessions_opts[:5]):
        space.append(("engulf", {"ema": ema, "sessions": sessions}))

    for sessions in sessions_opts[:6]:
        space.append(("macd_cross", {"sessions": sessions}))

    if phase >= 1:
        extras = []
        for kind, p in space:
            for vol in (False, True):
                for hours in (None, tuple(range(8, 17)), tuple(range(13, 21)), tuple(range(0, 22))):
                    pp = dict(p)
                    if vol:
                        pp["vol_ok"] = True
                    if hours:
                        pp["hours"] = hours
                    extras.append((kind, pp))
        space = extras

    RNG.shuffle(space)
    return space


def exit_grid(phase: int):
    # STRICT 1:1 — single rr distance for both SL and TP
    rrs = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    holds = [2, 3, 4, 6, 8, 12, 18, 24, 36]
    atrs = ["ATR_14", "ATR_7"]
    if phase == 0:
        rrs, holds, atrs = [0.5, 0.8, 1.0, 1.2], [4, 8, 12, 24], ["ATR_14"]
    sides = [
        {"allow_long": True, "allow_short": True},
        {"allow_long": True, "allow_short": False},
        {"allow_long": False, "allow_short": True},
    ]
    if phase == 0:
        sides = sides[:1]
    grids = []
    for rr, mh, atr, side in itertools.product(rrs, holds, atrs, sides):
        grids.append({"rr": rr, "max_hold": mh, "atr": atr, **side})
    return grids


def save_champion(r: Result):
    CHAMPION.write_text(json.dumps(asdict(r), indent=2))
    with open(CANDIDATES, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(r)) + "\n")


def write_bot(r: Result):
    params = {k: v for k, v in r.params.items() if k not in ("kind", "exit")}
    exit_p = r.params["exit"]
    code = f'''#!/usr/bin/env python3
"""AUTO-GENERATED 1:1 RR bot from rr1_search.
IS WR={r.winrate:.1%} OOS WR={r.oos_winrate:.1%} tpd={r.trades_per_day:.2f}/{r.oos_trades_per_day:.2f}
CAUSAL next-open · SL=TP={exit_p['rr']}×ATR
"""
from __future__ import annotations
import json
from pathlib import Path
from rr1_search import load_frames, add_indicators, gen_signals, simulate

KIND = {json.dumps(r.params["kind"])}
TF = {json.dumps(r.timeframe)}
PARAMS = {json.dumps(params, indent=2)}
EXIT = {json.dumps(exit_p, indent=2)}

def main():
    df = add_indicators(load_frames()[TF])
    p = dict(PARAMS)
    if p.get("sessions") is not None:
        p["sessions"] = tuple(p["sessions"])
    if p.get("hours") is not None:
        p["hours"] = tuple(p["hours"])
    if p.get("dows") is not None:
        p["dows"] = tuple(p["dows"])
    L, S = gen_signals(df, KIND, p)
    n = len(df); split = int(n * 0.7)
    common = dict(long_e=L, short_e=S, rr_atr=EXIT["rr"], max_hold=EXIT["max_hold"],
                  atr_col=EXIT.get("atr","ATR_14"),
                  allow_long=EXIT.get("allow_long", True), allow_short=EXIT.get("allow_short", True))
    full = simulate(df, start_i=250, end_i=n-2, **common)
    is_m = simulate(df, start_i=250, end_i=split, **common)
    oos_m = simulate(df, start_i=split, end_i=n-2, **common)
    report = {{"kind": KIND, "tf": TF, "params": PARAMS, "exit": EXIT, "rr": "1:1",
              "full": full, "is": is_m, "oos": oos_m}}
    Path("rr1_search_results").mkdir(exist_ok=True)
    Path("rr1_search_results/bot_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
'''
    BOT_PATH.write_text(code)


def main():
    log("=== RR 1:1 SEARCH START (never stop) ===")
    log(
        f"Targets: WR>={TARGET_WR:.0%} PF>={TARGET_PF} tpd>={MIN_TPD} "
        f"OOS WR>={OOS_WR:.0%} | STRICT SL=TP | causal next_open"
    )
    frames = {k: add_indicators(v) for k, v in load_frames().items()}
    tf_order = [t for t in ("5m", "15m", "1h") if t in frames]

    champion: Result | None = None
    tested = 0
    phase = 0
    qualified_count = 0

    while True:
        log(f"----- PHASE {phase} -----")
        space = param_space(phase)
        exits = exit_grid(phase)
        max_exits = 12 + phase * 4
        log(f"space={len(space)} exits={len(exits)}")

        for idx, (kind, params) in enumerate(space):
            for tf in tf_order:
                if phase == 0 and tf == "1h":
                    continue
                df = frames[tf]
                try:
                    long_e, short_e = gen_signals(df, kind, params)
                except Exception:
                    continue
                sig_n = int(long_e.sum() + short_e.sum())
                if sig_n < trading_days(df.index) * 0.7:
                    continue

                probe = exits[:: max(1, len(exits) // max_exits)][:max_exits]
                local = None
                for exit_p in probe:
                    tested += 1
                    res = evaluate(
                        f"{kind}|{tf}|p{phase}",
                        tf,
                        df,
                        long_e,
                        short_e,
                        exit_p,
                        {"kind": kind, **params},
                    )
                    if not res:
                        continue
                    if local is None or res.score > local.score:
                        local = res
                    q = qualifies(
                        {
                            "trades": res.trades,
                            "winrate": res.winrate,
                            "profit_factor": res.profit_factor,
                            "net_pnl": res.net_pnl,
                            "trades_per_day": res.trades_per_day,
                            "max_dd": res.max_dd,
                        },
                        {
                            "trades": res.oos_trades,
                            "winrate": res.oos_winrate,
                            "profit_factor": res.oos_profit_factor,
                            "net_pnl": res.oos_net_pnl,
                            "trades_per_day": res.oos_trades_per_day,
                        },
                    )
                    if champion is None or res.score > champion.score:
                        if res.winrate >= 0.70 and res.trades_per_day >= 0.7:
                            champion = res
                            save_champion(res)
                            write_bot(res)
                            if q:
                                qualified_count += 1
                            log(
                                f"{'QUALIFIED ' if q else ''}NEW CHAMP "
                                f"WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f} "
                                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                                f"tr={res.trades}/{res.oos_trades} "
                                f"pnl={res.net_pnl:.1f}/{res.oos_net_pnl:.1f} "
                                f"rr=1:1@{exit_p['rr']}ATR {res.name}"
                            )

                if local and local.winrate >= 0.75 and local.trades_per_day >= 0.9:
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
                        if champion is None or res.score > champion.score:
                            if res.winrate >= 0.70:
                                champion = res
                                save_champion(res)
                                write_bot(res)
                                log(
                                    f"NEW CHAMP deep WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                    f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                                    f"PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f} rr=1:1"
                                )

            if idx % 25 == 0:
                prog = {
                    "tested": tested,
                    "phase": phase,
                    "qualified_count": qualified_count,
                    "champion": asdict(champion) if champion else None,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "target": "WR>=80% tpd>=1 RR=1:1 causal",
                }
                PROGRESS.write_text(json.dumps(prog, indent=2))
                if champion:
                    log(
                        f"phase={phase} idx={idx}/{len(space)} tested={tested} quals={qualified_count} "
                        f"champ WR={champion.winrate:.1%}/{champion.oos_winrate:.1%} "
                        f"tpd={champion.trades_per_day:.2f}/{champion.oos_trades_per_day:.2f} "
                        f"score={champion.score:.0f}"
                    )
                else:
                    log(f"phase={phase} idx={idx}/{len(space)} tested={tested} no champ yet")

        # mutation around champion
        if champion and champion.params.get("kind") not in ("combo_and",):
            log("Mutation pass...")
            kind = champion.params["kind"]
            tf = champion.timeframe
            df = frames[tf]
            base = {k: v for k, v in champion.params.items() if k not in ("kind", "exit")}
            for _ in range(300):
                mp = dict(base)
                for key in ("os", "ob"):
                    if key in mp and isinstance(mp[key], (int, float)):
                        mp[key] = int(mp[key] + RNG.randint(-6, 6))
                if "os" in mp:
                    mp["os"] = int(min(40, max(5, mp["os"])))
                if "ob" in mp:
                    mp["ob"] = int(min(95, max(60, mp["ob"])))
                if mp.get("sessions") is not None and not isinstance(mp["sessions"], tuple):
                    mp["sessions"] = tuple(mp["sessions"])
                if mp.get("hours") is not None and not isinstance(mp["hours"], tuple):
                    mp["hours"] = tuple(mp["hours"])
                try:
                    long_e, short_e = gen_signals(df, kind, mp)
                except Exception:
                    continue
                for exit_p in exit_grid(min(phase, 2))[:30]:
                    tested += 1
                    res = evaluate(f"mutate|{kind}|{tf}", tf, df, long_e, short_e, exit_p, {"kind": kind, **mp})
                    if not res:
                        continue
                    if champion is None or res.score > champion.score:
                        champion = res
                        save_champion(res)
                        write_bot(res)
                        log(
                            f"NEW CHAMP mutate WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f}"
                        )

        # combo AND top two signal kinds if we have candidates
        if CANDIDATES.exists() and phase >= 0:
            log("Combo AND pass...")
            rows = []
            for line in CANDIDATES.read_text().splitlines()[-80:]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
            rows = sorted(rows, key=lambda r: r.get("score", -1e9), reverse=True)[:12]
            for a, b in itertools.combinations(rows, 2):
                if a["timeframe"] != b["timeframe"]:
                    continue
                if a["params"].get("kind") == b["params"].get("kind"):
                    continue
                tf = a["timeframe"]
                df = frames[tf]
                try:
                    pa = {k: v for k, v in a["params"].items() if k not in ("kind", "exit")}
                    pb = {k: v for k, v in b["params"].items() if k not in ("kind", "exit")}
                    if pa.get("sessions") is not None:
                        pa["sessions"] = tuple(pa["sessions"])
                    if pb.get("sessions") is not None:
                        pb["sessions"] = tuple(pb["sessions"])
                    la, sa = gen_signals(df, a["params"]["kind"], pa)
                    lb, sb = gen_signals(df, b["params"]["kind"], pb)
                except Exception:
                    continue
                long_e, short_e = la & lb, sa & sb
                if long_e.sum() + short_e.sum() < 30:
                    continue
                for exit_p in exit_grid(0)[:16]:
                    tested += 1
                    res = evaluate(
                        f"and|{a['params']['kind']}+{b['params']['kind']}|{tf}",
                        tf,
                        df,
                        long_e,
                        short_e,
                        exit_p,
                        {"kind": "combo_and", "a": a["name"], "b": b["name"]},
                    )
                    if not res:
                        continue
                    if champion is None or res.score > champion.score:
                        if res.winrate >= 0.7:
                            champion = res
                            save_champion(res)
                            write_bot(res)
                            log(
                                f"NEW CHAMP combo WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f}"
                            )

        phase += 1
        PROGRESS.write_text(
            json.dumps(
                {
                    "tested": tested,
                    "phase": phase,
                    "qualified_count": qualified_count,
                    "champion": asdict(champion) if champion else None,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
        log(f"Phase complete tested={tested} quals={qualified_count}. Looping forever...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
