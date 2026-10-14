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
# Twelve Data XAU sometimes emits near-flat quote bars (ATR≪$1).
# Trading those makes 0.5×ATR TP trivial noise — reject them.
MIN_ATR = 1.5

TARGET_WR = 0.75  # user goal ≥75% on 2y, 1:1, ≥1 tpd
TARGET_PF = 1.15
MIN_TPD = 1.0
MIN_IS = 120
MIN_OOS = 50
OOS_WR = 0.75
OOS_PF = 1.10
OOS_TPD = 0.85

RNG = random.Random(11)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_local_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # support datetime index col or Date
    for col in ("datetime", "Datetime", "date", "Date", "time", "Time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    rename = {c: c.capitalize() if c.lower() in ("open", "high", "low", "close") else c for c in df.columns}
    df = df.rename(columns=rename)
    # handle already-capitalized
    cols = {c.lower(): c for c in df.columns}
    need = {}
    for k in ("open", "high", "low", "close"):
        if k.capitalize() in df.columns:
            need[k.capitalize()] = df[k.capitalize()]
        elif k in cols:
            need[k.capitalize()] = df[cols[k]]
    out = pd.DataFrame(need).dropna().sort_index()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLC to a higher timeframe (right-labeled closed bars)."""
    ohlc = df[["Open", "High", "Low", "Close"]].resample(rule, label="right", closed="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    return ohlc


def align_signal_to_index(signal: np.ndarray | pd.Series, src_index: pd.DatetimeIndex, dst_index: pd.DatetimeIndex) -> np.ndarray:
    """Map HTF boolean signal onto LTF bars using only completed HTF bars (causal).

    Uses prior HTF bar (.shift(1)) then forward-fills onto the destination index.
    """
    s = pd.Series(np.asarray(signal, dtype=bool), index=src_index)
    # only use last fully closed HTF bar
    s = s.shift(1)
    aligned = s.reindex(dst_index, method="ffill").fillna(False)
    return aligned.to_numpy(dtype=bool)


def load_frames(include_resampled: bool = True) -> dict[str, pd.DataFrame]:
    """Prefer long local Twelve Data CSVs in data/, else Yahoo.

    When local 5m exists, also build 15m/30m/1h/2h/4h via resample for MTF search.
    Native 1m is only loaded if a local 1m CSV is present (cannot upsample from 5m).
    """
    frames: dict[str, pd.DataFrame] = {}
    data_dir = Path("data")
    local_map = {
        "1m": sorted(data_dir.glob("XAUUSD_1min_*.csv")) if data_dir.exists() else [],
        "5m": sorted(data_dir.glob("XAUUSD_5min_*.csv")) if data_dir.exists() else [],
        "15m": sorted(data_dir.glob("XAUUSD_15min_*.csv")) if data_dir.exists() else [],
        "1h": sorted(data_dir.glob("XAUUSD_1h_*.csv")) if data_dir.exists() else [],
    }
    for key, files in local_map.items():
        if files:
            df = _load_local_csv(files[-1])
            if len(df) > 500:
                frames[key] = df
                log(f"Loaded LOCAL {key}: {len(df)} {df.index[0]} → {df.index[-1]} ({files[-1].name})")
    # Yahoo fill for missing TFs (short windows — used only if no local 5m)
    if "5m" not in frames:
        for key, interval, period in [("5m", "5m", "60d"), ("15m", "15m", "60d"), ("1h", "1h", "730d")]:
            if key in frames:
                continue
            df = yf.download("GC=F", interval=interval, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close"]].dropna()
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            if len(df) > 500:
                frames[key] = df
                log(f"Loaded Yahoo {key}: {len(df)} {df.index[0]} → {df.index[-1]}")
    # Resample higher TFs from long local 5m (2y coverage)
    if include_resampled and "5m" in frames and (frames["5m"].index[-1] - frames["5m"].index[0]).days >= 200:
        base = frames["5m"]
        for tf, rule in (("15m", "15min"), ("30m", "30min"), ("1h", "1h"), ("2h", "2h"), ("4h", "4h")):
            if tf in frames and len(frames[tf]) > 5000:
                continue  # keep native if already long
            rs = resample_ohlc(base, rule)
            if len(rs) > 400:
                frames[tf] = rs
                log(f"Resampled {tf}: {len(rs)} {rs.index[0]} → {rs.index[-1]} from 5m")
        if "1m" not in frames:
            log("NOTE: no native 1m CSV — cannot upsample; pipeline skips 1m until downloaded")
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
    # --- ICT / SMC building blocks (all lagged for causality) ---
    # Swings from prior bars only
    for sw in (5, 10, 20, 40):
        d[f"SW_HI_{sw}"] = h.rolling(sw).max().shift(1)
        d[f"SW_LO_{sw}"] = l.rolling(sw).min().shift(1)
        # equilibrium / premium-discount of prior swing range
        mid = (d[f"SW_HI_{sw}"] + d[f"SW_LO_{sw}"]) / 2.0
        d[f"EQ_{sw}"] = mid
        rng_s = (d[f"SW_HI_{sw}"] - d[f"SW_LO_{sw}"]).replace(0, np.nan)
        d[f"FIB382_{sw}"] = d[f"SW_HI_{sw}"] - 0.382 * rng_s
        d[f"FIB500_{sw}"] = d[f"SW_HI_{sw}"] - 0.500 * rng_s
        d[f"FIB618_{sw}"] = d[f"SW_HI_{sw}"] - 0.618 * rng_s
        d[f"FIB786_{sw}"] = d[f"SW_HI_{sw}"] - 0.786 * rng_s
        # bullish fibs from swing low up
        d[f"FIB_UP382_{sw}"] = d[f"SW_LO_{sw}"] + 0.382 * rng_s
        d[f"FIB_UP500_{sw}"] = d[f"SW_LO_{sw}"] + 0.500 * rng_s
        d[f"FIB_UP618_{sw}"] = d[f"SW_LO_{sw}"] + 0.618 * rng_s
        d[f"FIB_UP786_{sw}"] = d[f"SW_LO_{sw}"] + 0.786 * rng_s
    # Fair Value Gaps: gap between bar i-2 and i (middle is i-1), lagged one bar
    # bullish FVG when Low[i] > High[i-2]; bearish when High[i] < Low[i-2]
    bull_fvg = l > h.shift(2)
    bear_fvg = h < l.shift(2)
    d["FVG_BULL_BOT"] = h.shift(2).where(bull_fvg)
    d["FVG_BULL_TOP"] = l.where(bull_fvg)
    d["FVG_BEAR_TOP"] = l.shift(2).where(bear_fvg)
    d["FVG_BEAR_BOT"] = h.where(bear_fvg)
    # propagate last known FVG levels forward, then shift so signal bar can't see current formation
    for col in ("FVG_BULL_BOT", "FVG_BULL_TOP", "FVG_BEAR_TOP", "FVG_BEAR_BOT"):
        d[col] = d[col].ffill().shift(1)
    # Order block proxy: last opposite candle before impulse (lagged)
    # bullish OB: bearish candle followed by strong bullish close above its high
    impulse_up = (c > h.shift(1)) & (c > d["Open"]) & ((c - d["Open"]) > 0.6 * d["range"])
    impulse_dn = (c < l.shift(1)) & (c < d["Open"]) & ((d["Open"] - c) > 0.6 * d["range"])
    d["OB_BULL_HI"] = h.shift(1).where(impulse_up & d["bear"].shift(1)).ffill().shift(1)
    d["OB_BULL_LO"] = l.shift(1).where(impulse_up & d["bear"].shift(1)).ffill().shift(1)
    d["OB_BEAR_HI"] = h.shift(1).where(impulse_dn & d["bull"].shift(1)).ffill().shift(1)
    d["OB_BEAR_LO"] = l.shift(1).where(impulse_dn & d["bull"].shift(1)).ffill().shift(1)
    # BOS flags vs prior swing (close break)
    d["BOS_UP_20"] = c > d["SW_HI_20"]
    d["BOS_DN_20"] = c < d["SW_LO_20"]
    d["BOS_UP_40"] = c > d["SW_HI_40"]
    d["BOS_DN_40"] = c < d["SW_LO_40"]

    # --- Public playbook indicators (Ichimoku, HA, Keltner, Supertrend, CCI, ADX, zscore, ORB) ---
    # Ichimoku (standard 9/26/52); cloud lagged so we don't use future span B shift as look-ahead for entry
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)  # classic displacement
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2.0).shift(26)
    d["ICH_TENKAN"] = tenkan
    d["ICH_KIJUN"] = kijun
    d["ICH_SPAN_A"] = span_a
    d["ICH_SPAN_B"] = span_b
    d["ICH_CLOUD_TOP"] = pd.concat([span_a, span_b], axis=1).max(axis=1)
    d["ICH_CLOUD_BOT"] = pd.concat([span_a, span_b], axis=1).min(axis=1)
    # Heikin Ashi (numpy recurrence — fast on 185k bars)
    o_np = d["Open"].to_numpy(float)
    c_np = c.to_numpy(float)
    ha_c = ((o_np + h.to_numpy(float) + l.to_numpy(float) + c_np) / 4.0)
    ha_o = np.empty_like(ha_c)
    ha_o[0] = 0.5 * (o_np[0] + c_np[0])
    for i in range(1, len(ha_c)):
        ha_o[i] = 0.5 * (ha_o[i - 1] + ha_c[i - 1])
    d["HA_OPEN"] = ha_o
    d["HA_CLOSE"] = ha_c
    d["HA_BULL"] = ha_c > ha_o
    d["HA_BEAR"] = ha_c < ha_o
    # Keltner
    for period, mult in ((20, 1.5), (20, 2.0), (10, 1.5)):
        mid = c.ewm(span=period, adjust=False).mean()
        d[f"KEL_MID_{period}_{mult}"] = mid
        d[f"KEL_UP_{period}_{mult}"] = mid + mult * d["ATR_14"]
        d[f"KEL_LO_{period}_{mult}"] = mid - mult * d["ATR_14"]
    # Supertrend-like ATR channel (causal: prior close vs prior bands)
    for mult in (2.0, 3.0):
        hl2 = (h + l) / 2.0
        d[f"ST_UP_{mult}"] = (hl2 + mult * d["ATR_14"]).shift(1)
        d[f"ST_LO_{mult}"] = (hl2 - mult * d["ATR_14"]).shift(1)
    # CCI
    for period in (14, 20):
        tp = (h + l + c) / 3.0
        sma = tp.rolling(period).mean()
        mad = (tp - sma).abs().rolling(period).mean()
        d[f"CCI_{period}"] = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    # ADX / DI
    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)
    tr_s = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["PLUS_DI"] = plus_di
    d["MINUS_DI"] = minus_di
    d["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    # Z-score of close
    for period in (20, 40):
        m = c.rolling(period).mean()
        s = c.rolling(period).std()
        d[f"Z_{period}"] = (c - m) / s.replace(0, np.nan)
    # Session opening range proxy: first UTC hour of Asian/London/NY (causal via shift)
    d["date"] = d.index.date
    for name, start_h in (("ASIAN", 0), ("LONDON", 8), ("NY", 13)):
        in_open = (d["hour"] >= start_h) & (d["hour"] < start_h + 1)
        d[f"OR_HI_{name}"] = h.where(in_open).groupby(d["date"]).cummax().groupby(d["date"]).ffill().shift(1)
        d[f"OR_LO_{name}"] = l.where(in_open).groupby(d["date"]).cummin().groupby(d["date"]).ffill().shift(1)

    return d.copy()  # defragment after many column inserts


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
        if not np.isfinite(atr) or atr < MIN_ATR or not np.isfinite(entry):
            i += 1
            continue
        risk = min(MAX_RISK, capital * RISK_PCT)
        dist = atr * rr_atr
        # Absolute stop floor (USD) — blocks tiny-target noise fills
        if dist < MIN_ATR:
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

    elif kind == "fvg_reclaim":
        # tap into prior FVG then close back through the near edge
        bt, tp = df["FVG_BULL_BOT"], df["FVG_BULL_TOP"]
        brt, brb = df["FVG_BEAR_TOP"], df["FVG_BEAR_BOT"]
        long_e = (df["Low"] <= tp) & (df["Low"] >= bt) & (c > tp) & sess
        short_e = (df["High"] >= brb) & (df["High"] <= brt) & (c < brb) & sess

    elif kind == "fvg_rsi":
        bt, tp = df["FVG_BULL_BOT"], df["FVG_BULL_TOP"]
        brt, brb = df["FVG_BEAR_TOP"], df["FVG_BEAR_BOT"]
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= tp) & (df["Low"] >= bt) & (c > tp) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= brb) & (df["High"] <= brt) & (c < brb) & (rsi > p["ob"]) & sess

    elif kind == "liq_sweep":
        # liquidity grab: pierce prior swing then close back inside (ICT sweep)
        sw = p["swing"]
        hi, lo = df[f"SW_HI_{sw}"], df[f"SW_LO_{sw}"]
        long_e = (df["Low"] < lo) & (c > lo) & (df["lower_wick"] > p.get("wick", 0.45)) & sess
        short_e = (df["High"] > hi) & (c < hi) & (df["upper_wick"] > p.get("wick", 0.45)) & sess

    elif kind == "liq_sweep_fvg":
        sw = p["swing"]
        hi, lo = df[f"SW_HI_{sw}"], df[f"SW_LO_{sw}"]
        bt, tp = df["FVG_BULL_BOT"], df["FVG_BULL_TOP"]
        brt, brb = df["FVG_BEAR_TOP"], df["FVG_BEAR_BOT"]
        long_e = (df["Low"] < lo) & (c > lo) & (df["Low"] <= tp) & (c > bt) & sess
        short_e = (df["High"] > hi) & (c < hi) & (df["High"] >= brb) & (c < brt) & sess

    elif kind == "ob_retest":
        # retest of prior order block with rejection close
        long_e = (
            (df["Low"] <= df["OB_BULL_HI"])
            & (df["Low"] >= df["OB_BULL_LO"])
            & (c > df["OB_BULL_HI"])
            & sess
        )
        short_e = (
            (df["High"] >= df["OB_BEAR_LO"])
            & (df["High"] <= df["OB_BEAR_HI"])
            & (c < df["OB_BEAR_LO"])
            & sess
        )

    elif kind == "bos_pullback":
        # after BOS, pullback to EMA then continue
        sw = p.get("swing", 20)
        ema = df[f"EMA_{p['ema']}"].shift(1)
        bos_up = df[f"BOS_UP_{sw}"].shift(1).fillna(False)
        bos_dn = df[f"BOS_DN_{sw}"].shift(1).fillna(False)
        # recent BOS in last N bars
        n = p.get("bos_look", 12)
        recent_up = bos_up.rolling(n).max().fillna(0).astype(bool)
        recent_dn = bos_dn.rolling(n).max().fillna(0).astype(bool)
        long_e = recent_up & (df["Low"] <= ema) & (c > ema) & sess
        short_e = recent_dn & (df["High"] >= ema) & (c < ema) & sess

    elif kind == "fib_bounce":
        # touch fib of prior swing + wick rejection (longs from swing-low fibs, shorts from swing-high fibs)
        sw = p["swing"]
        fib = str(p["fib"])  # '382'|'500'|'618'|'786'
        long_lvl = df[f"FIB_UP{fib}_{sw}"]
        short_lvl = df[f"FIB{fib}_{sw}"]
        long_e = (df["Low"] <= long_lvl) & (c > long_lvl) & (df["lower_wick"] > p.get("wick", 0.4)) & sess
        short_e = (df["High"] >= short_lvl) & (c < short_lvl) & (df["upper_wick"] > p.get("wick", 0.4)) & sess

    elif kind == "fib_discount":
        # only long in discount (<EQ), short in premium (>EQ) at fib
        sw = p["swing"]
        fib = str(p["fib"])
        long_lvl = df[f"FIB_UP{fib}_{sw}"]
        short_lvl = df[f"FIB{fib}_{sw}"]
        long_e = (c < df[f"EQ_{sw}"]) & (df["Low"] <= long_lvl) & (c > long_lvl) & sess
        short_e = (c > df[f"EQ_{sw}"]) & (df["High"] >= short_lvl) & (c < short_lvl) & sess

    elif kind == "smc_confluence":
        # sweep + discount/premium + FVG touch
        sw = p["swing"]
        hi, lo = df[f"SW_HI_{sw}"], df[f"SW_LO_{sw}"]
        bt, tp = df["FVG_BULL_BOT"], df["FVG_BULL_TOP"]
        brt, brb = df["FVG_BEAR_TOP"], df["FVG_BEAR_BOT"]
        long_e = (
            (df["Low"] < lo)
            & (c > lo)
            & (c < df[f"EQ_{sw}"])
            & (df["Low"] <= tp)
            & (c >= bt)
            & sess
        )
        short_e = (
            (df["High"] > hi)
            & (c < hi)
            & (c > df[f"EQ_{sw}"])
            & (df["High"] >= brb)
            & (c <= brt)
            & sess
        )

    # ----- Public free-playbook strategies -----
    elif kind == "ichimoku_tk":
        # TK cross with price vs cloud filter (lag cloud one bar for wick safety)
        ten, kij = df["ICH_TENKAN"], df["ICH_KIJUN"]
        ctop, cbot = df["ICH_CLOUD_TOP"].shift(1), df["ICH_CLOUD_BOT"].shift(1)
        long_e = (ten > kij) & (ten.shift(1) <= kij.shift(1)) & (c > ctop) & sess
        short_e = (ten < kij) & (ten.shift(1) >= kij.shift(1)) & (c < cbot) & sess

    elif kind == "ichimoku_kumo":
        # Kumo breakout: close breaks prior cloud extreme
        ctop, cbot = df["ICH_CLOUD_TOP"].shift(1), df["ICH_CLOUD_BOT"].shift(1)
        long_e = (c > ctop) & (c.shift(1) <= ctop) & (df["ICH_TENKAN"] > df["ICH_KIJUN"]) & sess
        short_e = (c < cbot) & (c.shift(1) >= cbot) & (df["ICH_TENKAN"] < df["ICH_KIJUN"]) & sess

    elif kind == "rsi_macd":
        # Public confluence: RSI cross + MACD hist flip
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (
            (rsi > p["os"])
            & (rsi.shift(1) <= p["os"])
            & (df["MACD_HIST"] > 0)
            & (df["MACD_HIST"].shift(1) <= 0)
            & sess
        )
        short_e = (
            (rsi < p["ob"])
            & (rsi.shift(1) >= p["ob"])
            & (df["MACD_HIST"] < 0)
            & (df["MACD_HIST"].shift(1) >= 0)
            & sess
        )

    elif kind == "ema_cross":
        fast = df[f"EMA_{p['fast']}"]
        slow = df[f"EMA_{p['slow']}"]
        long_e = (fast > slow) & (fast.shift(1) <= slow.shift(1)) & sess
        short_e = (fast < slow) & (fast.shift(1) >= slow.shift(1)) & sess

    elif kind == "ema_pull_rsi":
        # EMA pullback in trend + RSI filter (common gold scalp variant)
        fast = df[f"EMA_{p['fast']}"].shift(1)
        slow = df[f"EMA_{p['slow']}"].shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (fast > slow) & (df["Low"] <= fast) & (c > fast) & (rsi < p["os"]) & sess
        short_e = (fast < slow) & (df["High"] >= fast) & (c < fast) & (rsi > p["ob"]) & sess

    elif kind == "heikin_flip":
        long_e = df["HA_BULL"] & (~df["HA_BULL"].shift(1).fillna(False)) & sess
        short_e = df["HA_BEAR"] & (~df["HA_BEAR"].shift(1).fillna(False)) & sess

    elif kind == "heikin_ema":
        ema = df[f"EMA_{p['ema']}"].shift(1)
        long_e = df["HA_BULL"] & (~df["HA_BULL"].shift(1).fillna(False)) & (c > ema) & sess
        short_e = df["HA_BEAR"] & (~df["HA_BEAR"].shift(1).fillna(False)) & (c < ema) & sess

    elif kind == "keltner_fade":
        up = df[f"KEL_UP_{p['period']}_{p['mult']}"].shift(1)
        lo = df[f"KEL_LO_{p['period']}_{p['mult']}"].shift(1)
        long_e = (df["Low"] <= lo) & (c > lo) & sess
        short_e = (df["High"] >= up) & (c < up) & sess

    elif kind == "zscore_fade":
        z = df[f"Z_{p['period']}"]
        long_e = (z < -p["th"]) & (z.shift(1) >= -p["th"]) & sess
        short_e = (z > p["th"]) & (z.shift(1) <= p["th"]) & sess

    elif kind == "cci_fade":
        cci = df[f"CCI_{p['period']}"]
        long_e = (cci < -p["th"]) & (cci.shift(1) >= -p["th"]) & sess
        short_e = (cci > p["th"]) & (cci.shift(1) <= p["th"]) & sess

    elif kind == "adx_di_cross":
        long_e = (
            (df["PLUS_DI"] > df["MINUS_DI"])
            & (df["PLUS_DI"].shift(1) <= df["MINUS_DI"].shift(1))
            & (df["ADX"] > p["adx"])
            & sess
        )
        short_e = (
            (df["MINUS_DI"] > df["PLUS_DI"])
            & (df["MINUS_DI"].shift(1) <= df["PLUS_DI"].shift(1))
            & (df["ADX"] > p["adx"])
            & sess
        )

    elif kind == "supertrend_flip":
        up, lo = df[f"ST_UP_{p['mult']}"], df[f"ST_LO_{p['mult']}"]
        long_e = (c > up) & (c.shift(1) <= up) & sess
        short_e = (c < lo) & (c.shift(1) >= lo) & sess

    elif kind == "turtle_break":
        # classic Donchian breakout (trend follow)
        hi = df[f"DON_HI_{p['period']}"]
        lo = df[f"DON_LO_{p['period']}"]
        long_e = (c > hi) & (c.shift(1) <= hi) & sess
        short_e = (c < lo) & (c.shift(1) >= lo) & sess

    elif kind == "orb_break":
        # opening-range breakout of Asian/London/NY first hour
        name = p["orb"]
        hi, lo = df[f"OR_HI_{name}"], df[f"OR_LO_{name}"]
        long_e = (c > hi) & (c.shift(1) <= hi) & sess
        short_e = (c < lo) & (c.shift(1) >= lo) & sess

    elif kind == "orb_fade":
        name = p["orb"]
        hi, lo = df[f"OR_HI_{name}"], df[f"OR_LO_{name}"]
        long_e = (df["Low"] <= lo) & (c > lo) & sess
        short_e = (df["High"] >= hi) & (c < hi) & sess

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


def year_stable(df: pd.DataFrame, long_e, short_e, exit_p: dict, min_trades: int = 40) -> bool:
    """Reject one-year wonders (e.g. only works in 2026 melt-up)."""
    years = sorted(set(df.index.year))
    checked = 0
    for y in years:
        idxs = np.where(df.index.year == y)[0]
        if len(idxs) < 800:
            continue
        m = simulate(
            df,
            long_e=long_e,
            short_e=short_e,
            rr_atr=exit_p["rr"],
            max_hold=exit_p["max_hold"],
            start_i=int(max(idxs[0], 250)),
            end_i=int(idxs[-1]),
            atr_col=exit_p.get("atr", "ATR_14"),
            allow_long=exit_p.get("allow_long", True),
            allow_short=exit_p.get("allow_short", True),
        )
        if m["trades"] < min_trades:
            continue
        checked += 1
        if m["winrate"] < 0.52 or m["net_pnl"] <= 0 or m["profit_factor"] < 1.0:
            return False
    return checked >= 2


def score_of(is_m, oos_m) -> float:
    if is_m["trades"] < 20:
        return -1e9
    # Must be near target or get crushed
    if is_m["winrate"] < 0.70 or is_m["trades_per_day"] < 0.8 or is_m["net_pnl"] <= 0:
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
    warm = 500 if n > 50000 else 250
    common = dict(
        long_e=long_e,
        short_e=short_e,
        rr_atr=exit_p["rr"],  # SL=TP=rr*ATR
        max_hold=exit_p["max_hold"],
        atr_col=exit_p.get("atr", "ATR_14"),
        allow_long=exit_p.get("allow_long", True),
        allow_short=exit_p.get("allow_short", True),
    )
    is_m = simulate(df, start_i=warm, end_i=split, **common)
    oos_m = simulate(df, start_i=split, end_i=n - 2, **common)
    if is_m["trades"] < 20:
        return None
    # Penalize / reject short-sample Yahoo windows when long local data exists
    span_days = max(1, (df.index[-1] - df.index[0]).days)
    sc = score_of(is_m, oos_m)
    if span_days < 200:
        sc -= 5000  # don't let 60d Yahoo beat 2y local
    if qualifies(is_m, oos_m) and span_days >= 200 and not year_stable(df, long_e, short_e, exit_p):
        sc -= 8000  # fail year gate → not a real qualifier
        # mark by crushing score so save_champion won't prefer it as QUAL
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
        score=sc,
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

    # ICT / SMC / FVG / Fib families
    for sessions in sessions_opts[:8]:
        space.append(("fvg_reclaim", {"sessions": sessions}))
        space.append(("ob_retest", {"sessions": sessions}))
        for rsi, os_, ob in itertools.product([3, 7, 14], [20, 25, 30, 35], [65, 70, 75, 80]):
            space.append(("fvg_rsi", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))
    for sw, wick, sessions in itertools.product([5, 10, 20, 40], [0.4, 0.5, 0.6], sessions_opts[:7]):
        space.append(("liq_sweep", {"swing": sw, "wick": wick, "sessions": sessions}))
        space.append(("liq_sweep_fvg", {"swing": sw, "sessions": sessions}))
        space.append(("smc_confluence", {"swing": sw, "sessions": sessions}))
    for sw, fib, wick, sessions in itertools.product(
        [10, 20, 40], ["382", "500", "618", "786"], [0.35, 0.45, 0.55], sessions_opts[:6]
    ):
        space.append(("fib_bounce", {"swing": sw, "fib": fib, "wick": wick, "sessions": sessions}))
        space.append(("fib_discount", {"swing": sw, "fib": fib, "sessions": sessions}))
    for sw, ema, bos_look, sessions in itertools.product(
        [20, 40], [8, 12, 20, 50], [6, 12, 20], sessions_opts[:5]
    ):
        space.append(("bos_pullback", {"swing": sw, "ema": ema, "bos_look": bos_look, "sessions": sessions}))

    # Public free strategies (Ichimoku, HA, Turtle, ORB, Keltner, RSI+MACD, etc.)
    for sessions in sessions_opts[:8]:
        space.append(("ichimoku_tk", {"sessions": sessions}))
        space.append(("ichimoku_kumo", {"sessions": sessions}))
        space.append(("heikin_flip", {"sessions": sessions}))
        for ema in (20, 50, 200):
            space.append(("heikin_ema", {"ema": ema, "sessions": sessions}))
        for rsi, os_, ob in itertools.product([7, 14], [35, 40, 45], [55, 60, 65]):
            space.append(("rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))
        for fast, slow in ((5, 20), (9, 21), (12, 26), (21, 50)):
            space.append(("ema_cross", {"fast": fast, "slow": slow, "sessions": sessions}))
            for rsi, os_, ob in ((2, 30, 70), (7, 35, 65), (14, 40, 60)):
                space.append(
                    ("ema_pull_rsi", {"fast": fast, "slow": slow, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions})
                )
        for period, mult in ((20, 1.5), (20, 2.0), (10, 1.5)):
            space.append(("keltner_fade", {"period": period, "mult": mult, "sessions": sessions}))
        for period, th in ((20, 1.5), (20, 2.0), (40, 2.0), (40, 2.5)):
            space.append(("zscore_fade", {"period": period, "th": th, "sessions": sessions}))
        for period, th in ((14, 100), (14, 150), (20, 100), (20, 200)):
            space.append(("cci_fade", {"period": period, "th": th, "sessions": sessions}))
        for adx in (15, 20, 25, 30):
            space.append(("adx_di_cross", {"adx": adx, "sessions": sessions}))
        for mult in (2.0, 3.0):
            space.append(("supertrend_flip", {"mult": mult, "sessions": sessions}))
        for period in (10, 20, 40):
            space.append(("turtle_break", {"period": period, "sessions": sessions}))
        for orb in ("ASIAN", "LONDON", "NY"):
            space.append(("orb_break", {"orb": orb, "sessions": sessions}))
            space.append(("orb_fade", {"orb": orb, "sessions": sessions}))

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
    # STRICT 1:1 — single rr distance for both SL and TP.
    # Ban sub-1.0×ATR stops: they recreate fake 90%+ WR on flat Twelve Data bars.
    rrs = [1.0, 1.2, 1.5, 2.0]
    holds = [4, 6, 8, 12, 18, 24, 36]
    atrs = ["ATR_14", "ATR_7"]
    if phase == 0:
        rrs, holds, atrs = [1.0, 1.2], [6, 8, 12, 24], ["ATR_14"]
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


def _champ_key(winrate: float, oos_winrate: float, tpd: float, oos_tpd: float, score: float = 0.0):
    """Lexicographic frontier: min WR first, then min tpd, then score."""
    return (min(winrate, oos_winrate), min(tpd, oos_tpd), score)


def save_champion(r: Result, force: bool = False):
    """Persist champion only if better (or force). Always append to candidates.

    Reject tiny-sample WR spikes (e.g. 23 IS / 9 OOS '100% OOS') unless force
    and the result already passed qualifies() sample gates.
    """
    with open(CANDIDATES, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(r)) + "\n")
    # Block tiny-sample WR spikes (e.g. 9 OOS trades @ 100%) unless force/qualified
    if not force and (r.trades < 30 or r.oos_trades < 15):
        return False
    if CHAMPION.exists() and not force:
        try:
            d = json.loads(CHAMPION.read_text())
            old = _champ_key(
                d["winrate"], d["oos_winrate"],
                d.get("trades_per_day", 0.0), d.get("oos_trades_per_day", 0.0),
                d.get("score", 0.0),
            )
            new = _champ_key(r.winrate, r.oos_winrate, r.trades_per_day, r.oos_trades_per_day, r.score)
            old_oos = int(d.get("oos_trades", 0))
            # Never replace a decently sampled champ with a thinner OOS miracle
            if r.oos_trades < 15:
                return False
            if old_oos >= 15 and r.oos_trades < old_oos and new[0] <= old[0] + 0.03:
                return False
            if new <= old:
                return False
        except Exception:
            pass
    CHAMPION.write_text(json.dumps(asdict(r), indent=2))
    return True


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
PARAMS = {repr(params)}
EXIT = {repr(exit_p)}

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
    raw_frames = load_frames()
    frames = {k: add_indicators(v) for k, v in raw_frames.items()}
    # Prefer long local 5m when available (Twelve Data 2y+)
    long_5m = "5m" in frames and (frames["5m"].index[-1] - frames["5m"].index[0]).days >= 200
    if long_5m:
        tf_order = ["5m"]
        log("Long local 5m detected — searching ONLY on 5m (skip short Yahoo TFs)")
    else:
        tf_order = [t for t in ("5m", "15m", "1h") if t in frames]

    champion: Result | None = None
    if CHAMPION.exists():
        try:
            d = json.loads(CHAMPION.read_text())
            # One-time archive of pre-2y short-window champs only
            short_window = (
                long_5m
                and d.get("timeframe") in ("15m", "1h")
                and (d.get("trades", 0) + d.get("oos_trades", 0)) < 80
            )
            if short_window:
                arch = OUT / "CHAMPION_pre_2y_search.json"
                if not arch.exists():
                    arch.write_text(CHAMPION.read_text())
                log(f"Ignoring short-window champion (kept file; archived→{arch.name})")
            else:
                champion = Result(**{k: d[k] for k in Result.__dataclass_fields__ if k in d})
                log(
                    f"Loaded prior champion WR={champion.winrate:.1%}/{champion.oos_winrate:.1%} "
                    f"tpd={champion.trades_per_day:.2f}/{champion.oos_trades_per_day:.2f} score={champion.score:.0f}"
                )
        except Exception as e:
            log(f"Could not load prior champion: {e}")
    tested = 0
    phase = 0
    qualified_count = 0

    while True:
        log(f"----- PHASE {phase} -----")
        space = param_space(phase)
        exits = exit_grid(phase)
        max_exits = 12 + phase * 4
        log(f"space={len(space)} exits={len(exits)} tfs={tf_order}")

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
                    if q and (df.index[-1] - df.index[0]).days >= 200:
                        q = year_stable(df, long_e, short_e, exit_p)
                    if champion is None or res.score > champion.score:
                        # Track near-misses from 60% up so 2y progress is visible
                        if res.winrate >= 0.60 and res.trades_per_day >= 0.7 and res.net_pnl > 0:
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
