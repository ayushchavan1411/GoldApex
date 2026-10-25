#!/usr/bin/env python3
"""
Stoch cross 5× strategy (exact backtest logic)

Setup that printed ~$500 → $3,588 (7.18×) after charges on ~2y XAU:
  - TF: 30m (UTC)
  - Long-only Stochastic(14) bullish cross while %K still < 20
  - Sessions: London/NY overlap + NY only
  - Exit: 1.2% SL = 1.2% TP, max hold 96 bars
  - Size: 0.02 lot (2 oz), spread $0.30, commission $7/lot RT

Usage:
  python3 stoch_cross_5x_strategy.py
  # or pass a CSV with Datetime,Open,High,Low,Close
  python3 stoch_cross_5x_strategy.py path/to/xau_30m.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- fixed strategy params (do not change if you want the same backtest) ---
STOCH_PERIOD = 14
STOCH_D_SMOOTH = 3
OS = 20  # %K must still be below this on the cross bar
SESSIONS = ("overlap", "ny")  # UTC hour buckets below
SL_PCT = 0.012
TP_PCT = 0.012
MAX_HOLD = 96
LOT = 0.02
CONTRACT_SIZE = 100.0  # oz per 1.00 lot → 0.02 lot = 2 oz
SPREAD = 0.30  # $ per oz; half applied adverse on entry + exit
COMMISSION_PER_LOT_RT = 7.0
START_CAP = 500.0


def load_ohlc(path: str | Path | None = None) -> pd.DataFrame:
    """Load OHLC. Prefers local XAU CSV; else resamples repo 5m → 30m."""
    if path:
        df = pd.read_csv(path)
        for col in ("datetime", "Datetime", "date", "Date", "time", "Time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        rename = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close")}
        df = df.rename(columns=rename)
        out = df[["Open", "High", "Low", "Close"]].dropna().sort_index()
    else:
        data = Path("data")
        files = sorted(data.glob("XAUUSD_5min_*.csv")) if data.exists() else []
        if not files:
            raise FileNotFoundError("No CSV given and no data/XAUUSD_5min_*.csv found")
        raw = pd.read_csv(files[-1])
        for col in ("datetime", "Datetime", "date", "Date", "time", "Time"):
            if col in raw.columns:
                raw[col] = pd.to_datetime(raw[col])
                raw = raw.set_index(col)
                break
        if not isinstance(raw.index, pd.DatetimeIndex):
            raw.index = pd.to_datetime(raw.index)
        rename = {c: c.capitalize() for c in raw.columns if c.lower() in ("open", "high", "low", "close")}
        raw = raw.rename(columns=rename)[["Open", "High", "Low", "Close"]].dropna().sort_index()
        out = (
            raw.resample("30min", label="right", closed="right")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


def add_stoch_and_session(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    ll = l.rolling(STOCH_PERIOD).min()
    hh = h.rolling(STOCH_PERIOD).max()
    d["STOCH_K"] = 100.0 * (c - ll) / (hh - ll).replace(0, np.nan)
    d["STOCH_D"] = d["STOCH_K"].rolling(STOCH_D_SMOOTH).mean()

    hour = d.index.hour
    # Same buckets as rr1_search.py (UTC)
    d["session"] = np.select(
        [
            (hour >= 0) & (hour < 8),
            (hour >= 8) & (hour < 13),
            (hour >= 13) & (hour < 17),
            (hour >= 17) & (hour < 22),
        ],
        ["asian", "london", "overlap", "ny"],
        default="late",
    )
    return d


def long_signals(df: pd.DataFrame) -> np.ndarray:
    """
    Long signal on bar i close when ALL true:
      1) session in {overlap, ny}
      2) STOCH_K < 20
      3) STOCH_K > STOCH_D          (K above D now)
      4) STOCH_K[i-1] <= STOCH_D[i-1] (bullish cross this bar)
    Fill is Open[i+1] (see simulate).
    """
    k = df["STOCH_K"]
    d = df["STOCH_D"]
    sess = df["session"].isin(SESSIONS)
    long_e = (k < OS) & (k > d) & (k.shift(1) <= d.shift(1)) & sess
    return long_e.fillna(False).to_numpy(dtype=bool)


def simulate_long(
    df: pd.DataFrame,
    long_e: np.ndarray,
    start_i: int = 250,
    end_i: int | None = None,
) -> dict:
    """Causal next-open long sim with % SL/TP + spread + commission."""
    n = len(df)
    if end_i is None:
        end_i = n - 2
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    times = df.index

    oz = LOT * CONTRACT_SIZE
    spr = SPREAD
    comm = COMMISSION_PER_LOT_RT * LOT
    capital = START_CAP
    peak = START_CAP
    max_dd = 0.0
    pnls: list[float] = []

    i = start_i
    while i < end_i and i + 1 < n:
        if not bool(long_e[i]):
            i += 1
            continue

        raw_entry = opens[i + 1]
        if not np.isfinite(raw_entry):
            i += 1
            continue

        # long fill: pay half-spread
        entry = raw_entry + spr / 2.0
        sl = entry * (1.0 - SL_PCT)
        tp = entry * (1.0 + TP_PCT)

        j = i + 1
        bars = 0
        exit_px = entry
        hit = False
        while j < n and bars < MAX_HOLD:
            hi, lo = highs[j], lows[j]
            bars += 1
            # conservative: if both SL and TP touched same bar → SL first
            if lo <= sl:
                exit_px, hit = sl, True
                break
            if hi >= tp:
                exit_px, hit = tp, True
                break
            j += 1
        if not hit:
            exit_px = closes[min(j, n - 1)]

        # exit: give half-spread
        exit_px = exit_px - spr / 2.0
        gross = (exit_px - entry) * oz
        pnl = gross - comm
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0.0)
        pnls.append(pnl)
        i = j + 1 if j > i else i + 1

    if not pnls:
        return {"trades": 0, "winrate": 0.0, "net_pnl": 0.0, "final_equity": capital, "max_dd": max_dd}

    arr = np.asarray(pnls, float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    span_days = max(1.0, ((times[end_i - 1] - times[start_i]).days + 1) * 5.5 / 7.0)
    return {
        "trades": int(len(arr)),
        "winrate": float((arr > 0).mean()),
        "profit_factor": (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0),
        "net_pnl": float(arr.sum()),
        "final_equity": float(capital),
        "multiple": float(capital / START_CAP),
        "max_dd": float(max_dd),
        "trades_per_day": float(len(arr) / span_days),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    df = add_stoch_and_session(load_ohlc(csv))
    sig = long_signals(df)
    n = len(df)
    split = int(n * 0.7)

    full = simulate_long(df, sig, start_i=250, end_i=n - 2)
    is_m = simulate_long(df, sig, start_i=250, end_i=split)
    oos = simulate_long(df, sig, start_i=split, end_i=n - 2)

    print("=== Stoch cross LONG · overlap+ny · 30m · 1.2%/1.2% · hold96 · 0.02 lot ===")
    print(f"bars={n}  {df.index[0]} → {df.index[-1]}  signals={int(sig.sum())}")
    for label, m in (("FULL", full), ("IS70%", is_m), ("OOS30%", oos)):
        print(
            f"{label}: WR={m['winrate']:.1%}  n={m['trades']}  "
            f"PF={m.get('profit_factor', 0):.2f}  net=${m['net_pnl']:.1f}  "
            f"eq=${m['final_equity']:.1f} ({m.get('multiple', m['final_equity']/START_CAP):.2f}x)  "
            f"dd={m['max_dd']:.1%}  tpd={m.get('trades_per_day', 0):.2f}"
        )


if __name__ == "__main__":
    main()
