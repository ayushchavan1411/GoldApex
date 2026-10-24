#!/usr/bin/env python3
"""Backtest TradeIQ / uploaded playbooks on XAUUSD for our RR1 hunt context.

1) 4H SMA 20/200 retrace (doc includes Gold) — fixed-$ exits + RR1 ATR exits
2) First Red Day (equity parabolic short) — adapted to gold daily
3) COT / 8020 — documented as filters / NQ-only (not fully simulated here)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rr1_search import (
    OUT,
    START_CAP,
    RISK_PCT,
    MAX_RISK,
    HALT,
    MIN_ATR,
    load_frames,
    resample_ohlc,
    trading_days,
    log,
)

RESULTS = OUT / "TRADEIQ_PLAYBOOK_BACKTEST.json"


def metrics(pnls, times, start_i, end_i, capital, max_dd) -> dict:
    if not pnls:
        return {
            "trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "max_dd": max_dd,
            "trades_per_day": 0.0,
            "final_equity": capital,
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
        "max_dd": float(max_dd),
        "trades_per_day": float(len(arr) / tdays) if tdays else 0.0,
        "final_equity": float(capital),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def split_eval(df, sim_fn) -> dict:
    n = len(df)
    split = int(n * 0.7)
    is_m = sim_fn(0, split)
    oos_m = sim_fn(split, n - 2)
    return {
        "is": is_m,
        "oos": oos_m,
        "min_wr": min(is_m["winrate"], oos_m["winrate"]) if is_m["trades"] and oos_m["trades"] else 0.0,
        "n": [is_m["trades"], oos_m["trades"]],
        "tpd": [is_m["trades_per_day"], oos_m["trades_per_day"]],
    }


def sim_fixed_dollar_partial(
    df: pd.DataFrame,
    long_e: np.ndarray,
    short_e: np.ndarray,
    start_i: int,
    end_i: int,
    sl_usd: float = 50.0,
    tp1_usd: float = 40.0,
    trail_usd: float = 25.0,
    max_hold: int = 80,
) -> dict:
    """Published 4H SMA exits: SL $50, TP1 $40 close 50%, then $25 trail on remainder."""
    n = len(df)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    times = df.index
    capital = START_CAP
    peak = START_CAP
    max_dd = 0.0
    pnls: list[float] = []
    i = max(start_i, 210)
    while i < end_i and i + 1 < n:
        if capital < HALT:
            break
        gl, gs = bool(long_e[i]), bool(short_e[i])
        if gl == gs:
            i += 1
            continue
        direction = "LONG" if gl else "SHORT"
        entry = opens[i + 1]
        if not np.isfinite(entry):
            i += 1
            continue
        risk = min(MAX_RISK, capital * RISK_PCT)
        qty_full = risk / sl_usd
        qty_half = qty_full * 0.5
        sl = entry - sl_usd if direction == "LONG" else entry + sl_usd
        tp1 = entry + tp1_usd if direction == "LONG" else entry - tp1_usd

        j = i + 1
        bars = 0
        phase = 0  # 0 seeking TP1/SL, 1 trailing
        exit1 = None
        extreme = entry
        done = False
        pnl = 0.0
        while j < n and bars < max_hold:
            hi, lo = highs[j], lows[j]
            bars += 1
            if phase == 0:
                if direction == "LONG":
                    if lo <= sl:
                        pnl = (sl - entry) * qty_full
                        done = True
                        break
                    if hi >= tp1:
                        exit1 = tp1
                        extreme = hi
                        phase = 1
                else:
                    if hi >= sl:
                        pnl = (entry - sl) * qty_full
                        done = True
                        break
                    if lo <= tp1:
                        exit1 = tp1
                        extreme = lo
                        phase = 1
            else:
                if direction == "LONG":
                    extreme = max(extreme, hi)
                    trail = extreme - trail_usd
                    if lo <= trail:
                        pnl = (exit1 - entry) * qty_half + (trail - entry) * qty_half
                        done = True
                        break
                else:
                    extreme = min(extreme, lo)
                    trail = extreme + trail_usd
                    if hi >= trail:
                        pnl = (entry - exit1) * qty_half + (entry - trail) * qty_half
                        done = True
                        break
            j += 1

        if not done:
            exit_p = closes[min(j, n - 1)]
            if phase == 0:
                pnl = (exit_p - entry) * qty_full if direction == "LONG" else (entry - exit_p) * qty_full
            else:
                pnl = (
                    (exit1 - entry) * qty_half + (exit_p - entry) * qty_half
                    if direction == "LONG"
                    else (entry - exit1) * qty_half + (entry - exit_p) * qty_half
                )

        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0)
        pnls.append(pnl)
        i = j + 1 if j > i else i + 1

    return metrics(pnls, times, start_i, end_i, capital, max_dd)


def sim_rr1(
    df: pd.DataFrame,
    long_e: np.ndarray,
    short_e: np.ndarray,
    start_i: int,
    end_i: int,
    rr: float = 1.0,
    max_hold: int = 24,
    atr_col: str = "ATR_14",
) -> dict:
    """Causal next-open RR1 ATR exits (same as hunter)."""
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
    i = max(start_i, 210)
    while i < end_i and i + 1 < n:
        if capital < HALT:
            break
        gl, gs = bool(long_e[i]), bool(short_e[i])
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
        dist = atr * rr
        if dist < MIN_ATR:
            i += 1
            continue
        qty = risk / dist
        sl = entry - dist if direction == "LONG" else entry + dist
        tp = entry + dist if direction == "LONG" else entry - dist
        j = i + 1
        bars = 0
        exit_p = entry
        hit = False
        while j < n and bars < max_hold:
            hi, lo = highs[j], lows[j]
            bars += 1
            if direction == "LONG":
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
    return metrics(pnls, times, start_i, end_i, capital, max_dd)


def build_sma_retrace_signals(df4: pd.DataFrame, tol_usd: float = 5.0):
    """20/200 SMA crossover bias + retrace within tol of 20 SMA. Causal (shift)."""
    c = df4["Close"]
    sma20 = c.rolling(20).mean()
    sma200 = c.rolling(200).mean()
    # crossover on closed bar i → bias from i onward (use shift so we don't peek)
    bull = (sma20 > sma200) & (sma20.shift(1) <= sma200.shift(1))
    bear = (sma20 < sma200) & (sma20.shift(1) >= sma200.shift(1))
    bias = pd.Series(0, index=df4.index, dtype=int)
    # forward-fill bias after cross (causal: set at bar close, trade next open)
    state = 0
    for i in range(len(df4)):
        if bool(bull.iloc[i]):
            state = 1
        elif bool(bear.iloc[i]):
            state = -1
        bias.iloc[i] = state
    # retrace: low/high touches within tol of sma20, bias still active, not on cross bar
    near = (df4["Low"] <= sma20 + tol_usd) & (df4["High"] >= sma20 - tol_usd)
    # require slope of 20 SMA in trade direction
    slope = sma20 - sma20.shift(1)
    long_e = (bias == 1) & near & (slope > 0) & (~bull) & sma20.notna() & sma200.notna()
    short_e = (bias == -1) & near & (slope < 0) & (~bear) & sma20.notna() & sma200.notna()
    # one signal per retrace cluster: keep first touch after leaving zone
    def dedupe(sig: pd.Series) -> np.ndarray:
        out = np.zeros(len(sig), dtype=bool)
        armed = True
        vals = sig.fillna(False).to_numpy(bool)
        sma = sma20.to_numpy(float)
        mid = ((df4["High"] + df4["Low"]) / 2).to_numpy(float)
        for i in range(len(vals)):
            if not np.isfinite(sma[i]):
                continue
            in_zone = abs(mid[i] - sma[i]) <= tol_usd * 2
            if vals[i] and armed:
                out[i] = True
                armed = False
            if not in_zone:
                armed = True
        return out

    return dedupe(long_e), dedupe(short_e), bias


def build_first_red_day_signals(daily: pd.DataFrame, min_green: int = 3, min_ext_pct: float = 0.03):
    """Adapt First Red Day to gold: short when first red day after min_green up-days with extension."""
    c = daily["Close"]
    o = daily["Open"]
    green = c > o  # simplistic body green
    # consecutive green count ending yesterday
    greencnt = np.zeros(len(daily), dtype=int)
    run = 0
    g = green.to_numpy(bool)
    for i in range(len(daily)):
        if g[i]:
            run += 1
        else:
            run = 0
        greencnt[i] = run
    # extension: close vs close N days ago
    ext = c / c.shift(min_green) - 1.0
    # first red: today red (close < prior close) after greencnt[i-1] >= min_green
    prior_green_run = pd.Series(greencnt, index=daily.index).shift(1)
    first_red = (c < c.shift(1)) & (prior_green_run >= min_green) & (ext.shift(1) >= min_ext_pct)
    # entry on next day open after signal bar close — for daily we mark signal on the red day close
    # Intraday "attack" ≈ next bar open short (causal)
    short_e = first_red.fillna(False).to_numpy(bool)
    long_e = np.zeros(len(daily), dtype=bool)
    return long_e, short_e


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    d = df.copy()
    prev = d["Close"].shift(1)
    tr = pd.concat(
        [(d["High"] - d["Low"]), (d["High"] - prev).abs(), (d["Low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR_14"] = tr.rolling(period).mean()
    return d


def main():
    OUT.mkdir(exist_ok=True)
    log("=== TradeIQ playbook backtests on XAU ===")
    raw = load_frames(include_resampled=True)
    base = raw.get("5m")
    if base is None:
        raise SystemExit("no 5m gold")
    df4 = resample_ohlc(base, "4h")
    df4 = add_atr(df4)
    daily = resample_ohlc(base, "1D")
    daily = add_atr(daily)
    log(f"4h bars={len(df4)} daily={len(daily)} range={df4.index[0]}→{df4.index[-1]}")

    results = {
        "notes": {
            "cot": "Macro weekly filter only — not an entry. Needs CFTC non-commercial net for GC; lag Tue→Fri. Use as bias over chart entries.",
            "8020_okala": "Nasdaq NQ microstructure (prices ending 80/20). Not transferable to XAU RR1.",
            "first_red_day": "Equity hype-parabolic short (~4x/year claim). Adapted loosely to gold daily.",
            "sma_4h": "Published rules include Gold; pip≡$1 for XAU (SL$50/TP1$40/trail$25).",
        }
    }

    # --- 4H SMA ---
    L, S, bias = build_sma_retrace_signals(df4, tol_usd=5.0)
    log(f"SMA signals long={int(L.sum())} short={int(S.sum())} bias_nonzero={(bias!=0).sum()}")

    def run_fixed(a, b):
        return sim_fixed_dollar_partial(df4, L, S, a, b)

    def run_rr1(a, b, rr=1.0, mh=24):
        return sim_rr1(df4, L, S, a, b, rr=rr, max_hold=mh)

    sma_fixed = split_eval(df4, run_fixed)
    sma_rr1 = split_eval(df4, lambda a, b: run_rr1(a, b, 1.0, 24))
    sma_rr12 = split_eval(df4, lambda a, b: run_rr1(a, b, 1.2, 18))
    results["sma_4h_fixed_dollar"] = {
        **sma_fixed,
        "params": {"sma": [20, 200], "tol_usd": 5, "sl": 50, "tp1": 40, "trail": 25, "partial": 0.5},
    }
    results["sma_4h_rr1"] = {**sma_rr1, "params": {"rr": 1.0, "max_hold": 24, "atr": "ATR_14"}}
    results["sma_4h_rr1_2"] = {**sma_rr12, "params": {"rr": 1.2, "max_hold": 18, "atr": "ATR_14"}}

    # --- First Red Day adapted ---
    Ld, Sd = build_first_red_day_signals(daily, min_green=3, min_ext_pct=0.025)
    log(f"FirstRed signals={int(Sd.sum())}")

    def run_frd_rr1(a, b):
        return sim_rr1(daily, Ld, Sd, a, b, rr=1.0, max_hold=5)

    def run_frd_rr12(a, b):
        return sim_rr1(daily, Ld, Sd, a, b, rr=1.2, max_hold=8)

    # Also fixed $ stop reclaim-style: SL = prior day range / high, use 1.5*ATR as proxy
    frd1 = split_eval(daily, run_frd_rr1)
    frd2 = split_eval(daily, run_frd_rr12)
    results["first_red_day_rr1"] = {**frd1, "params": {"min_green": 3, "min_ext_pct": 0.025, "rr": 1.0, "max_hold": 5}}
    results["first_red_day_rr1_2"] = {**frd2, "params": {"min_green": 3, "min_ext_pct": 0.025, "rr": 1.2, "max_hold": 8}}

    # invert first red (fade the short = buy exhaustion) for our lowWR→invert path
    frd_inv = split_eval(daily, lambda a, b: sim_rr1(daily, Sd, Ld, a, b, rr=1.0, max_hold=5))
    results["first_red_day_INVERT_rr1"] = {
        **frd_inv,
        "params": {"note": "fade first-red short → long", "rr": 1.0, "max_hold": 5},
    }

    RESULTS.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote {RESULTS}")

    def fmt(tag, block):
        is_m, oos = block["is"], block["oos"]
        log(
            f"{tag}: WR {is_m['winrate']:.1%}/{oos['winrate']:.1%} "
            f"n={is_m['trades']}/{oos['trades']} tpd={is_m['trades_per_day']:.3f}/{oos['trades_per_day']:.3f} "
            f"PF={is_m['profit_factor']:.2f}/{oos['profit_factor']:.2f} pnl={is_m['net_pnl']:.1f}/{oos['net_pnl']:.1f}"
        )

    for k in [
        "sma_4h_fixed_dollar",
        "sma_4h_rr1",
        "sma_4h_rr1_2",
        "first_red_day_rr1",
        "first_red_day_rr1_2",
        "first_red_day_INVERT_rr1",
    ]:
        fmt(k, results[k])


if __name__ == "__main__":
    main()
