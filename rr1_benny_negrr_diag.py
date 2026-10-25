#!/usr/bin/env python3
"""One-shot diagnostic: Benny reel DdCYACaP-ef claims ~80% WR with *negative* RR.

Our live goal remains RR1 (SL=TP). This script only measures whether asymmetric
exits (TP < SL, i.e. negative RR) can approach the claimed WR on XAU under the
same causal ORB entries + walk-forward stability checks.

Does NOT write CHAMPION / live_rr1_bot — diagnostic only.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from rr1_orb_walkforward import walk_forward_score
from rr1_search import (
    OUT,
    add_indicators,
    evaluate,
    gen_signals,
    load_frames,
    log,
    simulate,
    year_stable,
)

SUMMARY = OUT / "BENNY_NEGRR_DIAG.json"
LEADER = OUT / "BENNY_NEGRR_LEADERBOARD.jsonl"


def neg_exits():
    # Negative RR: TP distance < SL distance via rr_atr on both sides is impossible
    # in simulate() (SL=TP=rr*ATR). Approximate Benny "negative RR" by short max_hold
    # + smaller rr on winners isn't separable — instead use rr in (0.5, 0.75) which
    # shrinks both SL and TP equally (still 1:1 geometrically) PLUS a parallel path
    # that uses asymmetric targets via custom simulate loop below.
    return [
        {"rr": rr, "max_hold": mh, "atr": atr, **side, "tp_mult": tp, "sl_mult": sl}
        for rr in (0.5, 0.75, 1.0)
        for mh in (2, 4, 6, 8, 12)
        for atr in ("ATR_14", "ATR_7")
        for tp, sl in ((0.5, 1.0), (0.6, 1.0), (0.75, 1.25), (0.5, 1.5), (1.0, 1.0))
        for side in (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        )
    ]


def simulate_asym(df, long_e, short_e, tp_mult, sl_mult, max_hold, start_i, end_i, atr_col, allow_long, allow_short):
    """Asymmetric TP/SL in ATR units (Benny-style negative RR when tp < sl)."""
    from rr1_search import MIN_ATR, trading_days

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    cl = df["Close"].to_numpy(float)
    atr = df[atr_col].to_numpy(float)
    L = np.asarray(long_e, dtype=bool)
    S = np.asarray(short_e, dtype=bool)
    wins = losses = 0
    pnls = []
    i = max(int(start_i), 1)
    end = int(end_i)
    while i < end - 1:
        sig_i = i - 1
        go_long = bool(allow_long and L[sig_i])
        go_short = bool(allow_short and S[sig_i])
        if not go_long and not go_short:
            i += 1
            continue
        a = atr[sig_i]
        entry = o[i]
        if not np.isfinite(a) or a < MIN_ATR or not np.isfinite(entry):
            i += 1
            continue
        tp_d = a * tp_mult
        sl_d = a * sl_mult
        if tp_d < MIN_ATR * 0.25 or sl_d < MIN_ATR:
            i += 1
            continue
        if go_long and not go_short:
            side = 1
        elif go_short and not go_long:
            side = -1
        else:
            side = 1  # prefer long on dual signal
        tp = entry + side * tp_d
        sl = entry - side * sl_d
        hit = None
        exit_i = min(i + max_hold, end) - 1
        for j in range(i, min(i + max_hold, end)):
            if side == 1:
                if l[j] <= sl:
                    hit = -sl_d
                    exit_i = j
                    break
                if h[j] >= tp:
                    hit = tp_d
                    exit_i = j
                    break
            else:
                if h[j] >= sl:
                    hit = -sl_d
                    exit_i = j
                    break
                if l[j] <= tp:
                    hit = tp_d
                    exit_i = j
                    break
        if hit is None:
            hit = side * (cl[exit_i] - entry)
        pnls.append(float(hit))
        if hit > 0:
            wins += 1
        else:
            losses += 1
        i = exit_i + 1

    n = wins + losses
    if n == 0:
        return {"trades": 0, "winrate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0, "trades_per_day": 0.0, "max_dd": 0.0}
    arr = np.array(pnls)
    gp = float(arr[arr > 0].sum())
    gl = float(-arr[arr < 0].sum())
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / np.maximum(np.abs(peak), 1e-9)).max()) if len(eq) else 0.0
    days = trading_days(df.index[start_i:end_i])
    return {
        "trades": n,
        "winrate": wins / n,
        "profit_factor": (gp / gl) if gl > 0 else (9.9 if gp > 0 else 0.0),
        "net_pnl": float(arr.sum()),
        "trades_per_day": n / days,
        "max_dd": dd,
    }


def specs():
    out = []
    for orb in ("ASIAN", "LONDON", "NY"):
        for win in (15, 30, 45, 60, 90):
            out.append(("orb_break_win", {"orb": orb, "win": win, "first_only": True, "sessions": None}))
            out.append(("orb_break_win", {"orb": orb, "win": win, "first_only": False, "sessions": None}))
            out.append(("orb_retest_win", {"orb": orb, "win": win, "sessions": None}))
            out.append(("orb_of_break", {"orb": orb, "win": win, "n": 2, "sessions": None}))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    if LEADER.exists():
        LEADER.unlink()
    log("=== BENNY NEG-RR DIAGNOSTIC (reel DdCYACaP-ef) ===")
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("5m", "15m", "30m"):
        if tf in raw:
            frames[tf] = add_indicators(raw[tf])

    best = None
    near80 = []
    tested = 0
    exs = neg_exits()
    # subsample exits for runtime
    exs = [e for e in exs if e["tp_mult"] <= e["sl_mult"]]  # true neg or flat RR
    log(f"specs={len(specs())} exits={len(exs)} tfs={list(frames)}")

    for kind, base in specs():
        for tf, df in frames.items():
            try:
                L, S = gen_signals(df, kind, base)
            except Exception:
                continue
            if int(L.sum() + S.sum()) < 40:
                continue
            n = len(df)
            split = int(n * 0.7)
            for ex in exs:
                tested += 1
                is_m = simulate_asym(
                    df, L, S, ex["tp_mult"], ex["sl_mult"], ex["max_hold"],
                    250, split, ex["atr"], ex["allow_long"], ex["allow_short"],
                )
                oos_m = simulate_asym(
                    df, L, S, ex["tp_mult"], ex["sl_mult"], ex["max_hold"],
                    split, n - 2, ex["atr"], ex["allow_long"], ex["allow_short"],
                )
                if is_m["trades"] < 25 or oos_m["trades"] < 12:
                    continue
                min_wr = min(is_m["winrate"], oos_m["winrate"])
                # RR geometric = tp/sl
                geo_rr = ex["tp_mult"] / ex["sl_mult"]
                row = {
                    "name": f"benny|{kind}|{tf}",
                    "min_wr": min_wr,
                    "wr": is_m["winrate"],
                    "oos_wr": oos_m["winrate"],
                    "tpd": is_m["trades_per_day"],
                    "oos_tpd": oos_m["trades_per_day"],
                    "pf": is_m["profit_factor"],
                    "oos_pf": oos_m["profit_factor"],
                    "n": is_m["trades"],
                    "oos_n": oos_m["trades"],
                    "geo_rr": geo_rr,
                    "params": {"kind": kind, **base, "exit": ex},
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(LEADER, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                if best is None or min_wr > best["min_wr"]:
                    best = row
                    log(
                        f"BEST negRR {min_wr:.1%} IS/OOS={is_m['winrate']:.1%}/{oos_m['winrate']:.1%} "
                        f"geoRR={geo_rr:.2f} tpd={is_m['trades_per_day']:.2f} | {row['name']}"
                    )
                if min_wr >= 0.70:
                    near80.append(row)
                if tested % 200 == 0:
                    bw = f"{best['min_wr']:.1%}" if best else "n/a"
                    log(f"benny-diag tested={tested} best={bw} near70={len(near80)}")

    near80.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
    SUMMARY.write_text(
        json.dumps(
            {
                "source": "Instagram reel DdCYACaP-ef @bennnytrades",
                "claim": "~80% WR ORB negative RR 5y",
                "tested": tested,
                "best": best,
                "near_70pct": near80[:20],
                "reproduced_80pct": bool(near80 and near80[0]["min_wr"] >= 0.80),
                "note": "Asymmetric TP/SL diagnostic only — does not qualify for RR1 live bot",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )
    log(
        f"DONE tested={tested} best_minWR={best['min_wr'] if best else None} "
        f"near70={len(near80)} hit80={bool(near80 and near80[0]['min_wr']>=0.80)}"
    )


if __name__ == "__main__":
    main()
