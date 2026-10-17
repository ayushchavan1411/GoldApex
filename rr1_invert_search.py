#!/usr/bin/env python3
"""Invert / fade hunter: flip low-WR RR1 signals (buy when base sells).

At true 1:1 barriers, WR_inv ≈ 1 − WR_base. We only promote fades that clear
high WR on BOTH IS and OOS with year_stable + ATR floor (same engine gates).

Approach:
  1) Score base kinds across TFs/exits.
  2) Keep bases with IS WR ≤ LOW_WR (genuinely anti-edge candidates).
  3) Re-test with params["invert"]=True (swapped long/short).
  4) Rank / save champions that approach ≥75% WR or useful ≥65% near-hits.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np

from rr1_search import (
    OUT,
    add_indicators,
    evaluate,
    gen_signals,
    load_frames,
    log,
    qualifies,
    save_champion,
    write_bot,
    year_stable,
)

RNG = random.Random(42)
LEADER = OUT / "INVERT_LEADERBOARD.jsonl"
PROGRESS = OUT / "invert_progress.json"
RESULTS = OUT / "INVERT_RESULTS.json"

LOW_WR = 0.45          # only fade bases this weak (or worse) on IS
MIN_BASE_TRADES = 40   # avoid tiny-sample "fake" losers
NEAR_WR = 0.65
TARGET_WR = 0.75


def exits():
    return [
        {"rr": rr, "max_hold": mh, "atr": atr, **side}
        for rr in (1.0, 1.2)
        for mh in (4, 6, 8, 12, 18, 24)
        for atr in ("ATR_14", "ATR_7")
        for side in (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        )
    ]


def specs():
    """Compact but diverse base signal grid (both mean-reversion and trend)."""
    out = []
    sessions = [
        None,
        ("asian",),
        ("london",),
        ("overlap",),
        ("ny",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
        ("asian", "london", "overlap", "ny"),
    ]
    for sess in sessions:
        for rsi, os_, ob in itertools.product([3, 5, 7, 14], [20, 25, 30, 35, 40], [60, 65, 70, 75, 80]):
            if os_ >= ob:
                continue
            out.append(("rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}))
            out.append(("rsi_level", {"rsi": rsi, "os": os_, "ob": ob, "band": 5, "sessions": sess}))
            if rsi in (7, 14) and os_ in (30, 35, 40) and ob in (60, 65, 70):
                out.append(("rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}))
        for fast, slow in itertools.product([5, 8, 12, 21], [21, 34, 50, 100]):
            if fast >= slow:
                continue
            out.append(("ema_cross", {"fast": fast, "slow": slow, "sessions": sess}))
            out.append(("ema_pull_rsi", {"fast": fast, "slow": slow, "rsi": 7, "os": 35, "ob": 65, "sessions": sess}))
        for period, k in ((10, 1.5), (10, 2.0), (20, 2.0), (20, 2.5)):
            out.append(("bb_fade", {"period": period, "k": k, "sessions": sess}))
            out.append(("bb_rsi_fade", {"period": period, "k": k, "rsi": 7, "os": 30, "ob": 70, "sessions": sess}))
            out.append(("bb_mid_reject", {"period": period, "k": k, "sessions": sess}))
        for period, os_, ob in itertools.product([5, 9, 14], [20, 25, 30], [70, 75, 80]):
            out.append(("stoch_cross", {"period": period, "os": os_, "ob": ob, "sessions": sess}))
        out.append(("macd_cross", {"sessions": sess}))
        out.append(("donch_fade", {"period": 20, "sessions": sess}))
        out.append(("donch_fade", {"period": 40, "sessions": sess}))
        out.append(("supertrend_flip", {"mult": 3.0, "sessions": sess}))
        out.append(("turtle_break", {"n": 20, "sessions": sess}))
        out.append(("heikin_ema", {"ema": 50, "sessions": sess}))
        out.append(("zscore_fade", {"period": 20, "th": 2.0, "sessions": sess}))
        for swing in (10, 20, 40):
            out.append(("liq_sweep", {"swing": swing, "wick": 0.45, "sessions": sess}))
            out.append(("smc_confluence", {"swing": swing, "sessions": sess}))
            out.append(("fvg_reclaim", {"sessions": sess}))
            out.append(("fvg_rsi", {"rsi": 7, "os": 35, "ob": 65, "sessions": sess}))
            for fib in ("500", "618"):
                out.append(("fib_bounce", {"swing": swing, "fib": fib, "sessions": sess}))
    # de-dupe by (kind, frozenset params)
    seen = set()
    uniq = []
    for kind, p in out:
        key = (kind, json.dumps(p, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((kind, p))
    return uniq


def track(res, tag: str, base_wr: float | None = None):
    if not res:
        return
    row = {
        "tag": tag,
        "name": res.name,
        "wr": res.winrate,
        "oos_wr": res.oos_winrate,
        "min_wr": min(res.winrate, res.oos_winrate),
        "tpd": res.trades_per_day,
        "oos_tpd": res.oos_trades_per_day,
        "n": res.trades,
        "oos_n": res.oos_trades,
        "pf": res.profit_factor,
        "oos_pf": res.oos_profit_factor,
        "pnl": res.net_pnl,
        "oos_pnl": res.oos_net_pnl,
        "base_wr": base_wr,
        "params": res.params,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(LEADER, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def dump_progress(tested, quals, best, extra=None):
    payload = {
        "tested": tested,
        "quals": quals,
        "best": None
        if best is None
        else {
            "name": best.name,
            "wr": best.winrate,
            "oos_wr": best.oos_winrate,
            "tpd": best.trades_per_day,
            "oos_tpd": best.oos_trades_per_day,
            "n": best.trades,
            "oos_n": best.oos_trades,
            "pf": best.profit_factor,
            "oos_pf": best.oos_profit_factor,
            "params": best.params,
        },
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    PROGRESS.write_text(json.dumps(payload, indent=2, default=str))


def maybe_promote(best, res, L, S, df, ex, quals):
    """Save if full qualifier or strong near-hit."""
    is_m = {
        "trades": res.trades,
        "winrate": res.winrate,
        "profit_factor": res.profit_factor,
        "net_pnl": res.net_pnl,
        "trades_per_day": res.trades_per_day,
        "max_dd": res.max_dd,
    }
    oos_m = {
        "trades": res.oos_trades,
        "winrate": res.oos_winrate,
        "profit_factor": res.oos_profit_factor,
        "net_pnl": res.oos_net_pnl,
        "trades_per_day": res.oos_trades_per_day,
    }
    q = qualifies(is_m, oos_m)
    if q:
        q = year_stable(df, L, S, ex)
    useful = (
        res.winrate >= NEAR_WR
        and res.oos_winrate >= NEAR_WR
        and res.trades >= 30
        and res.oos_trades >= 15
        and res.profit_factor >= 1.1
        and res.oos_profit_factor >= 1.1
    )
    score = (
        min(res.winrate, res.oos_winrate),
        min(res.trades_per_day, res.oos_trades_per_day),
        res.trades + res.oos_trades,
    )
    bscore = (
        (min(best.winrate, best.oos_winrate), min(best.trades_per_day, best.oos_trades_per_day), best.trades + best.oos_trades)
        if best
        else (-1.0, -1.0, -1)
    )
    if q or useful:
        if best is None or score > bscore:
            best = res
            save_champion(res, force=bool(q))
            write_bot(res)
            log(
                f"{'QUAL ' if q else 'NEAR '}INV WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                f"n={res.trades}/{res.oos_trades} PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f} {res.name}"
            )
        if q:
            quals += 1
    elif best is None or score > bscore:
        # keep best minWR even if below near gate (for reporting)
        if res.trades >= 25 and res.oos_trades >= 10:
            best = res
    return best, quals, q


def write_results(best, top_rows, quals, tested):
    RESULTS.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals": quals,
                "champion": None
                if best is None
                else {
                    "name": best.name,
                    "wr": best.winrate,
                    "oos_wr": best.oos_winrate,
                    "tpd": best.trades_per_day,
                    "oos_tpd": best.oos_trades_per_day,
                    "n": best.trades,
                    "oos_n": best.oos_trades,
                    "pf": best.profit_factor,
                    "oos_pf": best.oos_profit_factor,
                    "pnl": best.net_pnl,
                    "oos_pnl": best.oos_net_pnl,
                    "params": best.params,
                },
                "top": top_rows[:20],
                "note": "invert=True fades base signal at RR1; WR_inv ≈ 1−WR_base when barriers dominate",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )


def run_pass(frames, sp, exs, tested, quals, best, top_rows):
    """One shuffled pass over specs × TFs."""
    RNG.shuffle(sp)
    tfs = [t for t in ("5m", "15m", "1h", "1m") if t in frames]
    for kind, base_p in sp:
        for tf in tfs:
            # skip 1m for heavy kinds to limit RAM/CPU
            if tf == "1m" and kind in ("smc_confluence", "fib_bounce", "liq_sweep", "turtle_break"):
                continue
            df = frames[tf]
            try:
                L0, S0 = gen_signals(df, kind, base_p)
            except Exception:
                continue
            n_sig = int(L0.sum() + S0.sum())
            if n_sig < 80:
                continue
            # Score a few exits on the BASE (non-inverted) to find losers
            base_best = None
            for ex in exs[::5]:
                tested += 1
                res = evaluate(f"base|{kind}|{tf}", tf, df, L0, S0, ex, {"kind": kind, **base_p})
                if not res:
                    continue
                if base_best is None or res.winrate < base_best.winrate:
                    base_best = res
                    base_best_ex = ex
            if base_best is None or base_best.trades < MIN_BASE_TRADES:
                continue
            # Need a real loser on IS; OOS also ≤50% avoids fading IS-unlucky / OOS-fine noise
            if base_best.winrate > LOW_WR or base_best.oos_winrate > 0.50:
                continue

            # Fade: invert signals + test denser exit grid
            inv_p = {**base_p, "invert": True}
            try:
                Li, Si = gen_signals(df, kind, inv_p)
            except Exception:
                continue
            # also swap allow flags relative to each exit
            for ex in exs[::2]:
                tested += 1
                ex_inv = dict(ex)
                # if base was directional, flip side preference
                if ex.get("allow_long") != ex.get("allow_short"):
                    ex_inv["allow_long"] = ex.get("allow_short", True)
                    ex_inv["allow_short"] = ex.get("allow_long", True)
                res = evaluate(
                    f"inv|{kind}|{tf}",
                    tf,
                    df,
                    Li,
                    Si,
                    ex_inv,
                    {"kind": kind, **inv_p},
                )
                if not res:
                    continue
                track(res, "invert", base_wr=base_best.winrate)
                top_rows.append(
                    {
                        "name": res.name,
                        "wr": res.winrate,
                        "oos_wr": res.oos_winrate,
                        "min_wr": min(res.winrate, res.oos_winrate),
                        "tpd": res.trades_per_day,
                        "oos_tpd": res.oos_trades_per_day,
                        "n": res.trades,
                        "oos_n": res.oos_trades,
                        "pf": res.profit_factor,
                        "oos_pf": res.oos_profit_factor,
                        "base_wr": base_best.winrate,
                        "base_name": base_best.name,
                        "params": res.params,
                    }
                )
                best, quals, _ = maybe_promote(best, res, Li, Si, df, ex_inv, quals)
            if tested % 200 == 0:
                top_rows.sort(key=lambda r: (r["min_wr"], r["n"] + r["oos_n"]), reverse=True)
                top_rows[:] = top_rows[:50]
                dump_progress(tested, quals, best, {"low_wr_bases_faded": True})
                write_results(best, top_rows, quals, tested)
                log(
                    f"invert tested={tested} quals={quals} "
                    f"best={(f'{best.winrate:.1%}/{best.oos_winrate:.1%} tpd{best.trades_per_day:.2f}' if best else 'n/a')}"
                )
    return tested, quals, best, top_rows


def main():
    OUT.mkdir(exist_ok=True)
    log("=== RR1 INVERT/FADE HUNT (fade WR≤45% bases → high WR) ===")
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf, df in raw.items():
        log(f"indicators {tf} bars={len(df)}…")
        frames[tf] = add_indicators(df)

    sp = specs()
    exs = exits()
    log(f"specs={len(sp)} exits={len(exs)} tfs={list(frames)}")

    tested = 0
    quals = 0
    best = None
    top_rows: list[dict] = []

    while True:
        tested, quals, best, top_rows = run_pass(frames, sp, exs, tested, quals, best, top_rows)
        top_rows.sort(key=lambda r: (r["min_wr"], r["n"] + r["oos_n"]), reverse=True)
        top_rows[:] = top_rows[:50]
        dump_progress(tested, quals, best)
        write_results(best, top_rows, quals, tested)
        log(f"invert pass done tested={tested} quals={quals}; reshuffle+loop")
        RNG.shuffle(sp)


if __name__ == "__main__":
    main()
