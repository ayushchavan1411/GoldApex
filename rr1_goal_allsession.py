#!/usr/bin/env python3
"""Goal hunter: WR≥75% IS+OOS, tpd≥1, year-stable — ALL sessions (no session blocks).

Combines order-flow proxies with RSI/EMA/SMC legs via AND filters, and also
inverts weak bases. sessions is always None so Asian/London/overlap/NY all trade.
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

RNG = random.Random(99)
LEADER = OUT / "GOAL_ALLSESSION_LEADERBOARD.jsonl"
PROGRESS = OUT / "goal_allsession_progress.json"
SUMMARY = OUT / "GOAL_ALLSESSION_SUMMARY.json"

TARGET = 0.75
NEAR = 0.65


def exits_hf():
    """Exits biased toward enough trades for tpd≥1."""
    return [
        {"rr": rr, "max_hold": mh, "atr": atr, **side}
        for rr in (1.0, 1.2)
        for mh in (4, 6, 8, 12, 18)
        for atr in ("ATR_14", "ATR_7")
        for side in (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        )
    ]


def base_specs():
    """High-frequency + OF legs, sessions always unrestricted."""
    s = None
    out = []
    # Order flow
    for lo, hi in ((0.2, 0.8), (0.25, 0.75), (0.3, 0.7)):
        out.append(("of_clv_fade", {"lo": lo, "hi": hi, "sessions": s}))
    for n in (2, 3, 4):
        out.append(("of_stacked_imb", {"n": n, "sessions": s}))
        out.append(("of_stacked_fade", {"n": n, "sessions": s}))
        out.append(("of_delta_flip", {"n": n, "sessions": s}))
        out.append(("of_imb_absorb", {"n": n, "sessions": s}))
    out.append(("of_absorption", {"sessions": s}))
    out.append(("of_sweep_absorb", {"sessions": s}))
    for span in (8, 20):
        out.append(("of_cvd_cross", {"span": span, "sessions": s}))
        out.append(("of_cvd_rsi", {"span": span, "rsi": 7, "os": 30, "ob": 70, "sessions": s}))
        out.append(("of_cvd_rsi", {"span": span, "rsi": 14, "os": 35, "ob": 65, "sessions": s}))
    out.append(("of_cvd_div", {"look": 20, "sessions": s}))
    # Classic HF
    for rsi, os_, ob in itertools.product([3, 5, 7, 14], [25, 30, 35, 40], [60, 65, 70, 75]):
        if os_ >= ob:
            continue
        out.append(("rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": s}))
        out.append(("rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": s}))
    for fast, slow in ((5, 21), (8, 34), (12, 50), (21, 55)):
        out.append(("ema_cross", {"fast": fast, "slow": slow, "sessions": s}))
    out.append(("macd_cross", {"sessions": s}))
    for swing in (10, 20):
        out.append(("liq_sweep", {"swing": swing, "wick": 0.45, "sessions": s}))
        out.append(("mss_entry", {"swing": swing, "sessions": s}))
        out.append(("fvg_rsi", {"rsi": 7, "os": 35, "ob": 65, "sessions": s}))
    return out


def dump(tested, quals, best, near, tpd1):
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals": quals,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best["name"] if best else None,
                "best_tpd": best.get("tpd") if best else None,
                "near": len(near),
                "tpd1_hits": len(tpd1),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals": quals,
                "best": best,
                "near_hits": near[:15],
                "best_tpd1": tpd1[:15],
                "sessions": "ALL (None — no session blocks)",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )


def main():
    OUT.mkdir(exist_ok=True)
    log("=== GOAL ALL-SESSION HUNT (no session blocks) WR≥75% tpd≥1 ===")
    raw = load_frames(include_resampled=True)
    # Prefer TFs that can hit tpd≥1. Skip 1m by default (910k bars + OF OOMs the box).
    frames = {}
    for tf in ("5m", "15m", "30m"):
        if tf not in raw:
            continue
        try:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])
        except Exception as e:
            log(f"skip {tf}: {e}")

    bases = base_specs()
    exs = exits_hf()
    log(f"bases={len(bases)} exits={len(exs)} tfs={list(frames)}")

    tested = quals = 0
    best = None
    near: list[dict] = []
    tpd1: list[dict] = []

    # Pre-build signal cache for AND combos
    cache: dict[tuple, tuple] = {}

    def get_sig(tf, kind, p, invert=False):
        key = (tf, kind, json.dumps(p, sort_keys=True, default=str), invert)
        if key not in cache:
            pp = {**p, "invert": True} if invert else dict(p)
            # force no session block
            pp["sessions"] = None
            cache[key] = gen_signals(frames[tf], kind, pp)
        return cache[key]

    while True:
        RNG.shuffle(bases)
        # Phase A: singles (base + invert), all sessions
        for kind, p in bases:
            p = {**p, "sessions": None}
            for invert in (False, True):
                for tf, df in frames.items():
                    try:
                        L, S = get_sig(tf, kind, p, invert)
                    except Exception:
                        continue
                    if int(L.sum() + S.sum()) < 120:
                        continue
                    for ex in exs:
                        tested += 1
                        tag = f"{'inv|' if invert else 'g|'}{kind}|{tf}"
                        res = evaluate(tag, tf, df, L, S, dict(ex), {"kind": kind, **p, "invert": invert})
                        if not res or res.trades < 30 or res.oos_trades < 15:
                            continue
                        min_wr = min(res.winrate, res.oos_winrate)
                        row = {
                            "name": res.name,
                            "wr": res.winrate,
                            "oos_wr": res.oos_winrate,
                            "min_wr": min_wr,
                            "tpd": res.trades_per_day,
                            "oos_tpd": res.oos_trades_per_day,
                            "n": res.trades,
                            "oos_n": res.oos_trades,
                            "pf": res.profit_factor,
                            "oos_pf": res.oos_profit_factor,
                            "params": res.params,
                            "year_ok": year_stable(df, L, S, dict(ex)),
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        if best is None or (min_wr, min(res.trades_per_day, res.oos_trades_per_day)) > (
                            best["min_wr"],
                            min(best["tpd"], best.get("oos_tpd", 0)),
                        ):
                            best = row
                        if res.trades_per_day >= 1.0 and min_wr >= 0.55:
                            tpd1.append(row)
                            tpd1.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                            tpd1[:] = tpd1[:40]
                        if min_wr >= NEAR and res.trades >= 40:
                            near.append(row)
                            near.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                            near[:] = near[:40]
                            log(
                                f"NEAR {min_wr:.1%} {res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f} n={res.trades}/{res.oos_trades} "
                                f"year={row['year_ok']} | {res.name}"
                            )
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
                        if qualifies(is_m, oos_m) and year_stable(df, L, S, dict(ex)):
                            quals += 1
                            save_champion(res, force=True)
                            write_bot(res)
                            log(f"QUAL ALLSESS {res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f} | {res.name}")
                        if tested % 250 == 0:
                            dump(tested, quals, best, near, tpd1)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            log(f"goal tested={tested} quals={quals} best={bw} tpd1={len(tpd1)} near={len(near)}")

        # Phase B: AND of OF leg with RSI/EMA (still all sessions) — raise WR, keep frequency
        of_legs = [b for b in bases if b[0].startswith("of_")]
        hf_legs = [b for b in bases if b[0] in ("rsi_cross", "rsi_macd", "ema_cross", "fvg_rsi")]
        RNG.shuffle(of_legs)
        RNG.shuffle(hf_legs)
        for (k1, p1), (k2, p2) in itertools.product(of_legs[:20], hf_legs[:30]):
            p1, p2 = {**p1, "sessions": None}, {**p2, "sessions": None}
            for tf, df in frames.items():
                if tf not in ("5m", "15m", "1m"):
                    continue
                try:
                    L1, S1 = get_sig(tf, k1, p1, False)
                    L2, S2 = get_sig(tf, k2, p2, False)
                except Exception:
                    continue
                L, S = L1 & L2, S1 & S2
                if int(L.sum() + S.sum()) < 80:
                    # try OF inverted AND hf
                    try:
                        L1i, S1i = get_sig(tf, k1, p1, True)
                        L, S = L1i & L2, S1i & S2
                    except Exception:
                        continue
                    if int(L.sum() + S.sum()) < 80:
                        continue
                    inv_note = True
                else:
                    inv_note = False
                for ex in exs:
                    tested += 1
                    name = f"and|{'inv-' if inv_note else ''}{k1}+{k2}|{tf}"
                    res = evaluate(
                        name,
                        tf,
                        df,
                        L,
                        S,
                        dict(ex),
                        {"kind": "and2", "a": k1, "b": k2, "pa": p1, "pb": p2, "of_invert": inv_note, "sessions": None},
                    )
                    if not res or res.trades < 30 or res.oos_trades < 15:
                        continue
                    min_wr = min(res.winrate, res.oos_winrate)
                    row = {
                        "name": res.name,
                        "wr": res.winrate,
                        "oos_wr": res.oos_winrate,
                        "min_wr": min_wr,
                        "tpd": res.trades_per_day,
                        "oos_tpd": res.oos_trades_per_day,
                        "n": res.trades,
                        "oos_n": res.oos_trades,
                        "pf": res.profit_factor,
                        "oos_pf": res.oos_profit_factor,
                        "params": res.params,
                        "year_ok": year_stable(df, L, S, dict(ex)),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    with open(LEADER, "a") as f:
                        f.write(json.dumps(row, default=str) + "\n")
                    if best is None or min_wr > best["min_wr"]:
                        best = row
                    if res.trades_per_day >= 1.0 and min_wr >= 0.55:
                        tpd1.append(row)
                        tpd1.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                        tpd1[:] = tpd1[:40]
                    if min_wr >= NEAR:
                        near.append(row)
                        near.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                        near[:] = near[:40]
                        log(
                            f"NEAR-AND {min_wr:.1%} tpd={res.trades_per_day:.2f} "
                            f"n={res.trades}/{res.oos_trades} year={row['year_ok']} | {res.name}"
                        )
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
                    if qualifies(is_m, oos_m) and year_stable(df, L, S, dict(ex)):
                        quals += 1
                        save_champion(res, force=True)
                        write_bot(res)
                        log(f"QUAL AND {res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f} | {res.name}")

        dump(tested, quals, best, near, tpd1)
        log(f"goal pass done tested={tested} quals={quals}; reshuffle (cache={len(cache)})")
        # trim cache if huge
        if len(cache) > 4000:
            cache.clear()


if __name__ == "__main__":
    main()
