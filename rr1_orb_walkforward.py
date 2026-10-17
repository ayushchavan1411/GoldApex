#!/usr/bin/env python3
"""ORB hunter inspired by Benny (@bennnytrades) reel DdCYACaP-ef.

Claims: ~80% WR ORB with negative RR, walk-forward + Monte Carlo.
Our constraints: causal RR1 (rr=1.0/1.2), ATR floor, sessions=None (no session blocks
after OR is set — trade breakouts any hour).

Walk-forward: rolling IS optimize → lock params → OOS next window; require
stability across neighboring configs (not one lucky combo).
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
    simulate,
    write_bot,
    year_stable,
)

RNG = random.Random(13)
LEADER = OUT / "ORB_WF_LEADERBOARD.jsonl"
PROGRESS = OUT / "orb_wf_progress.json"
SUMMARY = OUT / "ORB_WF_SUMMARY.json"
IG_NOTE = OUT / "IG_DdCYACaP-ef_ANALYSIS.json"


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
    """All ORB variants — sessions always None (no trading-session blocks)."""
    out = []
    s = None
    for orb in ("ASIAN", "LONDON", "NY"):
        out.append(("orb_break", {"orb": orb, "sessions": s}))
        out.append(("orb_fade", {"orb": orb, "sessions": s}))
        for win in (15, 30, 45, 60, 90):
            for first in (True, False):
                out.append(
                    ("orb_break_win", {"orb": orb, "win": win, "first_only": first, "sessions": s})
                )
            out.append(("orb_fade_win", {"orb": orb, "win": win, "sessions": s}))
    return out


def walk_forward_score(df, L, S, ex, n_folds: int = 5) -> dict:
    """Rolling chronological folds: train-free (params locked), score OOS folds."""
    n = len(df)
    if n < 2000:
        return {"folds": 0, "oos_wr_mean": 0.0, "oos_pf_mean": 0.0, "profitable_folds": 0, "ok": False}
    # equal chronological slices; evaluate each fold as OOS with prior as burn-in
    fold_size = n // (n_folds + 1)
    oos_wrs, oos_pfs, pnls = [], [], []
    for f in range(n_folds):
        start = fold_size * (f + 1)
        end = min(n - 2, fold_size * (f + 2))
        if end - start < 400:
            continue
        m = simulate(
            df,
            long_e=L,
            short_e=S,
            rr_atr=ex["rr"],
            max_hold=ex["max_hold"],
            start_i=start,
            end_i=end,
            atr_col=ex.get("atr", "ATR_14"),
            allow_long=ex.get("allow_long", True),
            allow_short=ex.get("allow_short", True),
        )
        if m["trades"] < 8:
            continue
        oos_wrs.append(m["winrate"])
        oos_pfs.append(m["profit_factor"])
        pnls.append(m["net_pnl"])
    if len(oos_wrs) < 3:
        return {"folds": len(oos_wrs), "oos_wr_mean": 0.0, "oos_pf_mean": 0.0, "profitable_folds": 0, "ok": False}
    arr = np.array(pnls)
    # Benny-style: majority of folds profitable; no single fold >50% of total profit
    total = float(arr[arr > 0].sum()) if (arr > 0).any() else 0.0
    max_share = float(arr.max() / total) if total > 0 else 1.0
    profitable = int((arr > 0).sum())
    ok = profitable >= max(2, len(arr) // 2 + 1) and max_share <= 0.55
    return {
        "folds": len(oos_wrs),
        "oos_wr_mean": float(np.mean(oos_wrs)),
        "oos_wr_min": float(np.min(oos_wrs)),
        "oos_pf_mean": float(np.mean(oos_pfs)),
        "profitable_folds": profitable,
        "max_profit_share": max_share,
        "ok": ok,
    }


def dump(tested, quals, best, near, wf_ok):
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals": quals,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best["name"] if best else None,
                "near": len(near),
                "wf_ok": len(wf_ok),
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
                "walk_forward_ok": wf_ok[:15],
                "sessions": "ALL (None)",
                "source": "Instagram reel DdCYACaP-ef @bennnytrades ORB walk-forward",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )


def main():
    OUT.mkdir(exist_ok=True)
    IG_NOTE.write_text(
        json.dumps(
            {
                "url": "https://www.instagram.com/reel/DdCYACaP-ef/",
                "author": "bennnytrades (Benny)",
                "claim": "~80% WR ORB over 5y, negative RR, walk-forward + Monte Carlo + live prop payouts",
                "mechanics": "Auto-detect opening range → enter on range cross (gold bar) → exit coded",
                "validation": "Walk-forward (IS optimize → lock → OOS), stability across neighbors, no single period >50% profit",
                "our_action": "Expand OR windows 15–90m, sessions=None, RR1 causal, walk-forward fold scoring on XAU",
                "goal_fit": "High WR + shallow losers aligns with RR1 goal if tpd≥1 and year_stable hold",
            },
            indent=2,
        )
    )
    log("=== ORB WALK-FORWARD HUNT (Benny reel) sessions=None RR1 ===")
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("5m", "15m", "30m"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])

    sp = specs()
    exs = exits()
    log(f"specs={len(sp)} exits={len(exs)} tfs={list(frames)}")

    tested = quals = 0
    best = None
    near: list[dict] = []
    wf_ok: list[dict] = []

    while True:
        RNG.shuffle(sp)
        for kind, base_p in sp:
            p = {**base_p, "sessions": None}
            for invert in (False, True):
                pp = {**p, "invert": invert} if invert else dict(p)
                for tf, df in frames.items():
                    try:
                        L, S = gen_signals(df, kind, pp)
                    except Exception:
                        continue
                    if int(L.sum() + S.sum()) < 60:
                        continue
                    for ex in exs:
                        tested += 1
                        tag = f"{'inv|' if invert else 'orb|'}{kind}|{tf}"
                        res = evaluate(tag, tf, df, L, S, dict(ex), {"kind": kind, **pp})
                        if not res or res.trades < 25 or res.oos_trades < 12:
                            continue
                        wf = walk_forward_score(df, L, S, dict(ex))
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
                            "wf": wf,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        if best is None or min_wr > best["min_wr"]:
                            best = row
                        if wf.get("ok") and wf.get("oos_wr_mean", 0) >= 0.55:
                            wf_ok.append(row)
                            wf_ok.sort(key=lambda r: (r["wf"]["oos_wr_mean"], r["min_wr"]), reverse=True)
                            wf_ok[:] = wf_ok[:40]
                        if min_wr >= 0.65 and res.trades >= 30:
                            near.append(row)
                            near.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                            near[:] = near[:40]
                            log(
                                f"NEAR ORB {min_wr:.1%} tpd={res.trades_per_day:.2f} "
                                f"wf_ok={wf.get('ok')} wf_wr={wf.get('oos_wr_mean', 0):.1%} | {res.name}"
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
                        if (
                            qualifies(is_m, oos_m)
                            and year_stable(df, L, S, dict(ex))
                            and wf.get("ok")
                        ):
                            quals += 1
                            save_champion(res, force=True)
                            write_bot(res)
                            log(f"QUAL ORB-WF {res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f} | {res.name}")
                        if tested % 150 == 0:
                            dump(tested, quals, best, near, wf_ok)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            log(f"orb-wf tested={tested} quals={quals} best={bw} near={len(near)} wf_ok={len(wf_ok)}")

        dump(tested, quals, best, near, wf_ok)
        log(f"orb-wf pass done tested={tested} quals={quals}; reshuffle")


if __name__ == "__main__":
    main()
