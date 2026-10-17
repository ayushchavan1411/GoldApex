#!/usr/bin/env python3
"""ICT glossary hunt from Instagram reel DemxNo2IIjs (@mohamad.trades).

Reel lists ICT/SMC terms (FVG, IFVG, CE, OB, BB, OTE, BOS/MSS, PDH/PDL,
NDOG/NWOG, EQH/EQL, BPR, CISD, AMD, POI, 1PFVG, …). We encode causal proxies
and hunt for WR≥75% IS+OOS, tpd≥1, RR1, year_stable, sessions=None.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

from rr1_search import (
    OUT,
    TARGET_WR,
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

RNG = random.Random(75)
LEADER = OUT / "MOHAMAD_ICT_LEADERBOARD.jsonl"
PROGRESS = OUT / "mohamad_ict_progress.json"
SUMMARY = OUT / "MOHAMAD_ICT_SUMMARY.json"
NOTE = OUT / "IG_DemxNo2IIjs_ANALYSIS.json"


def exits():
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


def specs():
    """All-session ICT glossary entries (+ filters / stacks)."""
    s = None
    out = []
    # Core glossary singles
    for kind, extra in (
        ("fvg_reclaim", {}),
        ("fvg_ce", {}),
        ("ifvg", {"look": 12}),
        ("ifvg", {"look": 20}),
        ("fvg1_reclaim", {}),
        ("bpr_reclaim", {}),
        ("breaker_retest", {}),
        ("cisd", {}),
        ("ndog_fade", {}),
        ("nwog_fade", {}),
        ("ob_retest", {}),
        ("ob_fvg", {}),
        ("mss_entry", {"swing": 20}),
        ("mss_entry", {"swing": 40}),
        ("eq_fade", {"swing": 20}),
    ):
        out.append((kind, {**extra, "sessions": s}))
    for mode in ("fade", "break"):
        out.append(("pdh_pdl", {"mode": mode, "sessions": s}))
        out.append(("pwh_pwl", {"mode": mode, "sessions": s}))
    for sw, wick in itertools.product((10, 20), (0.4, 0.5)):
        out.append(("eqh_eql_sweep", {"swing": sw, "wick": wick, "sessions": s}))
        out.append(("liq_sweep", {"swing": sw, "wick": wick, "sessions": s}))
        out.append(("liq_sweep_fvg", {"swing": sw, "wick": wick, "sessions": s}))
        out.append(("sweep_ob", {"swing": sw, "sessions": s}))
    for sw, need in itertools.product((20, 40), (False, True)):
        out.append(("ote_entry", {"swing": sw, "need_sweep": need, "sessions": s}))
    for sw, need in itertools.product((20, 40), (False, True)):
        out.append(("poi_stack", {"swing": sw, "need_mss": need, "look": 20, "sessions": s}))
    for sw, look in ((20, 12), (20, 18), (40, 24)):
        out.append(("amd_sweep_mss", {"swing": sw, "look": look, "sessions": s}))
    # FVG + RSI (existing champ family)
    for rsi, os_, ob in ((14, 30, 70), (14, 35, 65), (14, 40, 80), (7, 30, 70)):
        out.append(("fvg_rsi", {"rsi": rsi, "os": os_, "ob": ob, "sessions": s}))
    # Fib OTE-adjacent
    for sw, fib in itertools.product((20, 40), ("618", "786", "500")):
        out.append(("fib_bounce", {"swing": sw, "fib": fib, "wick": 0.45, "sessions": s}))
        out.append(("fib_discount", {"swing": sw, "fib": fib, "sessions": s}))
    return out


def dump(tested, quals, best, near):
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals": quals,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best["name"] if best else None,
                "best_tpd": best.get("tpd") if best else None,
                "near": len(near),
                "gate": f"WR>={TARGET_WR:.0%} tpd>=1 RR1 sessions=None",
                "source": "IG reel DemxNo2IIjs @mohamad.trades ICT glossary",
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
                "near_hits": near[:20],
                "source": "https://www.instagram.com/reel/DemxNo2IIjs/",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )


def main():
    OUT.mkdir(exist_ok=True)
    NOTE.write_text(
        json.dumps(
            {
                "url": "https://www.instagram.com/reel/DemxNo2IIjs/",
                "author": "mohamad.trades",
                "type": "ICT/SMC terminology checklist (not a single coded bot)",
                "terms": [
                    "FVG", "IFVG", "CE", "OB", "BB", "RB", "MB", "CISD",
                    "HTF", "LTF", "OTE", "BOS", "MSS", "PDH", "PDL", "PWH", "PWL",
                    "DOL", "EQH", "EQL", "BSL", "SSL", "POI", "NDOG", "NWOG",
                    "MMXM", "AMD", "EQ", "BPR", "IRL", "ERL", "SMT", "1PFVG",
                ],
                "encoded_kinds": [
                    "fvg_ce", "ifvg", "fvg1_reclaim", "ote_entry", "pdh_pdl",
                    "pwh_pwl", "ndog_fade", "nwog_fade", "eqh_eql_sweep",
                    "breaker_retest", "cisd", "bpr_reclaim", "poi_stack",
                    "amd_sweep_mss",
                ],
                "goal": "WR≥75% IS+OOS, tpd≥1, RR1, year_stable, sessions=None",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    log("=== MOHAMAD ICT GLOSSARY HUNT (DemxNo2IIjs) → WR≥75% sessions=None ===")
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

    while True:
        RNG.shuffle(sp)
        for kind, base in sp:
            p = {**base, "sessions": None}
            for invert in (False, True):
                pp = {**p, "invert": invert} if invert else dict(p)
                for tf, df in frames.items():
                    try:
                        L, S = gen_signals(df, kind, pp)
                    except Exception as e:
                        if tested < 5:
                            log(f"skip {kind}: {e}")
                        continue
                    if int(L.sum() + S.sum()) < 60:
                        continue
                    for ex in exs:
                        tested += 1
                        tag = f"{'inv|' if invert else 'ict|'}{kind}|{tf}"
                        res = evaluate(tag, tf, df, L, S, dict(ex), {"kind": kind, **pp})
                        if not res or res.trades < 40 or res.oos_trades < 20:
                            continue
                        min_wr = min(res.winrate, res.oos_winrate)
                        yok = year_stable(df, L, S, dict(ex))
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
                            "year_ok": yok,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        sk = (
                            1 if (row["tpd"] >= 1.0 and min_wr >= 0.60) else 0,
                            min_wr,
                            row["tpd"],
                        )
                        if best is None or sk > best.get("_sk", (0, 0, 0)):
                            best = {**row, "_sk": sk}
                        if min_wr >= 0.65 and res.trades_per_day >= 0.8:
                            near.append(row)
                            near.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                            near[:] = near[:40]
                            log(
                                f"NEAR ICT {min_wr:.1%} tpd={res.trades_per_day:.2f} "
                                f"year={yok} | {res.name}"
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
                        if qualifies(is_m, oos_m) and yok:
                            quals += 1
                            if save_champion(res, force=True):
                                write_bot(res)
                            log(
                                f"QUAL ICT {res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f} | {res.name}"
                            )
                        if tested % 150 == 0:
                            dump(tested, quals, best, near)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            log(f"mohamad-ict tested={tested} quals={quals} best={bw} near={len(near)}")

        dump(tested, quals, best, near)
        log(f"mohamad-ict pass done tested={tested} quals={quals}; reshuffle")


if __name__ == "__main__":
    main()
