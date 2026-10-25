#!/usr/bin/env python3
"""High-frequency signal hunt under 1% price risk/reward (no ATR RR).

User (2026-10-18): drop ATR-based RR; SL=TP=1% of price, fixed 0.02 lot + fees.
Gate: WR≥75% IS+OOS, tpd≥1, year_stable (classic).
"""
from __future__ import annotations

import itertools
import json
import random
import time

from rr1_search import (
    OUT,
    PCT_RR,
    add_indicators,
    evaluate,
    fee_pct_exits,
    gen_signals,
    load_frames,
    log,
    qualifies,
    save_champion,
    write_bot,
    year_stable,
)

RNG = random.Random(60)
LEADER = OUT / "POSRR_LEADERBOARD.jsonl"
PROGRESS = OUT / "posrr_progress.json"
SUMMARY = OUT / "POSRR_SUMMARY.json"


def exits():
    """User: drop ATR RR — 1% price risk/reward only."""
    return fee_pct_exits(PCT_RR, holds=(12, 18, 24, 36, 48, 72))


def specs():
    """High-frequency-leaning kinds, sessions always None."""
    s = None
    out = []
    # OF proxies
    for n in (2, 3, 4):
        out.append(("of_stacked_imb", {"n": n, "sessions": s}))
        out.append(("of_stacked_fade", {"n": n, "sessions": s}))
        out.append(("of_imb_absorb", {"n": n, "sessions": s}))
        out.append(("of_delta_flip", {"n": n, "sessions": s}))
    out.append(("of_absorption", {"sessions": s}))
    out.append(("of_sweep_absorb", {"sessions": s}))
    for span in (8, 20):
        out.append(("of_cvd_cross", {"span": span, "sessions": s}))
    for look in (10, 20):
        out.append(("of_cvd_div", {"look": look, "sessions": s}))
    for lo, hi in ((0.2, 0.8), (0.25, 0.75)):
        out.append(("of_clv_fade", {"lo": lo, "hi": hi, "sessions": s}))
    for span, rsi, os_, ob in itertools.product((20,), (14,), (30, 35), (65, 70)):
        out.append(("of_cvd_rsi", {"span": span, "rsi": rsi, "os": os_, "ob": ob, "sessions": s}))
    # ORB / retest / OF-ORB
    for orb in ("ASIAN", "LONDON", "NY"):
        out.append(("orb_break", {"orb": orb, "sessions": s}))
        out.append(("orb_fade", {"orb": orb, "sessions": s}))
        for win in (15, 30, 45, 60, 90):
            for first in (True, False):
                out.append(("orb_break_win", {"orb": orb, "win": win, "first_only": first, "sessions": s}))
            out.append(("orb_fade_win", {"orb": orb, "win": win, "sessions": s}))
            out.append(("orb_retest_win", {"orb": orb, "win": win, "sessions": s}))
            out.append(("orb_of_break", {"orb": orb, "win": win, "n": 2, "sessions": s}))
            out.append(("orb_of_fade", {"orb": orb, "win": win, "sessions": s}))
    # Classic mean-revert / momentum (full required params)
    for kind, extra in (
        ("rsi_level", {"rsi": 14, "os": 30, "ob": 70}),
        ("rsi_level", {"rsi": 14, "os": 35, "ob": 65}),
        ("rsi_cross", {"rsi": 14, "os": 30, "ob": 70}),
        ("rsi_macd", {"rsi": 14, "os": 35, "ob": 65}),
        ("bb_rsi_fade", {"period": 20, "k": 2.0, "rsi": 14, "os": 35, "ob": 65}),
        ("bb_mid_reject", {"period": 20, "k": 2.0}),
        ("ema_cross", {"fast": 8, "slow": 21}),
        ("ema_cross", {"fast": 12, "slow": 26}),
        ("fvg_rsi", {"rsi": 14, "os": 40, "ob": 80}),
        ("mss_entry", {}),
        ("zscore_fade", {"period": 20, "th": 2.0}),
        ("keltner_fade", {"period": 20, "mult": 1.5}),
    ):
        out.append((kind, {**extra, "sessions": s}))
    return out


def dump(tested, quals, best, near):
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals": quals,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best["name"] if best else None,
                "best_geo_rr": best.get("geo_rr") if best else None,
                "best_tpd": best.get("tpd") if best else None,
                "near": len(near),
                "gate": f"1% price RR (no ATR) WR>=75% tpd>=1",
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
                "gate": {
                    "wr": 0.75,
                    "pct_rr": PCT_RR,
                    "tpd": 1.0,
                    "sessions": None,
                },
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )


def main():
    OUT.mkdir(exist_ok=True)
    log(f"=== 1% PRICE RR HUNT (was posrr) WR>=75% tpd>=1 sessions=None ===")
    raw = load_frames(include_resampled=True)
    frames = {}
    # Skip 1h to keep RSS under control alongside other hunters
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
                    except Exception:
                        continue
                    if int(L.sum() + S.sum()) < 80:
                        continue
                    for ex in exs:
                        tested += 1
                        tag = f"{'inv|' if invert else 'pos|'}{kind}|{tf}"
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
                            "net_pnl": res.net_pnl,
                            "params": res.params,
                            "year_ok": yok,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")

                        score_key = (
                            1 if row["tpd"] >= 1.0 and row["oos_tpd"] >= 0.85 else 0,
                            min_wr,
                            row["tpd"],
                        )
                        if best is None or score_key > best.get("_sk", (0, 0, 0)):
                            best = {**row, "_sk": score_key}

                        if min_wr >= 0.65 and res.trades_per_day >= 0.7 and res.net_pnl > 0:
                            near.append(row)
                            near.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                            near[:] = near[:40]
                            log(
                                f"NEAR 1pct {min_wr:.1%} tpd={res.trades_per_day:.2f} "
                                f"net=${res.net_pnl:.1f} year={yok} | {res.name}"
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
                                f"QUAL 1PCT {res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f} net=${res.net_pnl:.1f} | {res.name}"
                            )

                        if tested % 200 == 0:
                            dump(tested, quals, best, near)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            log(
                                f"posrr tested={tested} quals={quals} best={bw} "
                                f"tpd={best.get('tpd') if best else None} near={len(near)}"
                            )

        dump(tested, quals, best, near)
        log(f"posrr pass done tested={tested} quals={quals}; reshuffle")


if __name__ == "__main__":
    main()
