#!/usr/bin/env python3
"""Order-flow proxy hunter for XAU RR1.

No L2/tick volume in local CSVs — uses price-based OF proxies:
  CLV, signed delta, CVD, stacked imbalances, absorption, CVD divergence,
  sweep+absorb, delta flip. Also tests invert (fade) of weak OF bases.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

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

RNG = random.Random(77)
LEADER = OUT / "ORDERFLOW_LEADERBOARD.jsonl"
PROGRESS = OUT / "orderflow_progress.json"
SUMMARY = OUT / "ORDERFLOW_SUMMARY.json"

NEAR_WR = 0.65


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
    """All-session only (sessions=None) — do not block Asian/London/NY/overlap."""
    out = []
    sess = None  # trade every hour
    for lo, hi in ((0.15, 0.85), (0.20, 0.80), (0.25, 0.75), (0.30, 0.70), (0.35, 0.65)):
        out.append(("of_clv_fade", {"lo": lo, "hi": hi, "sessions": sess}))
    for n in (2, 3, 4, 5, 6):
        out.append(("of_stacked_imb", {"n": n, "sessions": sess}))
        out.append(("of_stacked_fade", {"n": n, "sessions": sess}))
        out.append(("of_delta_flip", {"n": n, "sessions": sess}))
        out.append(("of_imb_absorb", {"n": n, "sessions": sess}))
    out.append(("of_absorption", {"sessions": sess}))
    out.append(("of_sweep_absorb", {"sessions": sess}))
    for span in (5, 8, 13, 20, 34, 50):
        out.append(("of_cvd_cross", {"span": span, "sessions": sess}))
        for rsi, os_, ob in (
            (3, 25, 75),
            (5, 25, 75),
            (5, 30, 70),
            (7, 30, 70),
            (7, 35, 65),
            (14, 30, 70),
            (14, 35, 65),
            (14, 40, 60),
        ):
            out.append(("of_cvd_rsi", {"span": span, "rsi": rsi, "os": os_, "ob": ob, "sessions": sess}))
    for look in (8, 10, 14, 20, 30):
        out.append(("of_cvd_div", {"look": look, "sessions": sess}))
    seen, uniq = set(), []
    for kind, p in out:
        key = (kind, json.dumps(p, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((kind, p))
    return uniq


def append_row(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def dump_progress(tested, quals, best, near):
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals": quals,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best["name"] if best else None,
                "near": len(near),
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
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Price-based order-flow proxies (no L2/volume in XAU CSV)",
            },
            indent=2,
            default=str,
        )
    )


def main():
    OUT.mkdir(exist_ok=True)
    log("=== ORDER-FLOW PROXY HUNT (CLV/CVD/imbalance/absorption) ===")
    raw = load_frames(include_resampled=True)
    frames = {}
    # Skip 1m — OF indicators + 910k bars OOMs the box; 5m+ is enough for OF structure
    for tf in ("5m", "15m", "30m", "1h"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])

    sp = specs()
    exs = exits()
    log(f"specs={len(sp)} exits={len(exs)} tfs={list(frames)}")

    tested = 0
    quals = 0
    best = None
    near: list[dict] = []
    champion_holder = [None]

    while True:
        RNG.shuffle(sp)
        for kind, base_p in sp:
            for invert in (False, True):
                p = {**base_p, "invert": invert} if invert else dict(base_p)
                tag = f"{'inv|' if invert else 'of|'}{kind}"
                for tf, df in frames.items():
                    try:
                        L, S = gen_signals(df, kind, p)
                    except Exception:
                        continue
                    if int(L.sum() + S.sum()) < 80:
                        continue
                    for ex in exs:
                        tested += 1
                        res = evaluate(f"{tag}|{tf}", tf, df, L, S, dict(ex), {"kind": kind, **p})
                        if not res or res.trades < 25 or res.oos_trades < 12:
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
                            "pnl": res.net_pnl,
                            "oos_pnl": res.oos_net_pnl,
                            "params": res.params,
                            "invert": invert,
                            "year_ok": year_stable(df, L, S, dict(ex)),
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        append_row(LEADER, row)
                        if best is None or min_wr > best["min_wr"] or (
                            min_wr == best["min_wr"] and row["n"] + row["oos_n"] > best["n"] + best["oos_n"]
                        ):
                            best = row

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
                        q = qualifies(is_m, oos_m) and year_stable(df, L, S, dict(ex))
                        if res.winrate >= NEAR_WR and res.oos_winrate >= NEAR_WR and res.trades >= 30:
                            near.append(row)
                            near.sort(key=lambda r: (r["min_wr"], r["n"] + r["oos_n"]), reverse=True)
                            near[:] = near[:40]
                            log(
                                f"NEAR OF WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                                f"n={res.trades}/{res.oos_trades} year={row['year_ok']} | {res.name}"
                            )
                        if q:
                            quals += 1
                            champion_holder[0] = res
                            save_champion(res, force=True)
                            write_bot(res)
                            log(
                                f"QUAL OF WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f} | {res.name}"
                            )
                        if tested % 200 == 0:
                            dump_progress(tested, quals, best, near)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            log(f"of tested={tested} quals={quals} best_minWR={bw} near={len(near)}")

        dump_progress(tested, quals, best, near)
        log(f"of pass done tested={tested} quals={quals}; reshuffle")


if __name__ == "__main__":
    main()
