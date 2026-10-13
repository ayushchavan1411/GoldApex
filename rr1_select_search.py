#!/usr/bin/env python3
"""High-selectivity AND hunter: stack ICT/classic filters to push WR toward 75%+ at ≥1 tpd on 2y."""
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

RNG = random.Random(123)
LEADER = OUT / "LEADERBOARD.jsonl"


def exits():
    return [
        {"rr": rr, "max_hold": mh, "atr": atr, **side}
        for rr in (0.5, 0.6, 0.8, 1.0, 1.2)
        for mh in (2, 3, 4, 6, 8, 12)
        for atr in ("ATR_14", "ATR_7")
        for side in (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        )
    ]


def track(res, tag=""):
    if not res:
        return
    row = {
        "tag": tag,
        "name": res.name,
        "wr": res.winrate,
        "oos_wr": res.oos_winrate,
        "tpd": res.trades_per_day,
        "oos_tpd": res.oos_trades_per_day,
        "n": res.trades,
        "oos_n": res.oos_trades,
        "pf": res.profit_factor,
        "oos_pf": res.oos_profit_factor,
        "pnl": res.net_pnl,
        "oos_pnl": res.oos_net_pnl,
        "score": res.score,
        "params": res.params,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(LEADER, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    log("=== SELECTIVITY AND HUNT (≥75% / 1:1 / ≥1tpd / 2y) ===")
    df = add_indicators(load_frames()["5m"])
    log(f"bars={len(df)} days={(df.index[-1]-df.index[0]).days}")

    # Build library of moderately selective signals
    lib = []
    specs = []
    sessions = [None, ("london", "overlap"), ("overlap", "ny"), ("london", "overlap", "ny"), ("asian", "london", "overlap", "ny")]
    for sess in sessions:
        specs.append(("fvg_reclaim", {"sessions": sess}))
        specs.append(("ob_retest", {"sessions": sess}))
        for sw in (10, 20, 40):
            specs.append(("liq_sweep", {"swing": sw, "wick": 0.5, "sessions": sess}))
            specs.append(("smc_confluence", {"swing": sw, "sessions": sess}))
            for fib in ("500", "618", "786"):
                specs.append(("fib_discount", {"swing": sw, "fib": fib, "sessions": sess}))
                specs.append(("fib_bounce", {"swing": sw, "fib": fib, "wick": 0.45, "sessions": sess}))
        for rsi, os_, ob in itertools.product([3, 7], [20, 25, 30], [70, 75, 80]):
            specs.append(("rsi_level", {"rsi": rsi, "os": os_, "ob": ob, "band": 5, "sessions": sess}))
            specs.append(("fvg_rsi", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}))
        for per, k in ((20, 2.0), (20, 2.5), (10, 2.0)):
            specs.append(("bb_fade", {"period": per, "k": k, "sessions": sess}))
            specs.append(("bb_rsi_fade", {"period": per, "k": k, "rsi": 7, "os": 30, "ob": 70, "sessions": sess}))

    for kind, p in specs:
        try:
            L, S = gen_signals(df, kind, p)
        except Exception:
            continue
        n = int(L.sum() + S.sum())
        if 60 <= n <= 8000:
            lib.append((kind, p, L, S, n))
    log(f"library={len(lib)}")

    # Score singles for WR
    exs = exits()
    scored = []
    for kind, p, L, S, n in lib:
        best = None
        for ex in exs[::4]:
            res = evaluate(f"sel|{kind}", "5m", df, L, S, ex, {"kind": kind, **p})
            if res and (best is None or min(res.winrate, res.oos_winrate) > min(best.winrate, best.oos_winrate)):
                best = res
        if best:
            scored.append((best, kind, p, L, S))
    scored.sort(key=lambda x: (min(x[0].winrate, x[0].oos_winrate), x[0].trades_per_day), reverse=True)
    log(
        f"top single minWR={min(scored[0][0].winrate, scored[0][0].oos_winrate):.1%} "
        f"WR={scored[0][0].winrate:.1%}/{scored[0][0].oos_winrate:.1%} tpd={scored[0][0].trades_per_day:.2f}"
        if scored
        else "no singles"
    )
    for b, *_ in scored[:8]:
        track(b, "single")
        log(f"  single WR={b.winrate:.1%}/{b.oos_winrate:.1%} tpd={b.trades_per_day:.2f}/{b.oos_trades_per_day:.2f} n={b.trades}/{b.oos_trades} {b.name}")

    tested = 0
    quals = 0
    best = scored[0][0] if scored else None
    if best and best.winrate >= 0.60:
        save_champion(best)
        write_bot(best)

    top = scored[:30]
    while True:
        # AND pairs
        for i, j in itertools.combinations(range(min(20, len(top))), 2):
            a, ka, pa, La, Sa = top[i]
            b, kb, pb, Lb, Sb = top[j]
            if ka == kb:
                continue
            L = La & Lb
            S = Sa & Sb
            both = L & S
            L = L & ~both
            S = S & ~both
            if L.sum() + S.sum() < 100:
                continue
            for ex in exs[::3]:
                tested += 1
                res = evaluate(f"and|{ka}+{kb}", "5m", df, L, S, ex, {"kind": "combo_and", "a": ka, "b": kb})
                if not res:
                    continue
                track(res, "and2")
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
                if q:
                    q = year_stable(df, L, S, ex)
                key = (min(res.winrate, res.oos_winrate), min(res.trades_per_day, res.oos_trades_per_day), res.trades + res.oos_trades)
                bkey = (
                    (min(best.winrate, best.oos_winrate), min(best.trades_per_day, best.oos_trades_per_day), best.trades + best.oos_trades)
                    if best
                    else (-1,)
                )
                if q or key > bkey:
                    if res.winrate >= 0.60 or q:
                        best = res
                        save_champion(res)
                        write_bot(res)
                        if q:
                            quals += 1
                        log(
                            f"{'QUALIFIED ' if q else ''}CHAMP WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades} {res.name}"
                        )
            if tested % 500 == 0:
                log(f"select tested={tested} quals={quals} best={(f'{best.winrate:.1%}/{best.oos_winrate:.1%}' if best else 'n/a')}")
                Path(OUT / "select_progress.json").write_text(
                    json.dumps({"tested": tested, "quals": quals, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
                )

        # AND triples from top 12
        for i, j, k in itertools.combinations(range(min(12, len(top))), 3):
            La, Lb, Lc = top[i][3], top[j][3], top[k][3]
            Sa, Sb, Sc = top[i][4], top[j][4], top[k][4]
            L = La & Lb & Lc
            S = Sa & Sb & Sc
            both = L & S
            L, S = L & ~both, S & ~both
            if L.sum() + S.sum() < 80:
                continue
            for ex in exs[::5]:
                tested += 1
                res = evaluate("and3", "5m", df, L, S, ex, {"kind": "combo_and3"})
                if not res:
                    continue
                track(res, "and3")
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
                if q and year_stable(df, L, S, ex):
                    quals += 1
                    save_champion(res)
                    write_bot(res)
                    best = res
                    log(f"QUALIFIED and3 WR={res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f}")
                elif best is None or min(res.winrate, res.oos_winrate) > min(best.winrate, best.oos_winrate):
                    if res.winrate >= 0.62 and res.trades_per_day >= 0.8:
                        best = res
                        save_champion(res)
                        write_bot(res)
                        log(f"NEAR and3 WR={res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f}")

        # mutate top by adding hour filters
        for b, kind, p, L0, S0 in top[:15]:
            for hours in (tuple(range(13, 20)), tuple(range(8, 16)), tuple(range(14, 21)), tuple(range(7, 22))):
                pp = dict(p)
                pp["hours"] = hours
                try:
                    L, S = gen_signals(df, kind, pp)
                except Exception:
                    continue
                if L.sum() + S.sum() < 80:
                    continue
                for ex in exs[::6]:
                    tested += 1
                    res = evaluate(f"hr|{kind}", "5m", df, L, S, ex, {"kind": kind, **pp})
                    if not res:
                        continue
                    if res.winrate >= 0.65 and res.oos_winrate >= 0.65:
                        track(res, "hours")
                        log(f"NEAR hours WR={res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} {kind}")
                        if best is None or min(res.winrate, res.oos_winrate) > min(best.winrate, best.oos_winrate):
                            best = res
                            save_champion(res)
                            write_bot(res)

        log(f"select pass done tested={tested} quals={quals}; reshuffle+loop")
        RNG.shuffle(top)


if __name__ == "__main__":
    main()
