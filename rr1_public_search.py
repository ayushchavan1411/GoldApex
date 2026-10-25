#!/usr/bin/env python3
"""
Never-stop hunter for PUBLIC free strategies + combinations on 2y XAU 5m.
Sources inspired by common free playbooks: Ichimoku, Heikin Ashi, Turtle/Donchian,
ORB, Keltner, RSI+MACD confluence, EMA cross/pullback, ADX DI, CCI/Z-score, Supertrend.
Goal: ≥75% WR, 1:1 RR, ≥1 tpd, year-stable.
"""
from __future__ import annotations

import itertools
import json
import random
import time
import traceback
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

RNG = random.Random(2026)
SESS = [
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
PUBLIC_KINDS_BUILDERS = []


def build_space():
    sp = []
    for sessions in SESS:
        sp += [
            ("ichimoku_tk", {"sessions": sessions}),
            ("ichimoku_kumo", {"sessions": sessions}),
            ("heikin_flip", {"sessions": sessions}),
        ]
        for ema in (20, 50, 200):
            sp.append(("heikin_ema", {"ema": ema, "sessions": sessions}))
        for rsi, os_, ob in itertools.product([7, 14], [35, 40, 45], [55, 60, 65]):
            sp.append(("rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))
        for fast, slow in ((5, 20), (9, 21), (12, 26), (21, 50)):
            sp.append(("ema_cross", {"fast": fast, "slow": slow, "sessions": sessions}))
            for rsi, os_, ob in ((2, 30, 70), (7, 35, 65), (14, 40, 60)):
                sp.append(
                    (
                        "ema_pull_rsi",
                        {"fast": fast, "slow": slow, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions},
                    )
                )
        for period, mult in ((20, 1.5), (20, 2.0), (10, 1.5)):
            sp.append(("keltner_fade", {"period": period, "mult": mult, "sessions": sessions}))
        for period, th in ((20, 1.5), (20, 2.0), (40, 2.0), (40, 2.5)):
            sp.append(("zscore_fade", {"period": period, "th": th, "sessions": sessions}))
        for period, th in ((14, 100), (14, 150), (20, 100), (20, 200)):
            sp.append(("cci_fade", {"period": period, "th": th, "sessions": sessions}))
        for adx in (15, 20, 25, 30):
            sp.append(("adx_di_cross", {"adx": adx, "sessions": sessions}))
        for mult in (2.0, 3.0):
            sp.append(("supertrend_flip", {"mult": mult, "sessions": sessions}))
        for period in (10, 20, 40):
            sp.append(("turtle_break", {"period": period, "sessions": sessions}))
        for orb in ("ASIAN", "LONDON", "NY"):
            sp.append(("orb_break", {"orb": orb, "sessions": sessions}))
            sp.append(("orb_fade", {"orb": orb, "sessions": sessions}))
        # also keep strong SMC/classic mix-ins for AND later
        sp.append(("fvg_reclaim", {"sessions": sessions}))
        sp.append(("liq_sweep", {"swing": 20, "wick": 0.5, "sessions": sessions}))
        sp.append(("smc_confluence", {"swing": 20, "sessions": sessions}))
        for rsi, os_, ob in ((3, 20, 80), (7, 25, 75), (14, 30, 70)):
            sp.append(("rsi_level", {"rsi": rsi, "os": os_, "ob": ob, "band": 5, "sessions": sessions}))
    # hour / vol variants
    extras = []
    for kind, p in sp:
        for hours in (None, tuple(range(8, 17)), tuple(range(13, 21)), tuple(range(7, 22))):
            for vol in (False, True):
                pp = dict(p)
                if hours:
                    pp["hours"] = hours
                if vol:
                    pp["vol_ok"] = True
                extras.append((kind, pp))
    RNG.shuffle(extras)
    return extras


def exits():
    return [
        {"rr": rr, "max_hold": mh, "atr": atr, **side}
        for rr in (0.5, 0.8, 1.0, 1.2, 1.5)
        for mh in (2, 3, 4, 6, 8, 12, 18)
        for atr in ("ATR_14", "ATR_7")
        for side in (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        )
    ]


def main():
    OUT.mkdir(exist_ok=True)
    log("=== PUBLIC STRATEGY HUNT (Ichimoku/HA/Turtle/ORB/Keltner/RSI+MACD/…) ===")
    df = add_indicators(load_frames()["5m"])
    log(f"bars={len(df)} {df.index[0]} → {df.index[-1]}")
    ex_all = exits()
    tested = 0
    quals = 0
    best = None
    leader = OUT / "PUBLIC_LEADERBOARD.jsonl"

    # Precompute library for AND combos
    lib = []
    base_space = []
    for sessions in SESS[:6]:
        for kind, p in [
            ("ichimoku_tk", {"sessions": sessions}),
            ("heikin_ema", {"ema": 50, "sessions": sessions}),
            ("rsi_macd", {"rsi": 14, "os": 40, "ob": 60, "sessions": sessions}),
            ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": sessions}),
            ("keltner_fade", {"period": 20, "mult": 2.0, "sessions": sessions}),
            ("turtle_break", {"period": 20, "sessions": sessions}),
            ("orb_fade", {"orb": "LONDON", "sessions": sessions}),
            ("orb_break", {"orb": "LONDON", "sessions": sessions}),
            ("adx_di_cross", {"adx": 20, "sessions": sessions}),
            ("zscore_fade", {"period": 20, "th": 2.0, "sessions": sessions}),
            ("smc_confluence", {"swing": 20, "sessions": sessions}),
            ("fvg_reclaim", {"sessions": sessions}),
            ("liq_sweep", {"swing": 20, "wick": 0.5, "sessions": sessions}),
            ("rsi_level", {"rsi": 7, "os": 25, "ob": 75, "band": 5, "sessions": sessions}),
        ]:
            base_space.append((kind, p))
    for kind, p in base_space:
        try:
            L, S = gen_signals(df, kind, p)
        except Exception as e:
            log(f"skip {kind}: {e}")
            continue
        n = int(L.sum() + S.sum())
        if 50 <= n <= 15000:
            lib.append((kind, p, L, S))
    log(f"AND library={len(lib)}")

    while True:
        for kind, params in build_space():
            try:
                L, S = gen_signals(df, kind, params)
            except Exception:
                continue
            if L.sum() + S.sum() < 80:
                continue
            for ex in ex_all[:: max(1, len(ex_all) // 18)][:18]:
                tested += 1
                res = evaluate(f"pub|{kind}|5m", "5m", df, L, S, ex, {"kind": kind, **params})
                if not res:
                    continue
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
                if res.winrate >= 0.60:
                    with open(leader, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "wr": res.winrate,
                                    "oos_wr": res.oos_winrate,
                                    "tpd": res.trades_per_day,
                                    "oos_tpd": res.oos_trades_per_day,
                                    "n": res.trades,
                                    "oos_n": res.oos_trades,
                                    "name": res.name,
                                    "params": res.params,
                                },
                                default=str,
                            )
                            + "\n"
                        )
                if best is None or res.score > best.score:
                    if res.winrate >= 0.60 and res.net_pnl > 0:
                        best = res
                        save_champion(res)
                        write_bot(res)
                        log(
                            f"{'QUALIFIED ' if q else ''}NEW CHAMP WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                            f"n={res.trades}/{res.oos_trades} {res.name}"
                        )
                        if q:
                            quals += 1
            if tested % 1000 == 0:
                Path(OUT / "public_progress.json").write_text(
                    json.dumps(
                        {
                            "tested": tested,
                            "quals": quals,
                            "best": None
                            if not best
                            else {
                                "wr": best.winrate,
                                "oos_wr": best.oos_winrate,
                                "tpd": best.trades_per_day,
                                "oos_tpd": best.oos_trades_per_day,
                                "name": best.name,
                            },
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        indent=2,
                    )
                )
                log(
                    f"pub tested={tested} quals={quals} "
                    f"best={(f'{best.winrate:.1%}/{best.oos_winrate:.1%} tpd{best.trades_per_day:.2f}' if best else 'n/a')}"
                )

        # AND combinations of public + SMC
        log("Public AND-combo pass...")
        for i, j in itertools.combinations(range(len(lib)), 2):
            ka, pa, La, Sa = lib[i]
            kb, pb, Lb, Sb = lib[j]
            if ka == kb:
                continue
            L, S = La & Lb, Sa & Sb
            both = L & S
            L, S = L & ~both, S & ~both
            if L.sum() + S.sum() < 90:
                continue
            for ex in ex_all[::5][:12]:
                tested += 1
                res = evaluate(f"pubAND|{ka}+{kb}", "5m", df, L, S, ex, {"kind": "combo_and", "a": ka, "b": kb})
                if not res:
                    continue
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
                if best is None or res.score > best.score:
                    if res.winrate >= 0.62 and res.net_pnl > 0:
                        best = res
                        save_champion(res)
                        write_bot(res)
                        if q:
                            quals += 1
                        log(
                            f"{'QUALIFIED ' if q else ''}AND CHAMP WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} {ka}+{kb}"
                        )
        log(f"Public pass done tested={tested} quals={quals}; looping forever...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
