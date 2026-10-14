#!/usr/bin/env python3
"""Never-stop ICT/SMC/FVG/Fib hunter on 2y local 5m, 1:1 RR, ≥75% WR, ≥1 tpd."""
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

RNG = random.Random(99)
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


def exits():
    out = []
    # Strict RR ≥ 1.0 only (1:1 hunt); sub-1.0 ATR multiples banned.
    for rr, mh, atr, side in itertools.product(
        [1.0, 1.2, 1.5],
        [2, 3, 4, 6, 8, 12, 18, 24],
        ["ATR_14", "ATR_7"],
        [
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        ],
    ):
        out.append({"rr": rr, "max_hold": mh, "atr": atr, **side})
    return out


def space():
    sp = []
    for sessions in SESS:
        sp.append(("fvg_reclaim", {"sessions": sessions}))
        sp.append(("ob_retest", {"sessions": sessions}))
        sp.append(("ob_fvg", {"sessions": sessions}))
        for rsi, os_, ob in itertools.product([3, 7, 14], [20, 25, 30, 35], [65, 70, 75, 80]):
            sp.append(("fvg_rsi", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))
    for sw, wick, sessions in itertools.product([5, 10, 20, 40], [0.4, 0.5, 0.6], SESS):
        sp.append(("liq_sweep", {"swing": sw, "wick": wick, "sessions": sessions}))
        sp.append(("liq_sweep_fvg", {"swing": sw, "sessions": sessions}))
        sp.append(("smc_confluence", {"swing": sw, "sessions": sessions}))
        sp.append(("mss_entry", {"swing": sw, "sessions": sessions}))
        sp.append(("sweep_ob", {"swing": sw, "sessions": sessions}))
        sp.append(("discount_ob", {"swing": sw, "sessions": sessions}))
        sp.append(("eq_fade", {"swing": sw, "sessions": sessions}))
    for sw, fib, wick, sessions in itertools.product(
        [10, 20, 40], ["382", "500", "618", "786"], [0.35, 0.45, 0.55], SESS[:7]
    ):
        sp.append(("fib_bounce", {"swing": sw, "fib": fib, "wick": wick, "sessions": sessions}))
        sp.append(("fib_discount", {"swing": sw, "fib": fib, "sessions": sessions}))
    for sw, ema, bos_look, sessions in itertools.product([20, 40], [8, 12, 20, 50], [6, 12, 20], SESS[:6]):
        sp.append(("bos_pullback", {"swing": sw, "ema": ema, "bos_look": bos_look, "sessions": sessions}))
    # AND combos of SMC with RSI/BB fades via filter flags
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


def main():
    OUT.mkdir(exist_ok=True)
    log("=== RR1 SMC/ICT/FVG/FIB SEARCH (≥75% WR, 1:1, ≥1 tpd, 2y) ===")
    frames = {k: add_indicators(v) for k, v in load_frames().items()}
    assert "5m" in frames and (frames["5m"].index[-1] - frames["5m"].index[0]).days >= 200
    df = frames["5m"]
    log(f"5m bars={len(df)} {df.index[0]} → {df.index[-1]}")
    tested = 0
    quals = 0
    best = None
    ex_all = exits()
    while True:
        for kind, params in space():
            try:
                L, S = gen_signals(df, kind, params)
            except Exception:
                continue
            if L.sum() + S.sum() < 80:
                continue
            # probe subset of exits then deepen
            probe = ex_all[:: max(1, len(ex_all) // 20)][:20]
            local = None
            for ex in probe:
                tested += 1
                res = evaluate(f"smc|{kind}|5m", "5m", df, L, S, ex, {"kind": kind, **params})
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
                if local is None or res.score > local.score:
                    local = res
                if best is None or res.score > best.score:
                    if res.winrate >= 0.70 and res.trades_per_day >= 0.8:
                        best = res
                        save_champion(res)
                        write_bot(res)
                        tag = "QUALIFIED " if q else ""
                        if q:
                            quals += 1
                        log(
                            f"{tag}NEW CHAMP WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                            f"n={res.trades}/{res.oos_trades} PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f} "
                            f"{res.name}"
                        )
            if local and local.winrate >= 0.72 and local.trades_per_day >= 0.9:
                for ex in ex_all:
                    tested += 1
                    res = evaluate(f"smc|{kind}|deep", "5m", df, L, S, ex, {"kind": kind, **params})
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
                    if q and year_stable(df, L, S, ex):
                        quals += 1
                        save_champion(res)
                        write_bot(res)
                        best = res
                        log(
                            f"QUALIFIED deep WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}"
                        )
                    elif best is None or res.score > best.score:
                        if res.winrate >= 0.70:
                            best = res
                            save_champion(res)
                            write_bot(res)
            if tested % 1000 == 0:
                Path(OUT / "smc_progress.json").write_text(
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
                                "n": best.trades,
                                "oos_n": best.oos_trades,
                                "name": best.name,
                            },
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "target": "WR>=75% tpd>=1 RR=1:1 2y ICT/SMC",
                        },
                        indent=2,
                    )
                )
                log(
                    f"smc tested={tested} quals={quals} "
                    f"best={(f'{best.winrate:.1%}/{best.oos_winrate:.1%} tpd{best.trades_per_day:.2f}' if best else 'n/a')}"
                )
        log(f"SMC space pass done tested={tested} quals={quals}; looping 24h...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
