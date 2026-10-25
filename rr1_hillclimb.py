#!/usr/bin/env python3
"""Hillclimb around best 1:1 near-miss until full qualifier or killed."""
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
)

RNG = random.Random(42)


def main():
    OUT.mkdir(exist_ok=True)
    log("=== RR1 HILLCLIMB START ===")
    frames = {k: add_indicators(v) for k, v in load_frames().items()}
    if (OUT / "CHAMPION.json").exists():
        seed = json.loads((OUT / "CHAMPION.json").read_text())
        log(f"seed WR={seed['winrate']:.1%}/{seed['oos_winrate']:.1%} tpd={seed['trades_per_day']:.2f}")
    else:
        seed = {
            "winrate": 0.0,
            "oos_winrate": 0.0,
            "trades_per_day": 0.0,
            "oos_trades_per_day": 0.0,
            "trades": 0,
            "oos_trades": 0,
            "params": {
                "kind": "rsi_level",
                "rsi": 7,
                "os": 20,
                "ob": 80,
                "band": 5,
                "sessions": ["overlap", "ny"],
                "exit": {"rr": 1.0, "max_hold": 6, "atr": "ATR_14", "allow_long": True, "allow_short": True},
            },
        }
        log("no champion yet — seeding from default rsi_level")

    tested = 0
    quals = 0
    best_key = (
        min(seed["winrate"], seed["oos_winrate"]),
        min(seed["trades_per_day"], seed["oos_trades_per_day"]),
        seed["trades"] + seed["oos_trades"],
    )

    sessions_opts = [
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
        ("ny",),
        ("overlap",),
        ("london", "overlap"),
        ("asian", "london", "overlap", "ny"),
        None,
    ]
    hours_opts = [
        None,
        tuple(range(13, 21)),
        tuple(range(12, 22)),
        tuple(range(8, 21)),
        tuple(range(14, 20)),
        tuple(range(13, 22)),
    ]

    while True:
        # reload seed if another process found better
        try:
            cur = json.loads((OUT / "CHAMPION.json").read_text())
            if min(cur["winrate"], cur["oos_winrate"]) >= min(seed["winrate"], seed["oos_winrate"]):
                seed = cur
        except Exception:
            pass

        base_kind = seed["params"].get("kind", "rsi_level")
        if base_kind == "combo_and":
            base_kind = "rsi_level"

        for tf in ("5m", "15m"):
            df = frames[tf]
            for kind in ("rsi_level", "rsi_cross"):
                for rsi, os_, ob, band, sess, hours in itertools.product(
                    [3, 5, 7, 9],
                    range(12, 36, 2),
                    range(70, 96, 2),
                    [3, 5, 7, 9, 12],
                    sessions_opts,
                    hours_opts,
                ):
                    params = {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
                    if kind == "rsi_level":
                        params["band"] = band
                    if hours:
                        params["hours"] = hours
                    if RNG.random() < 0.2:
                        params["vol_ok"] = True
                    try:
                        L, S = gen_signals(df, kind, params)
                    except Exception:
                        continue
                    if L.sum() + S.sum() < 45:
                        continue
                    for rr, mh, atr, sides in itertools.product(
                        [0.8, 1.0, 1.2, 1.5],
                        [2, 3, 4, 6],
                        ["ATR_14", "ATR_7"],
                        [
                            dict(allow_long=True, allow_short=True),
                            dict(allow_long=True, allow_short=False),
                            dict(allow_long=False, allow_short=True),
                        ],
                    ):
                        # skip most to keep speed; denser near seed
                        near = (
                            abs(rsi - seed["params"].get("rsi", 7)) <= 2
                            and abs(os_ - seed["params"].get("os", 20)) <= 6
                            and abs(ob - seed["params"].get("ob", 90)) <= 8
                        )
                        if not near and RNG.random() > 0.08:
                            continue
                        ex = {"rr": rr, "max_hold": mh, "atr": atr, **sides}
                        tested += 1
                        res = evaluate(
                            f"hill|{kind}|{tf}",
                            tf,
                            df,
                            L,
                            S,
                            ex,
                            {"kind": kind, **params},
                        )
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
                        key = (
                            min(res.winrate, res.oos_winrate),
                            min(res.trades_per_day, res.oos_trades_per_day),
                            res.trades + res.oos_trades,
                        )
                        if q:
                            quals += 1
                            save_champion(res)
                            write_bot(res)
                            log(
                                f"QUALIFIED #{quals} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                                f"n={res.trades}/{res.oos_trades} PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f}"
                            )
                            seed = json.loads((OUT / "CHAMPION.json").read_text())
                            best_key = key
                        elif key > best_key and res.winrate >= 0.78 and res.oos_winrate >= 0.78:
                            best_key = key
                            save_champion(res)
                            write_bot(res)
                            seed = json.loads((OUT / "CHAMPION.json").read_text())
                            log(
                                f"NEAR WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                                f"n={res.trades}/{res.oos_trades}"
                            )
                        if tested % 2000 == 0:
                            log(f"hill tested={tested} quals={quals} best_minWR={best_key[0]:.1%}")
                            Path(OUT / "hillclimb_progress.json").write_text(
                                json.dumps(
                                    {
                                        "tested": tested,
                                        "quals": quals,
                                        "best_key": best_key,
                                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    },
                                    indent=2,
                                )
                            )
        log(f"hill grid pass done tested={tested} quals={quals}; looping forever...")


if __name__ == "__main__":
    main()
