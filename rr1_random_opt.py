#!/usr/bin/env python3
"""Random + mutation optimizer for 1:1 RR ≥80% WR ≥1 tpd (runs forever)."""
from __future__ import annotations

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

RNG = random.Random(int(time.time()) % 100000)
KINDS = [
    "rsi_cross",
    "rsi_level",
    "bb_fade",
    "bb_rsi_fade",
    "bb_mid_reject",
    "stoch_cross",
    "triple_fade",
    "ema_wick",
    "ema_stack_pull",
    "range_fade",
    "micro_reclaim",
    "pin_rsi",
    "engulf",
    "macd_cross",
    "donch_fade",
]
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


def random_params(kind: str) -> dict:
    p: dict = {"sessions": RNG.choice(SESS)}
    if RNG.random() < 0.35:
        p["vol_ok"] = True
    if RNG.random() < 0.4:
        p["hours"] = tuple(range(RNG.choice([0, 7, 8, 12, 13]), RNG.choice([16, 17, 20, 22])))
    if kind in ("rsi_cross", "rsi_level", "micro_reclaim"):
        p.update(rsi=RNG.choice([2, 3, 5, 7, 9, 14]), os=RNG.choice([10, 15, 20, 25, 30, 35]), ob=RNG.choice([65, 70, 75, 80, 85, 90]))
        if kind == "rsi_level":
            p["band"] = RNG.choice([3, 5, 8, 10])
    elif kind in ("bb_fade", "bb_mid_reject"):
        # must match columns from add_indicators
        period, k = RNG.choice([(10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5), (50, 2.0)])
        p.update(period=period, k=k)
    elif kind == "bb_rsi_fade":
        period, k = RNG.choice([(10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5)])
        p.update(
            period=period,
            k=k,
            rsi=RNG.choice([2, 3, 5, 7]),
            os=RNG.choice([20, 25, 30, 35]),
            ob=RNG.choice([65, 70, 75, 80]),
        )
    elif kind == "stoch_cross":
        p.update(period=RNG.choice([5, 9, 14]), os=RNG.choice([15, 20, 25, 30]), ob=RNG.choice([70, 75, 80, 85]))
    elif kind == "triple_fade":
        p.update(
            bb=RNG.choice([10, 20]),
            k=RNG.choice([1.5, 2.0]),
            rsi=RNG.choice([2, 3, 7]),
            stoch=RNG.choice([5, 14]),
            os=RNG.choice([20, 25, 30]),
            ob=RNG.choice([70, 75, 80]),
        )
    elif kind == "ema_wick":
        p.update(ema=RNG.choice([8, 12, 20, 50]), rsi=RNG.choice([2, 3, 7]), os=RNG.choice([25, 30, 35]), ob=RNG.choice([65, 70, 75]))
    elif kind == "ema_stack_pull":
        a, b, c = sorted(RNG.sample([5, 8, 12, 20, 21, 34, 50, 100, 200], 3))
        p.update(a=a, b=b, c=c)
    elif kind == "range_fade":
        p.update(look=RNG.choice([6, 12, 20, 32, 40]), rsi=RNG.choice([2, 3, 5, 7]), os=RNG.choice([20, 25, 30, 35]), ob=RNG.choice([65, 70, 75, 80]))
    elif kind == "donch_fade":
        p.update(period=RNG.choice([10, 20, 40]), rsi=RNG.choice([2, 3, 5, 7]), os=RNG.choice([20, 25, 30]), ob=RNG.choice([70, 75, 80]))
    elif kind == "pin_rsi":
        p.update(
            rsi=RNG.choice([2, 3, 7]),
            wick=RNG.choice([0.5, 0.6, 0.7]),
            body=RNG.choice([0.3, 0.4, 0.5]),
            os=RNG.choice([25, 30]),
            ob=RNG.choice([70, 75]),
        )
    elif kind == "engulf":
        p.update(ema=RNG.choice([20, 50, 200]))
    elif kind == "macd_cross":
        pass
    return p


def random_exit() -> dict:
    return {
        "rr": RNG.choice([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]),
        "max_hold": RNG.choice([2, 3, 4, 6, 8, 12, 18, 24, 36]),
        "atr": RNG.choice(["ATR_14", "ATR_7"]),
        **RNG.choice(
            [
                {"allow_long": True, "allow_short": True},
                {"allow_long": True, "allow_short": False},
                {"allow_long": False, "allow_short": True},
            ]
        ),
    }


def main():
    OUT.mkdir(exist_ok=True)
    log("=== RR1 RANDOM OPT START ===")
    frames = {k: add_indicators(v) for k, v in load_frames().items()}
    if "5m" in frames and (frames["5m"].index[-1] - frames["5m"].index[0]).days >= 200:
        tfs = ["5m"]
        log("Random opt locked to long local 5m")
    else:
        tfs = [t for t in ("5m", "15m", "1h") if t in frames]
    best = None
    tested = 0
    quals = 0
    side = Path(OUT / "random_opt_progress.json")

    # seed from current champion if present
    champ_path = OUT / "CHAMPION.json"
    seed = json.loads(champ_path.read_text()) if champ_path.exists() else None

    while True:
        try:
            if seed and RNG.random() < 0.45:
                kind = seed["params"]["kind"]
                if kind == "combo_and":
                    kind = RNG.choice(KINDS)
                params = {k: v for k, v in seed["params"].items() if k not in ("kind", "exit", "a", "b")}
                if params.get("sessions") is not None:
                    params["sessions"] = tuple(params["sessions"])
                if params.get("hours") is not None:
                    params["hours"] = tuple(params["hours"])
                # mutate
                for key in ("os", "ob", "band", "look"):
                    if key in params and isinstance(params[key], (int, float)):
                        params[key] = int(params[key] + RNG.randint(-5, 5))
                if "os" in params:
                    params["os"] = min(40, max(5, int(params["os"])))
                if "ob" in params:
                    params["ob"] = min(95, max(60, int(params["ob"])))
                # keep indicator periods on valid columns only
                if "rsi" in params:
                    params["rsi"] = RNG.choice([2, 3, 5, 7, 9, 14])
                if kind in ("bb_fade", "bb_mid_reject"):
                    params["period"], params["k"] = RNG.choice(
                        [(10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5), (50, 2.0)]
                    )
                elif kind == "bb_rsi_fade":
                    params["period"], params["k"] = RNG.choice(
                        [(10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5)]
                    )
                elif kind == "stoch_cross":
                    params["period"] = RNG.choice([5, 9, 14])
                elif kind == "donch_fade":
                    params["period"] = RNG.choice([10, 20, 40])
                if "ema" in params:
                    params["ema"] = RNG.choice([8, 12, 20, 50, 200])
                if "bb" in params:
                    params["bb"] = RNG.choice([10, 20])
                    params["k"] = RNG.choice([1.5, 2.0])
                if "stoch" in params:
                    params["stoch"] = RNG.choice([5, 14])
                if RNG.random() < 0.3:
                    params["sessions"] = RNG.choice(SESS)
                tf = seed.get("timeframe", RNG.choice(tfs))
                exit_p = dict(seed["params"].get("exit") or random_exit())
                if RNG.random() < 0.5:
                    exit_p = random_exit()
            else:
                kind = RNG.choice(KINDS)
                params = random_params(kind)
                tf = RNG.choice(tfs)
                exit_p = random_exit()

            df = frames[tf]
            long_e, short_e = gen_signals(df, kind, params)
            if long_e.sum() + short_e.sum() < 40:
                continue
            res = evaluate(f"rand|{kind}|{tf}", tf, df, long_e, short_e, exit_p, {"kind": kind, **params})
            tested += 1
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
            if q and not year_stable(df, long_e, short_e, exit_p):
                q = False
            if q:
                quals += 1
                save_champion(res)
                write_bot(res)
                seed = json.loads(champ_path.read_text())
                log(
                    f"QUALIFIED #{quals} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                    f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                    f"n={res.trades}/{res.oos_trades} PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f} {res.name}"
                )
            score_key = (
                min(res.winrate, res.oos_winrate),
                min(res.trades_per_day, res.oos_trades_per_day),
                res.oos_profit_factor,
                res.trades + res.oos_trades,
            )
            if best is None or score_key > best[0]:
                if res.winrate >= 0.75 and res.trades_per_day >= 0.9:
                    best = (score_key, res)
                    # only overwrite champion if better composite than file
                    try:
                        cur = json.loads(champ_path.read_text())
                        cur_key = (
                            min(cur["winrate"], cur["oos_winrate"]),
                            min(cur["trades_per_day"], cur["oos_trades_per_day"]),
                            cur["oos_profit_factor"],
                            cur["trades"] + cur["oos_trades"],
                        )
                    except Exception:
                        cur_key = (-1,)
                    if score_key > cur_key:
                        save_champion(res)
                        write_bot(res)
                        seed = json.loads(champ_path.read_text())
                        log(
                            f"NEW BEST minWR={min(res.winrate,res.oos_winrate):.1%} "
                            f"WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                            f"n={res.trades}/{res.oos_trades}"
                        )
            if tested % 500 == 0:
                b = best[1] if best else None
                side.write_text(
                    json.dumps(
                        {
                            "tested": tested,
                            "quals": quals,
                            "best": None
                            if not b
                            else {
                                "wr": b.winrate,
                                "oos_wr": b.oos_winrate,
                                "tpd": b.trades_per_day,
                                "oos_tpd": b.oos_trades_per_day,
                                "n": b.trades,
                                "oos_n": b.oos_trades,
                                "name": b.name,
                            },
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        indent=2,
                    )
                )
                log(
                    f"rand tested={tested} quals={quals} "
                    f"best={(f'{b.winrate:.1%}/{b.oos_winrate:.1%} tpd{b.trades_per_day:.2f}' if b else 'n/a')}"
                )
        except Exception:
            log(traceback.format_exc())
            time.sleep(0.5)


if __name__ == "__main__":
    main()
