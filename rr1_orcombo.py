#!/usr/bin/env python3
"""OR-combine mid/high-WR signals to push tpd>=1 while keeping WR high under 1:1 RR."""
from __future__ import annotations
import itertools, json, random, time
from pathlib import Path
from rr1_search import OUT, add_indicators, evaluate, gen_signals, load_frames, log, qualifies, save_champion, write_bot

RNG = random.Random(7)

KINDS_GRIDS = [
    ("rsi_level", lambda: itertools.product(
        [3, 5, 7, 9, 14], range(15, 41, 3), range(65, 96, 3), [3, 5, 7, 9],
        [("overlap", "ny"), ("london", "overlap", "ny"), ("ny",), ("overlap",), ("london",)],
    )),
    ("rsi_cross", lambda: itertools.product(
        [3, 5, 7, 9, 14], range(15, 41, 3), range(65, 96, 3), [0],
        [("overlap", "ny"), ("london", "overlap", "ny"), ("ny",), ("london",)],
    )),
    ("bb_rsi_fade", lambda: itertools.product(
        [(10, 2.0), (20, 2.0), (20, 2.5), (30, 2.0)], [3, 5, 7, 14], [20, 25, 30, 35], [65, 70, 75, 80],
        [("overlap", "ny"), ("london", "overlap", "ny"), ("ny",)],
    )),
    ("rsi_macd", lambda: itertools.product(
        [3, 5, 7, 14], range(20, 36, 5), range(65, 86, 5),
        [("overlap", "ny"), ("london", "overlap", "ny"), ("ny",)],
    )),
    ("stoch_cross", lambda: itertools.product(
        [5, 9, 14], range(15, 31, 5), range(70, 91, 5),
        [("overlap", "ny"), ("london", "overlap", "ny")],
    )),
    ("ema_pull_rsi", lambda: itertools.product(
        [9, 21], [21, 50], [5, 9, 14], range(25, 41, 5), range(60, 81, 5),
        [("overlap", "ny"), ("london", "overlap", "ny")],
    )),
]


def _params(kind: str, item):
    if kind == "rsi_level":
        rsi, os_, ob, band, sess = item
        return {"rsi": rsi, "os": os_, "ob": ob, "band": band, "sessions": sess}
    if kind == "rsi_cross":
        rsi, os_, ob, _, sess = item
        return {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    if kind == "bb_rsi_fade":
        (period, k), rsi, os_, ob, sess = item
        return {"period": period, "k": k, "rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    if kind == "rsi_macd":
        rsi, os_, ob, sess = item
        return {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    if kind == "stoch_cross":
        period, os_, ob, sess = item
        return {"period": period, "os": os_, "ob": ob, "sessions": sess}
    if kind == "ema_pull_rsi":
        fast, slow, rsi, os_, ob, sess = item
        return {"fast": fast, "slow": slow, "rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    return {}


def build_singles(frames, tfs, wr_floor: float, nsig_hi: int):
    # Fast pre-score exits (strict 1:1 hunt); fuller exit grid used on OR combos.
    pre_exits = [
        {"rr": 1.0, "max_hold": mh, "atr": "ATR_14", "allow_long": True, "allow_short": True}
        for mh in (2, 3, 4)
    ]
    exits = [
        {"rr": rr, "max_hold": mh, "atr": "ATR_14", "allow_long": True, "allow_short": True}
        for rr in (1.0, 1.2, 1.5)
        for mh in (2, 3, 4, 6)
    ]
    singles = []
    for tf in tfs:
        if tf not in frames:
            continue
        # Skip 1m full-grid pre-score (too slow under memory pressure); sample later if needed.
        if tf == "1m" and wr_floor >= 0.58:
            continue
        df = frames[tf]
        lib = []
        for kind, grid_fn in KINDS_GRIDS:
            for item in grid_fn():
                p = _params(kind, item)
                if not p:
                    continue
                try:
                    L, S = gen_signals(df, kind, p)
                except Exception:
                    continue
                nsig = int(L.sum() + S.sum())
                if 30 <= nsig <= nsig_hi:
                    lib.append((kind, p, L, S, nsig))
        # Cap pre-score budget; prefer mid-frequency signals
        lib.sort(key=lambda x: abs(x[4] - 180))
        if len(lib) > 400:
            lib = lib[:400]
        log(f"library[{tf}] size={len(lib)} wr_floor={wr_floor:.2f}")
        scored = 0
        for kind, p, L, S, nsig in lib:
            best = None
            for ex in pre_exits:
                res = evaluate(f"s|{kind}|{tf}", tf, df, L, S, ex, {"kind": kind, **p})
                if not res:
                    continue
                if best is None or min(res.winrate, res.oos_winrate) > min(best.winrate, best.oos_winrate):
                    best = res
            scored += 1
            if scored % 100 == 0:
                log(f"pre-score[{tf}] {scored}/{len(lib)} kept={len([s for s in singles if s[5]==tf])}")
            if best and min(best.winrate, best.oos_winrate) >= wr_floor and best.trades >= 20:
                singles.append((best, kind, p, L, S, tf, df))
    singles = sorted(singles, key=lambda x: min(x[0].winrate, x[0].oos_winrate), reverse=True)[:60]
    if singles:
        t = singles[0][0]
        log(f"strong singles={len(singles)} top={t.winrate:.1%}/{t.oos_winrate:.1%} tpd={t.trades_per_day:.2f}")
    else:
        log("no singles")
    return singles, exits


def main():
    OUT.mkdir(exist_ok=True)
    log("=== RR1 OR-COMBO START ===")
    frames = {k: add_indicators(v) for k, v in load_frames().items()}

    tested = 0
    quals = 0
    best_key = (-1.0, -1.0, -1)
    wr_floors = [0.62, 0.58, 0.55, 0.52]
    nsig_his = [800, 1500, 3000]
    floor_i = 0
    nsig_i = 0

    while True:
        wr_floor = wr_floors[min(floor_i, len(wr_floors) - 1)]
        nsig_hi = nsig_his[min(nsig_i, len(nsig_his) - 1)]
        tfs = ["5m", "1m", "15m"] if floor_i >= 1 else ["5m", "1m"]
        singles, exits = build_singles(frames, tfs, wr_floor, nsig_hi)

        if len(singles) < 2:
            log(f"pool too small ({len(singles)}); loosening floors")
            if floor_i < len(wr_floors) - 1:
                floor_i += 1
            elif nsig_i < len(nsig_his) - 1:
                nsig_i += 1
            else:
                floor_i = 0
                nsig_i = 0
                time.sleep(30)
            continue

        # OR pairs within same TF
        by_tf = {}
        for row in singles:
            by_tf.setdefault(row[5], []).append(row)

        for tf, pool in by_tf.items():
            df = pool[0][6]
            for (a, ka, pa, La, Sa, _, _), (b, kb, pb, Lb, Sb, _, _) in itertools.combinations(pool[:30], 2):
                if ka == kb and pa == pb:
                    continue
                L = La | Lb
                S = Sa | Sb
                both = L & S
                L = L & ~both
                S = S & ~both
                if int(L.sum() + S.sum()) < 50:
                    continue
                for ex in exits:
                    tested += 1
                    res = evaluate(
                        f"or|{ka}+{kb}|{tf}",
                        tf,
                        df,
                        L,
                        S,
                        ex,
                        {"kind": "combo_or", "a": ka, "b": kb, "pa": pa, "pb": pb},
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
                            f"n={res.trades}/{res.oos_trades}"
                        )
                    elif key > best_key and res.winrate >= 0.70 and res.oos_winrate >= 0.70:
                        best_key = key
                        save_champion(res)
                        write_bot(res)
                        log(
                            f"NEAR WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                            f"n={res.trades}/{res.oos_trades}"
                        )
                    if tested % 500 == 0:
                        msg = (
                            f"or tested={tested} quals={quals} best_minWR={best_key[0]:.1%}"
                            if best_key[0] >= 0
                            else f"or tested={tested} quals={quals}"
                        )
                        log(msg)
                        Path(OUT / "orcombo_progress.json").write_text(
                            json.dumps(
                                {
                                    "tested": tested,
                                    "quals": quals,
                                    "pool": len(singles),
                                    "wr_floor": wr_floor,
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                },
                                indent=2,
                            )
                        )

            # OR three random from top of this TF
            top = pool[: max(3, min(25, len(pool)))]
            if len(top) >= 3:
                for _ in range(250):
                    picks = RNG.sample(top, 3)
                    L = picks[0][3] | picks[1][3] | picks[2][3]
                    S = picks[0][4] | picks[1][4] | picks[2][4]
                    both = L & S
                    L = L & ~both
                    S = S & ~both
                    if int(L.sum() + S.sum()) < 55:
                        continue
                    for ex in exits:
                        tested += 1
                        res = evaluate("or3|" + tf, tf, df, L, S, ex, {"kind": "combo_or3"})
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
                            quals += 1
                            save_champion(res)
                            write_bot(res)
                            log(
                                f"QUALIFIED or3 #{quals} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f}"
                            )
                        elif (
                            min(res.winrate, res.oos_winrate) > best_key[0]
                            and res.winrate >= 0.70
                            and res.oos_winrate >= 0.70
                        ):
                            best_key = (
                                min(res.winrate, res.oos_winrate),
                                min(res.trades_per_day, res.oos_trades_per_day),
                                res.trades + res.oos_trades,
                            )
                            save_champion(res)
                            write_bot(res)

        log(f"or pass done tested={tested} quals={quals}; rebuild+loop")
        Path(OUT / "orcombo_progress.json").write_text(
            json.dumps(
                {
                    "tested": tested,
                    "quals": quals,
                    "pool": len(singles),
                    "wr_floor": wr_floor,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
        RNG.seed(int(time.time()) % 100000)
        # slowly loosen if still no near-misses after a full pass
        if best_key[0] < 0.70 and floor_i < len(wr_floors) - 1:
            floor_i += 1


if __name__ == "__main__":
    main()
