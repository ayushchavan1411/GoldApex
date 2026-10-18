#!/usr/bin/env python3
"""Bridge hunter: lift high-frequency 1:1 bases toward ≥75% WR via causal masks / light ORs.

Phase 1 — OR high-WR RSI+MACD seeds with partners (frequency up).
Phase 2 — AND soft regime/session/EMA masks onto high-tpd bases (winrate up).
Strict RR ≥ 1.0 only.
"""
from __future__ import annotations
import itertools, json, random, time
from pathlib import Path
import numpy as np
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

RNG = random.Random(77)


def _maybe_save(best, res, L, S, df, ex, quals, tag):
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
    score = (min(res.winrate, res.oos_winrate), min(res.trades_per_day, res.oos_trades_per_day), res.trades + res.oos_trades)
    bscore = (
        (min(best.winrate, best.oos_winrate), min(best.trades_per_day, best.oos_trades_per_day), best.trades + best.oos_trades)
        if best
        else (-1.0, -1.0, -1)
    )
    useful = res.winrate >= 0.65 and res.oos_winrate >= 0.65 and res.trades_per_day >= 0.5
    near = res.winrate >= 0.70 and res.oos_winrate >= 0.70
    if q or useful or near:
        if best is None or score > bscore:
            best = res
            save_champion(res)
            write_bot(res)
            if q:
                quals += 1
            log(
                f"{'QUAL ' if q else ''}{tag} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}"
            )
    return best, quals, q


def main():
    OUT.mkdir(exist_ok=True)
    log("=== FREQBOOST bridge (RR≥1.0, high-tpd masks + high-WR ORs) ===")
    df = add_indicators(load_frames()["5m"])

    seeds = []
    for rsi, os_, ob, sess, hours, side in itertools.product(
        [7, 14],
        [40, 45, 50],
        [55, 60, 65],
        [("asian",), ("asian", "london"), None, ("london", "overlap")],
        [None, tuple(range(7, 22)), tuple(range(0, 12)), tuple(range(8, 17))],
        [
            dict(allow_long=True, allow_short=False),
            dict(allow_long=True, allow_short=True),
            dict(allow_long=False, allow_short=True),
        ],
    ):
        p = {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
        if hours:
            p["hours"] = hours
        seeds.append(("rsi_macd", p, side))

    partners = []
    for kind, p in [
        ("fvg_reclaim", {"sessions": ("asian", "london", "overlap")}),
        ("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": ("asian", "london", "overlap", "ny")}),
        ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": None}),
        ("heikin_ema", {"ema": 50, "sessions": None}),
        ("ichimoku_tk", {"sessions": ("london", "overlap", "ny")}),
        ("keltner_fade", {"period": 20, "mult": 2.0, "sessions": None}),
        ("zscore_fade", {"period": 20, "th": 2.0, "sessions": None}),
        ("orb_fade", {"orb": "LONDON", "sessions": ("london", "overlap")}),
        ("rsi_level", {"rsi": 7, "os": 25, "ob": 75, "band": 5, "sessions": None}),
        ("smc_confluence", {"swing": 20, "sessions": ("overlap", "ny")}),
    ]:
        partners.append((kind, p))

    # High-frequency bases to lift with masks (stay on 5m to limit RAM)
    freq_bases = []
    for kind, grid in [
        ("rsi_cross", itertools.product([3, 5, 7, 14], [25, 30, 35], [65, 70, 75], [None, ("london", "overlap", "ny")])),
        ("bb_fade", itertools.product([(10, 2.0), (20, 2.0), (20, 2.5)], [None, ("overlap", "ny")])),
        ("stoch_cross", itertools.product([5, 9, 14], [20, 25], [75, 80], [None, ("london", "overlap", "ny")])),
        ("macd_cross", itertools.product([None, ("london", "overlap", "ny"), ("overlap", "ny")])),
        ("ema_cross", itertools.product([8, 9, 12], [21, 26, 50], [None, ("london", "overlap", "ny")])),
    ]:
        for item in grid:
            if kind == "rsi_cross":
                rsi, os_, ob, sess = item
                p = {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
            elif kind == "bb_fade":
                (period, k), sess = item
                p = {"period": period, "k": k, "sessions": sess}
            elif kind == "stoch_cross":
                period, os_, ob, sess = item
                p = {"period": period, "os": os_, "ob": ob, "sessions": sess}
            elif kind == "macd_cross":
                (sess,) = item if isinstance(item, tuple) else (item,)
                p = {"sessions": sess}
            else:
                fast, slow, sess = item
                p = {"fast": fast, "slow": slow, "sessions": sess}
            try:
                L, S = gen_signals(df, kind, p)
            except Exception:
                continue
            n = int(L.sum() + S.sum())
            if 200 <= n <= 8000:
                freq_bases.append((kind, p, L, S, n))
    RNG.shuffle(freq_bases)
    freq_bases = freq_bases[:80]

    # Causal boolean masks (already shift-safe columns from add_indicators)
    masks = []
    ema_fast = df["EMA_21"]
    ema_slow = df["EMA_50"]
    adx = df["ADX"]
    atr = df["ATR_14"]
    atr_ma = atr.rolling(50).mean()
    hour = df.index.hour
    masks.append(("ema_bull", (ema_fast > ema_slow).to_numpy(), (ema_fast < ema_slow).to_numpy()))
    masks.append(("adx20", (adx > 20).to_numpy(), (adx > 20).to_numpy()))
    masks.append(("adx25", (adx > 25).to_numpy(), (adx > 25).to_numpy()))
    masks.append(("atr_exp", (atr > atr_ma).fillna(False).to_numpy(), (atr > atr_ma).fillna(False).to_numpy()))
    masks.append(("lon_ny", np.isin(hour, list(range(7, 21))), np.isin(hour, list(range(7, 21)))))
    masks.append(("overlap", np.isin(hour, list(range(12, 17))), np.isin(hour, list(range(12, 17)))))
    rsi7 = df["RSI_7"]
    masks.append(("rsi_mid", ((rsi7 > 40) & (rsi7 < 60)).to_numpy(), ((rsi7 > 40) & (rsi7 < 60)).to_numpy()))
    masks.append(("rsi_ext", ((rsi7 < 35) | (rsi7 > 65)).to_numpy(), ((rsi7 < 35) | (rsi7 > 65)).to_numpy()))

    pc = []
    for kind, p in partners:
        try:
            L, S = gen_signals(df, kind, p)
            if L.sum() + S.sum() >= 40:
                pc.append((kind, p, L, S))
        except Exception:
            pass
    log(f"partners={len(pc)} seeds={len(seeds)} freq_bases={len(freq_bases)} masks={len(masks)}")

    tested = 0
    quals = 0
    best = None
    # User: drop ATR RR — 1% price risk/reward only
    exits_1_1 = fee_pct_exits(PCT_RR, holds=(12, 24, 36, 48), sides=False)

    while True:
        # Phase 1: OR high-WR family
        for kind, p, side in seeds:
            try:
                L0, S0 = gen_signals(df, kind, p)
            except Exception:
                continue
            for ex0 in fee_pct_exits(PCT_RR, holds=(24, 48), sides=False):
                ex0 = {**ex0, **side}
                tested += 1
                res = evaluate(f"fb|{kind}", "5m", df, L0, S0, ex0, {"kind": kind, **p})
                if not res:
                    continue
                best, quals, _ = _maybe_save(best, res, L0, S0, df, ex0, quals, "NEAR")
            for pk, pp, Lp, Sp in pc:
                L = L0 | Lp
                S = S0 | Sp
                both = L & S
                L, S = L & ~both, S & ~both
                if int(L.sum() + S.sum()) < 80:
                    continue
                for ex0 in fee_pct_exits(PCT_RR, holds=(24, 48), sides=False):
                    ex0 = {**ex0, **side}
                    tested += 1
                    res = evaluate(f"fbOR|{kind}+{pk}", "5m", df, L, S, ex0, {"kind": "combo_or", "a": kind, "b": pk})
                    if not res:
                        continue
                    best, quals, _ = _maybe_save(best, res, L, S, df, ex0, quals, f"OR {kind}+{pk}")

        # Phase 2: AND masks onto high-frequency bases (seek WR↑ while tpd≥1)
        for kind, p, L0, S0, n in freq_bases:
            for mname, mL, mS in masks:
                L = L0 & mL
                S = S0 & mS
                both = L & S
                L, S = L & ~both, S & ~both
                n2 = int(L.sum() + S.sum())
                if n2 < 120:
                    continue
                # two-mask stacks (sampled)
                extra = RNG.sample(masks, k=min(3, len(masks)))
                combos = [(mname, L, S)] + [
                    (f"{mname}+{en}", L & eL, S & eS) for en, eL, eS in extra if en != mname
                ]
                for cname, Lc, Sc in combos:
                    both = Lc & Sc
                    Lc, Sc = Lc & ~both, Sc & ~both
                    if int(Lc.sum() + Sc.sum()) < 100:
                        continue
                    for ex in exits_1_1:
                        tested += 1
                        res = evaluate(
                            f"fbAND|{kind}|{cname}",
                            "5m",
                            df,
                            Lc,
                            Sc,
                            ex,
                            {"kind": "freq_mask", "base": kind, "mask": cname, **p},
                        )
                        if not res:
                            continue
                        # track anything approaching the dual gate
                        bridge = (
                            min(res.winrate, res.oos_winrate) >= 0.60
                            and min(res.trades_per_day, res.oos_trades_per_day) >= 0.85
                        )
                        if bridge or (res.winrate >= 0.70 and res.oos_winrate >= 0.70):
                            best, quals, _ = _maybe_save(best, res, Lc, Sc, df, ex, quals, f"AND {kind}|{cname}")
                        elif qualifies(
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
                        ):
                            if year_stable(df, Lc, Sc, ex):
                                quals += 1
                                save_champion(res)
                                write_bot(res)
                                best = res
                                log(
                                    f"QUAL AND WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                    f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f}"
                                )
                    if tested % 500 == 0:
                        btxt = (
                            f"{best.winrate:.1%}/{best.oos_winrate:.1%} tpd{best.trades_per_day:.2f}"
                            if best
                            else "n/a"
                        )
                        log(f"freqboost tested={tested} quals={quals} best={btxt}")
                        Path(OUT / "freqboost_progress.json").write_text(
                            json.dumps(
                                {"tested": tested, "quals": quals, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                                indent=2,
                            )
                        )

        log(f"freqboost pass done tested={tested}; reshuffle+loop")
        RNG.shuffle(freq_bases)
        Path(OUT / "freqboost_progress.json").write_text(
            json.dumps({"tested": tested, "quals": quals, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
        )


if __name__ == "__main__":
    main()
