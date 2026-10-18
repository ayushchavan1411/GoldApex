#!/usr/bin/env python3
"""5× portfolio hunt — any SL%/TP%, after charges, fixed 0.02 lot on $500.

Goal: final_equity ≥ $2500 over ~2y full window (5×), IS+OOS both profitable,
maximize min(WR_IS, WR_OOS). ATR RR allowed only via percent-of-price exits.
"""
from __future__ import annotations

import itertools
import json
import random
import time

import numpy as np

from rr1_search import (
    COMMISSION_PER_LOT_RT,
    FIXED_LOT,
    OUT,
    SPREAD_PRICE,
    START_CAP,
    _exit_sim_kwargs,
    add_indicators,
    gen_signals,
    load_frames,
    log,
    simulate,
    year_stable,
)

RNG = random.Random(42)
LEADER = OUT / "FIVE_X_LEADERBOARD.jsonl"
PROGRESS = OUT / "five_x_progress.json"
SUMMARY = OUT / "FIVE_X_SUMMARY.json"
CHAMP = OUT / "FIVE_X_CHAMPION.json"
TARGET_EQ = START_CAP * 5.0  # $2500


def fee_asym_exits(
    pairs: list[tuple[float, float]] | None = None,
    holds: tuple[int, ...] = (24, 36, 48, 72, 96, 120),
) -> list[dict]:
    if pairs is None:
        pairs = [
            (0.005, 0.005),
            (0.006, 0.006),
            (0.008, 0.008),
            (0.01, 0.01),
            (0.012, 0.012),
            (0.015, 0.015),
            (0.02, 0.02),
            (0.005, 0.01),
            (0.005, 0.015),
            (0.006, 0.012),
            (0.008, 0.016),
            (0.01, 0.015),
            (0.01, 0.02),
            (0.008, 0.024),
            (0.015, 0.01),
            (0.01, 0.005),
            (0.02, 0.01),
        ]
    sides = (
        dict(allow_long=True, allow_short=True),
        dict(allow_long=True, allow_short=False),
        dict(allow_long=False, allow_short=True),
    )
    out = []
    for (sl, tp), mh, side in itertools.product(pairs, holds, sides):
        out.append(
            {
                "rr": (tp / sl) if sl else 1.0,
                "max_hold": mh,
                "atr": "ATR_14",
                "sl_pct": sl,
                "tp_pct": tp,
                "fixed_lot": FIXED_LOT,
                "spread_price": SPREAD_PRICE,
                "commission_per_lot_rt": COMMISSION_PER_LOT_RT,
                "start_cap": START_CAP,
                "reward_label": f"{sl:.1%}/{tp:.1%}",
                **side,
            }
        )
    return out


def specs() -> list[tuple[str, dict]]:
    sess_opts = [
        None,
        ("london",),
        ("ny",),
        ("overlap",),
        ("london", "overlap", "ny"),
        ("london", "overlap"),
        ("overlap", "ny"),
    ]
    base = [
        ("of_absorption", {}),
        ("of_sweep_absorb", {}),
        ("of_imb_absorb", {"n": 2}),
        ("of_imb_absorb", {"n": 3}),
        ("of_clv_fade", {}),
        ("of_stacked_imb", {}),
        ("of_delta_flip", {"n": 3}),
        ("of_cvd_cross", {}),
        ("of_cvd_rsi", {"span": 20, "rsi": 14, "os": 35, "ob": 65}),
        ("ob_retest", {}),
        ("cisd", {}),
        ("fvg_reclaim", {}),
        ("fvg_ce", {}),
        ("fvg_rsi", {"rsi": 14, "os": 40, "ob": 80}),
        ("fvg_rsi", {"rsi": 14, "os": 35, "ob": 65}),
        ("fvg_rsi", {"rsi": 7, "os": 30, "ob": 70}),
        ("inside_break", {}),
        ("fakey", {}),
        ("turtle_soup", {"period": 20, "wick": 0.35}),
        ("turtle_soup", {"period": 10, "wick": 0.35}),
        ("wyckoff_spring", {"swing": 20, "wick": 0.45}),
        ("wyckoff_upthrust", {"swing": 20, "wick": 0.45}),
        ("liq_sweep", {"swing": 20, "wick": 0.45}),
        ("liq_sweep_fvg", {"swing": 20, "wick": 0.45}),
        ("mss_entry", {"swing": 20}),
        ("orb_break", {"orb": "LONDON"}),
        ("orb_break", {"orb": "NY"}),
        ("orb_fade", {"orb": "LONDON"}),
        ("rsi_cross", {"rsi": 14, "os": 30, "ob": 70}),
        ("rsi_cross", {"rsi": 7, "os": 25, "ob": 75}),
        ("rsi_level", {"rsi": 14, "os": 35, "ob": 65, "band": 5}),
        ("stoch_cross", {"period": 14, "os": 20, "ob": 80}),
        ("bb_rsi_fade", {"period": 20, "k": 2.0, "rsi": 14, "os": 35, "ob": 65}),
        ("donch_fade", {"period": 20, "rsi": 14, "os": 35, "ob": 65}),
        ("pin_rsi", {"rsi": 14, "os": 30, "ob": 70, "wick": 0.55, "body": 0.35}),
        ("engulf", {"ema": 21}),
        ("silver_bullet", {"hours": (14, 15, 16)}),
    ]
    out = []
    for kind, extra in base:
        for sess in sess_opts:
            out.append((kind, {**extra, "sessions": sess}))
    return out


def dump(tested: int, hits: int, best: dict | None, near: list) -> None:
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits_5x_oos": hits,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best.get("name") if best else None,
                "best_eq": best.get("full_eq") if best else None,
                "best_mult": best.get("mult") if best else None,
                "near": len(near),
                "gate": f"full_eq>={TARGET_EQ} IS+OOS net>0 after fees lot={FIXED_LOT}",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    SUMMARY.write_text(
        json.dumps({"tested": tested, "hits": hits, "best": best, "near": near[:25]}, indent=2, default=str)
    )


def maybe_save_champ(row: dict) -> None:
    prev = None
    if CHAMP.exists():
        try:
            prev = json.loads(CHAMP.read_text()).get("champion")
        except Exception:
            prev = None
    better = prev is None or (
        row["min_wr"],
        row["full_eq"],
        -row["full_dd"],
    ) > (
        float(prev.get("min_wr", 0)),
        float(prev.get("full_eq", 0)),
        -float(prev.get("full_dd", 1)),
    )
    if not better:
        return
    CHAMP.write_text(
        json.dumps(
            {
                "goal": "5x $500→$2500 after fees, maximize WR",
                "assumptions": {
                    "start_cap": START_CAP,
                    "lot": FIXED_LOT,
                    "spread": SPREAD_PRICE,
                    "commission_rt_per_lot": COMMISSION_PER_LOT_RT,
                },
                "champion": row,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )
    log(
        f"NEW 5x CHAMP {row['name']} minWR={row['min_wr']:.1%} "
        f"eq=${row['full_eq']:.0f} ({row['mult']:.2f}x) oos=${row['oos_net']:.0f}"
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    log(
        f"=== 5X HUNT lot={FIXED_LOT} cap=${START_CAP:.0f} target_eq=${TARGET_EQ:.0f} "
        f"spread=${SPREAD_PRICE} comm=${COMMISSION_PER_LOT_RT}/lot RT (any SL%/TP%) ==="
    )
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("5m", "15m", "30m"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])

    sp = specs()
    exs = fee_asym_exits()
    log(f"specs={len(sp)} exits={len(exs)} tfs={list(frames)}")

    tested = hits = 0
    best: dict | None = None
    near: list[dict] = []

    while True:
        RNG.shuffle(sp)
        for kind, base in sp:
            for invert in (False, True):
                pp = {**base}
                if invert:
                    pp["invert"] = True
                for tf, df in frames.items():
                    try:
                        L0, S0 = gen_signals(df, kind, {k: v for k, v in pp.items() if k != "invert"})
                    except Exception as e:
                        if tested < 5:
                            log(f"skip {kind}: {e}")
                        continue
                    L0 = np.asarray(L0, bool)
                    S0 = np.asarray(S0, bool)
                    if invert:
                        L0, S0 = S0, L0
                    if int(L0.sum() + S0.sum()) < 40:
                        continue
                    n = len(df)
                    split = int(n * 0.7)
                    warm = 250
                    for ex in exs:
                        tested += 1
                        kw = _exit_sim_kwargs(ex)
                        full = simulate(df, L0, S0, start_i=warm, end_i=n - 2, **kw)
                        if full["trades"] < 40 or full["final_equity"] < TARGET_EQ:
                            if tested % 500 == 0:
                                dump(tested, hits, best, near)
                            continue
                        is_m = simulate(df, L0, S0, start_i=warm, end_i=split, **kw)
                        oos_m = simulate(df, L0, S0, start_i=split, end_i=n - 2, **kw)
                        if is_m["trades"] < 25 or oos_m["trades"] < 15:
                            continue
                        if is_m["net_pnl"] <= 0 or oos_m["net_pnl"] <= 0:
                            continue
                        min_wr = min(is_m["winrate"], oos_m["winrate"])
                        side_tag = (
                            "LS"
                            if ex.get("allow_long") and ex.get("allow_short")
                            else ("L" if ex.get("allow_long") else "S")
                        )
                        sess = pp.get("sessions")
                        sess_tag = "all" if not sess else "+".join(sess)
                        name = (
                            f"{'inv|' if invert else '5x|'}{kind}|{ex['reward_label']}|"
                            f"h{ex['max_hold']}|{tf}|{side_tag}|{sess_tag}"
                        )
                        yok = year_stable(df, L0, S0, dict(ex), min_trades=20)
                        row = {
                            "name": name,
                            "min_wr": min_wr,
                            "full_wr": full["winrate"],
                            "is_wr": is_m["winrate"],
                            "oos_wr": oos_m["winrate"],
                            "full_eq": full["final_equity"],
                            "mult": full["final_equity"] / START_CAP,
                            "full_net": full["net_pnl"],
                            "is_net": is_m["net_pnl"],
                            "oos_net": oos_m["net_pnl"],
                            "full_n": full["trades"],
                            "is_n": is_m["trades"],
                            "oos_n": oos_m["trades"],
                            "full_tpd": full["trades_per_day"],
                            "full_dd": full["max_dd"],
                            "full_pf": full["profit_factor"],
                            "charges": full.get("total_charges"),
                            "year_ok": yok,
                            "params": {"kind": kind, **pp, "tf": tf, "exit": dict(ex)},
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        hits += 1
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        sk = (min_wr, row["full_eq"], -row["full_dd"])
                        if best is None or sk > best.get("_sk", (0, 0, 0)):
                            best = {**row, "_sk": sk}
                            maybe_save_champ(row)
                        if min_wr >= 0.55:
                            near.append(row)
                            near.sort(key=lambda r: (r["min_wr"], r["full_eq"]), reverse=True)
                            near[:] = near[:40]
                            log(
                                f"5x HIT {min_wr:.1%} eq=${row['full_eq']:.0f} "
                                f"oos=${row['oos_net']:.0f} year={yok} | {name}"
                            )
                        if tested % 200 == 0:
                            dump(tested, hits, best, near)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            log(f"5x tested={tested} hits={hits} best={bw}")

        dump(tested, hits, best, near)
        log(f"5x pass done tested={tested} hits={hits}; reshuffle")


if __name__ == "__main__":
    main()
