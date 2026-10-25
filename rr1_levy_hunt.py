#!/usr/bin/env python3
"""Guy Levy / marketscoach price-volume hunt (IG reel DerP1HHjpUg).

Rule (no real volume in XAU CSV → range effort proxy):
  Bullish: price and volume move together (up + high relative range).
  Bearish: heavy volume on drops, light volume on rallies (fade light bounces).

Hunt: any SL%/TP% after fees, $500 start, lot grid sized for DD≤20%.
Hard gate: max_dd ≤ 20% on IS, OOS, and FULL (reject otherwise).
Rank by min(WR_IS, WR_OOS) with IS+OOS net > 0. Also track 5× equity hits.
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

RNG = random.Random(1910)
LEADER = OUT / "LEVY_LEADERBOARD.jsonl"
PROGRESS = OUT / "levy_progress.json"
SUMMARY = OUT / "LEVY_SUMMARY.json"
CHAMP = OUT / "LEVY_CHAMPION.json"
TARGET_EQ = START_CAP * 5.0
# User hard rule: holy drawdown must not exceed 20%.
MAX_DD = 0.20
# 0.02 lot blows past 20% DD on the regime champ; keep smaller sizes in the grid.
LOT_GRID = (0.02, 0.01, 0.005, 0.003, 0.0024, 0.001)


def exits() -> list[dict]:
    pairs = [
        (0.005, 0.005),
        (0.008, 0.008),
        (0.01, 0.01),
        (0.012, 0.012),
        (0.015, 0.015),
        (0.005, 0.01),
        (0.005, 0.015),
        (0.008, 0.016),
        (0.01, 0.02),
        (0.01, 0.015),
        (0.015, 0.01),
        (0.01, 0.005),
    ]
    holds = (24, 36, 48, 72, 96)
    sides = (
        dict(allow_long=True, allow_short=True),
        dict(allow_long=True, allow_short=False),
        dict(allow_long=False, allow_short=True),
    )
    out = []
    for (sl, tp), mh, side, lot in itertools.product(pairs, holds, sides, LOT_GRID):
        out.append(
            {
                "rr": tp / sl if sl else 1.0,
                "max_hold": mh,
                "atr": "ATR_14",
                "sl_pct": sl,
                "tp_pct": tp,
                "fixed_lot": lot,
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
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
    ]
    out: list[tuple[str, dict]] = []
    for sess, rvol, thr, clv in itertools.product(
        sess_opts, (10, 20, 40), (1.1, 1.2, 1.4, 1.6), (0.55, 0.65, 0.75)
    ):
        out.append(("levy_accum", {"sessions": sess, "rvol": rvol, "rvol_thr": thr, "clv": clv}))
    for sess, rvol, hi, lo, look in itertools.product(
        sess_opts, (10, 20), (1.2, 1.4, 1.6), (0.6, 0.8, 0.9), (4, 6, 10)
    ):
        out.append(
            ("levy_distrib", {"sessions": sess, "rvol": rvol, "hi_thr": hi, "lo_thr": lo, "look": look})
        )
    for sess, rvol, thr in itertools.product(sess_opts, (10, 20, 40), (1.1, 1.2, 1.4)):
        out.append(("levy_pv_confirm", {"sessions": sess, "rvol": rvol, "rvol_thr": thr}))
    for sess, rvol, lo, look, mdn in itertools.product(
        sess_opts, (10, 20), (0.6, 0.8, 0.9), (6, 8, 12), (2, 3, 4)
    ):
        out.append(
            (
                "levy_light_bounce_fade",
                {"sessions": sess, "rvol": rvol, "lo_thr": lo, "look": look, "min_dn": mdn},
            )
        )
    for sess, look, rvol, thr in itertools.product(
        sess_opts, (8, 16, 24), (10, 20), (1.0, 1.1, 1.3)
    ):
        out.append(("levy_regime", {"sessions": sess, "look": look, "rvol": rvol, "rvol_thr": thr}))
        out.append(("levy_regime_pull", {"sessions": sess, "look": look, "rvol": rvol, "lo_thr": 0.85}))
    return out


def dump(tested: int, hits: int, best: dict | None, near: list, dd_ok: int = 0) -> None:
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits_5x_oos": hits,
                "dd20_ok": dd_ok,
                "max_dd_cap": MAX_DD,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best.get("name") if best else None,
                "best_eq": best.get("full_eq") if best else None,
                "best_net": best.get("full_net") if best else None,
                "best_dd": best.get("full_dd") if best else None,
                "near": len(near),
                "source": "Guy Levy IG reel DerP1HHjpUg — price+volume (range proxy)",
                "gate": f"max_dd<={MAX_DD:.0%} IS+OOS+FULL, IS+OOS net>0",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits": hits,
                "dd20_ok": dd_ok,
                "max_dd_cap": MAX_DD,
                "best": best,
                "near": near[:25],
            },
            indent=2,
            default=str,
        )
    )


def maybe_save(row: dict) -> None:
    if row["full_dd"] > MAX_DD:
        return
    prev = None
    prev_meta = None
    if CHAMP.exists():
        try:
            prev_meta = json.loads(CHAMP.read_text())
            prev = prev_meta.get("champion")
        except Exception:
            prev = None
    # Prefer DD-compliant champs only; ignore prior champs that breached the cap.
    if prev is not None and float(prev.get("full_dd", 1.0)) > MAX_DD:
        prev = None
    key = (row["min_wr"], row["full_eq"], -row["full_dd"])
    old = (
        (float(prev.get("min_wr", 0)), float(prev.get("full_eq", 0)), -float(prev.get("full_dd", 1)))
        if prev
        else None
    )
    if old is not None and key <= old:
        return
    payload = {
        "source": "Guy Levy / marketscoach price-volume (range proxy)",
        "reel": "https://www.instagram.com/reel/DerP1HHjpUg/",
        "note": "XAU CSV has no real volume — LEVY_VOL = bar range / prior SMA",
        "gate": f"max_dd <= {MAX_DD:.0%} on IS/OOS/FULL",
        "champion": row,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if prev_meta and prev_meta.get("dd_cap_note"):
        payload["dd_cap_note"] = prev_meta["dd_cap_note"]
    CHAMP.write_text(json.dumps(payload, indent=2, default=str))
    log(
        f"NEW LEVY CHAMP {row['name']} minWR={row['min_wr']:.1%} "
        f"eq=${row['full_eq']:.0f} dd={row['full_dd']:.1%}"
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    log(
        f"=== LEVY PV HUNT lots={LOT_GRID} cap=${START_CAP:.0f} "
        f"MAX_DD={MAX_DD:.0%} spread=${SPREAD_PRICE} (range-proxy volume) ==="
    )
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("5m", "15m", "30m"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])

    sp = specs()
    exs = exits()
    log(f"specs={len(sp)} exits={len(exs)} tfs={list(frames)}")

    tested = hits = dd_ok = 0
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
                        if tested < 8:
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
                        if full["trades"] < 40:
                            if tested % 400 == 0:
                                dump(tested, hits, best, near, dd_ok)
                            continue
                        is_m = simulate(df, L0, S0, start_i=warm, end_i=split, **kw)
                        oos_m = simulate(df, L0, S0, start_i=split, end_i=n - 2, **kw)
                        if is_m["trades"] < 25 or oos_m["trades"] < 15:
                            continue
                        if is_m["net_pnl"] <= 0 or oos_m["net_pnl"] <= 0:
                            continue
                        # Hard reject: holy DD must stay ≤ 20% on every split.
                        if (
                            full["max_dd"] > MAX_DD
                            or is_m["max_dd"] > MAX_DD
                            or oos_m["max_dd"] > MAX_DD
                        ):
                            if tested % 400 == 0:
                                dump(tested, hits, best, near, dd_ok)
                            continue
                        dd_ok += 1
                        min_wr = min(is_m["winrate"], oos_m["winrate"])
                        side_tag = (
                            "LS"
                            if ex.get("allow_long") and ex.get("allow_short")
                            else ("L" if ex.get("allow_long") else "S")
                        )
                        sess = pp.get("sessions")
                        sess_tag = "all" if not sess else "+".join(sess)
                        lot = float(ex.get("fixed_lot", FIXED_LOT))
                        name = (
                            f"{'inv|' if invert else 'levy|'}{kind}|{ex['reward_label']}|"
                            f"h{ex['max_hold']}|{tf}|{side_tag}|{sess_tag}|lot{lot:g}"
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
                            "is_dd": is_m["max_dd"],
                            "oos_dd": oos_m["max_dd"],
                            "full_pf": full["profit_factor"],
                            "lot": lot,
                            "charges": full.get("total_charges"),
                            "year_ok": yok,
                            "params": {"kind": kind, **pp, "tf": tf, "exit": dict(ex)},
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        if full["final_equity"] >= TARGET_EQ:
                            hits += 1
                        sk = (min_wr, row["full_eq"], -row["full_dd"])
                        if best is None or sk > best.get("_sk", (0, 0, 0)):
                            best = {**row, "_sk": sk}
                            maybe_save(row)
                        if min_wr >= 0.55 and (full["final_equity"] >= TARGET_EQ or min_wr >= 0.60):
                            near.append(row)
                            near.sort(
                                key=lambda r: (r["min_wr"], r["full_eq"], -r["full_dd"]),
                                reverse=True,
                            )
                            near[:] = near[:40]
                            log(
                                f"LEVY HIT {min_wr:.1%} eq=${row['full_eq']:.0f} "
                                f"dd={row['full_dd']:.1%} oos=${row['oos_net']:.0f} "
                                f"year={yok} | {name}"
                            )
                        if tested % 200 == 0:
                            dump(tested, hits, best, near, dd_ok)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            bd = f"{best['full_dd']:.1%}" if best else "n/a"
                            log(f"levy tested={tested} dd20={dd_ok} hits5x={hits} best={bw} dd={bd}")

        dump(tested, hits, best, near, dd_ok)
        log(f"levy pass done tested={tested} dd20={dd_ok} hits5x={hits}; reshuffle")


if __name__ == "__main__":
    main()
