#!/usr/bin/env python3
"""IS-first 2y hunt for ≥80% WR (then require OOS≈IS).

Skips full-sample probe filter — evaluates IS WR directly, then OOS.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np

from rr1_hf80_10d_hunt import MIN_NET_TP, net_on_tp, simulate_multi
from rr1_hf80_2y_hunt import simulate_one
from rr1_search import add_indicators, gen_signals, load_frames, log

RNG = random.Random(77)
OUT = Path("rr1_search_results")
LEADER = OUT / "HF80_2Y_ISFIRST_LEADERBOARD.jsonl"
SUMMARY = OUT / "HF80_2Y_ISFIRST_SUMMARY.json"
CHAMP = OUT / "HF80_2Y_ISFIRST_CHAMPION.json"
CEILING = OUT / "HF80_2Y_WR_CEILING.json"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    LEADER.write_text("")
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("15m", "30m", "1h", "2h", "4h"):
        if tf not in raw:
            continue
        log(f"indicators {tf}…")
        frames[tf] = add_indicators(raw[tf])

    windows = {tf: (250, int(len(df) * 0.7), len(df) - 2) for tf, df in frames.items()}
    for tf, df in frames.items():
        w, s, e = windows[tf]
        log(f"{tf} bars={len(df)} split={df.index[s]}")

    # Signal grid
    sess_opts = [None, ("london",), ("overlap",), ("ny",), ("london", "overlap"), ("overlap", "ny"), ("asian",)]
    kinds = []
    for sess, rsi, os, ob in itertools.product(
        sess_opts, (2, 3, 5, 7, 14), (1, 2, 5, 8, 10, 15, 20, 25), (75, 80, 85, 90, 95, 98)
    ):
        kinds.append(("rsi_cross", {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}))
        kinds.append(("rsi_level", {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}))
    for sess, (per, k) in itertools.product(sess_opts, [(20, 2.0), (20, 2.5), (50, 2.0)]):
        kinds.append(("bb_fade", {"period": per, "k": k, "sessions": sess}))
        kinds.append(("bb_rsi_fade", {"period": per, "k": k, "rsi": 2, "os": 10, "ob": 90, "sessions": sess}))
        kinds.append(("bb_rsi_fade", {"period": per, "k": k, "rsi": 5, "os": 15, "ob": 85, "sessions": sess}))
    for sess, per, os, ob in itertools.product(sess_opts[:4], (5, 14), (10, 20), (80, 90)):
        kinds.append(("stoch_cross", {"period": per, "os": os, "ob": ob, "sessions": sess}))
    for z, th in itertools.product((20, 40), (2.0, 2.5, 3.0)):
        kinds.append(("zscore_fade", {"period": z, "th": th}))

    # Exit grid — SL=2×TP bias, lots that clear $2
    exits = []
    for lot, tp, mh, side in itertools.product(
        (0.02, 0.05, 0.1),
        (1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0),
        (8, 12, 20, 30, 48, 72, 96),
        (
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
            dict(allow_long=True, allow_short=True),
        ),
    ):
        if net_on_tp(tp, lot) < MIN_NET_TP:
            continue
        exits.append(
            dict(tp_dist=tp, sl_dist=2.0 * tp, max_hold=mh, lot=lot, sl_mult=2.0, net_tp=net_on_tp(tp, lot), **side)
        )
    log(f"kinds={len(kinds)} exits={len(exits)}")

    tested = hits = 0
    best80 = None
    # ceiling tracking: best min(IS,OOS), best IS alone, best OOS alone
    ceiling = {"best_min": None, "best_is": None, "best_oos": None, "ge75": [], "ge78": []}
    near = []

    def consider_ceiling(row):
        for key, metric in (("best_min", min(row["is_wr"], row["oos_wr"])), ("best_is", row["is_wr"]), ("best_oos", row["oos_wr"])):
            prev = ceiling[key]
            if prev is None or metric > prev["metric"]:
                ceiling[key] = {**row, "metric": metric}
        if min(row["is_wr"], row["oos_wr"]) >= 0.75:
            ceiling["ge75"].append(row)
            ceiling["ge75"] = sorted(ceiling["ge75"], key=lambda r: (min(r["is_wr"], r["oos_wr"]), -r["wr_gap"]), reverse=True)[:30]
        if min(row["is_wr"], row["oos_wr"]) >= 0.78:
            ceiling["ge78"].append(row)
            ceiling["ge78"] = sorted(ceiling["ge78"], key=lambda r: (min(r["is_wr"], r["oos_wr"]), -r["wr_gap"]), reverse=True)[:30]

    # Precompute signals
    cache = []
    for kind, base in kinds:
        for invert in (False, True):
            for tf, df in frames.items():
                warm, split, end = windows[tf]
                try:
                    L, S = gen_signals(df, kind, base)
                    L, S = np.asarray(L, bool), np.asarray(S, bool)
                except Exception:
                    continue
                if invert:
                    L, S = S, L
                # optional EMA50 trend align variants
                ema = df["EMA_50"].shift(1)
                above = (df["Close"] > ema).fillna(False).to_numpy(bool)
                below = (df["Close"] < ema).fillna(False).to_numpy(bool)
                variants = [
                    ("raw", L, S),
                    ("withTrend", L & above, S & below),
                    ("longOnlyTrend", L & above, np.zeros(len(L), bool)),
                    ("shortOnlyTrend", np.zeros(len(L), bool), S & below),
                ]
                for vname, Lv, Sv in variants:
                    nsig = int(Lv[warm:split].sum() + Sv[warm:split].sum())  # IS signals
                    if nsig < 40:
                        continue
                    cache.append((kind, {**base, **({"invert": True} if invert else {})}, tf, Lv, Sv, invert, vname, nsig))
    RNG.shuffle(cache)
    log(f"signal cache={len(cache)}")

    # Sort: prefer fewer IS signals (more selective → higher WR chance)
    cache.sort(key=lambda x: x[7])

    t0 = time.time()
    for idx, (kind, pp, tf, L, S, invert, vname, nsig) in enumerate(cache):
        df = frames[tf]
        warm, split, end = windows[tf]
        # Sample exits; denser for selective signals
        n_ex = 80 if nsig <= 120 else 40
        sample = RNG.sample(exits, min(len(exits), n_ex))
        for ex in sample:
            for sim, mode in ((simulate_one, "one"), (simulate_multi, "multi")):
                # Skip multi if very selective — similar
                if mode == "multi" and nsig < 80:
                    continue
                tested += 1
                is_m = sim(
                    df, L, S,
                    tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"], lot=ex["lot"],
                    allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                    start_i=warm, end_i=split,
                )
                if is_m["trades"] < 40 or is_m["net"] <= 0:
                    continue
                # Track high IS even before OOS
                if is_m["winrate"] < 0.75:
                    continue
                oos = sim(
                    df, L, S,
                    tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"], lot=ex["lot"],
                    allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                    start_i=split, end_i=end,
                )
                if oos["trades"] < 20:
                    continue
                gap = abs(is_m["winrate"] - oos["winrate"])
                side = "LS" if ex["allow_long"] and ex["allow_short"] else ("L" if ex["allow_long"] else "S")
                name = (
                    f"{'inv|' if invert else ''}{kind}|{vname}|tp{ex['tp_dist']:g}/sl{ex['sl_dist']:g}|"
                    f"h{ex['max_hold']}|{tf}|{side}|lot{ex['lot']:g}|{mode}"
                )
                row = dict(
                    name=name, kind=kind, variant=vname, params=pp, exit=ex, invert=invert, mode=mode, tf=tf,
                    is_wr=is_m["winrate"], oos_wr=oos["winrate"], wr_gap=gap,
                    is_n=is_m["trades"], oos_n=oos["trades"],
                    is_net=is_m["net"], oos_net=oos["net"],
                    is_dd=is_m["dd"], oos_dd=oos["dd"],
                    tpd_is=is_m["tpd"], avg_win_is=is_m["avg_win"], pf_is=is_m["pf"],
                    max_open=is_m.get("max_concurrent", 1),
                )
                consider_ceiling(row)

                if (
                    is_m["winrate"] >= 0.80
                    and oos["winrate"] >= 0.80
                    and gap <= 0.05
                    and oos["net"] > 0
                    and is_m["avg_win"] >= MIN_NET_TP * 0.75
                    and oos["avg_win"] >= MIN_NET_TP * 0.75
                ):
                    full = sim(
                        df, L, S,
                        tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"], lot=ex["lot"],
                        allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                        start_i=warm, end_i=end,
                    )
                    row.update(
                        full_wr=full["winrate"], full_net=full["net"], full_dd=full["dd"],
                        full_n=full["trades"], tpd=full["tpd"], pf=full["pf"], avg_win=full["avg_win"],
                    )
                    with open(LEADER, "a") as f:
                        f.write(json.dumps(row, default=str) + "\n")
                    hits += 1
                    sk = (min(row["is_wr"], row["oos_wr"]), -gap, row["oos_net"], row["full_net"])
                    if best80 is None or sk > best80.get("_sk", (0, 0, 0, 0)):
                        best80 = {**row, "_sk": sk}
                        CHAMP.write_text(
                            json.dumps(
                                {
                                    "rule": "2y IS&OOS≥80% gap≤5pp TP net≥$2 SL=2×TP causal",
                                    "champion": {k: v for k, v in row.items()},
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                },
                                indent=2,
                                default=str,
                            )
                        )
                    near.append(row)
                    near = sorted(near, key=lambda r: (min(r["is_wr"], r["oos_wr"]), -r["wr_gap"]), reverse=True)[:40]
                    log(
                        f"HIT80 IS {is_m['winrate']:.1%} OOS {oos['winrate']:.1%} gap={gap:.1%} "
                        f"netOOS=${oos['net']:.0f} FULL=${full['net']:.0f} | {name}"
                    )

        if (idx + 1) % 50 == 0:
            bmin = ceiling["best_min"]
            msg = (
                f"ceil minWR={bmin['metric']:.1%} IS={bmin['is_wr']:.1%} OOS={bmin['oos_wr']:.1%} | {bmin['name']}"
                if bmin
                else "ceil n/a"
            )
            log(f"progress cache {idx+1}/{len(cache)} tested={tested} hits80={hits} {msg}")
            CEILING.write_text(
                json.dumps(
                    {
                        "tested": tested,
                        "hits80": hits,
                        "best_min": ceiling["best_min"],
                        "best_is": ceiling["best_is"],
                        "best_oos": ceiling["best_oos"],
                        "n_ge75": len(ceiling["ge75"]),
                        "n_ge78": len(ceiling["ge78"]),
                        "ge75_top": ceiling["ge75"][:10],
                        "ge78_top": ceiling["ge78"][:10],
                        "elapsed_s": time.time() - t0,
                    },
                    indent=2,
                    default=str,
                )
            )

    SUMMARY.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits80": hits,
                "best80": best80,
                "ceiling": {
                    "best_min": ceiling["best_min"],
                    "best_is": ceiling["best_is"],
                    "best_oos": ceiling["best_oos"],
                    "n_ge75": len(ceiling["ge75"]),
                    "n_ge78": len(ceiling["ge78"]),
                    "ge78_top": ceiling["ge78"][:15],
                    "ge75_top": ceiling["ge75"][:15],
                },
                "near80": near[:20],
                "elapsed_s": time.time() - t0,
            },
            indent=2,
            default=str,
        )
    )
    CEILING.write_text(json.dumps(SUMMARY.read_text()))  # noop safeguard
    CEILING.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits80": hits,
                "best_min": ceiling["best_min"],
                "best_is": ceiling["best_is"],
                "best_oos": ceiling["best_oos"],
                "n_ge75": len(ceiling["ge75"]),
                "n_ge78": len(ceiling["ge78"]),
                "ge75_top": ceiling["ge75"][:15],
                "ge78_top": ceiling["ge78"][:15],
                "elapsed_s": time.time() - t0,
            },
            indent=2,
            default=str,
        )
    )
    log(f"DONE hits80={hits}")
    if best80:
        log(f"CHAMP {best80['name']}")
    elif ceiling["best_min"]:
        b = ceiling["best_min"]
        log(f"CEILING minWR={b['metric']:.1%} IS={b['is_wr']:.1%} OOS={b['oos_wr']:.1%} | {b['name']}")


if __name__ == "__main__":
    main()
