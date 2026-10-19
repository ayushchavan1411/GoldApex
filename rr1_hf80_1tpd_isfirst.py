#!/usr/bin/env python3
"""IS-first hunt for WR≥80% with tpd≥1 on 2y (then require OOS≈IS).

Efficiency: one fixed probe exit per signal; only expand if IS WR≥72% and tpd≥1.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from rr1_hf80_10d_hunt import MIN_NET_TP, local_signals, net_on_tp, simulate_multi
from rr1_hf80_2y_hunt import simulate_one
from rr1_search import add_indicators, gen_signals, load_frames, log

OUT = Path("rr1_search_results")
LEADER = OUT / "HF80_1TPD_LEADERBOARD.jsonl"
CHAMP = OUT / "HF80_1TPD_CHAMPION.json"
SUMMARY = OUT / "HF80_1TPD_SUMMARY.json"
CEILING = OUT / "HF80_1TPD_CEILING.json"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    LEADER.write_text("")
    log("1TPD IS-first load…")
    raw = load_frames(include_resampled=True)
    frames = {tf: add_indicators(raw[tf]) for tf in ("5m", "15m", "30m") if tf in raw}

    windows = {}
    for tf, df in frames.items():
        n = len(df)
        warm = 400 if tf == "5m" else 250
        windows[tf] = (warm, int(n * 0.7), n - 2)
        log(f"{tf} bars={n} split={df.index[windows[tf][1]]}")

    sess_opts = [
        None,
        ("london",),
        ("ny",),
        ("overlap",),
        ("asian",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
    ]
    kinds: list[tuple[str, dict]] = []
    for sess, rsi, os, ob in itertools.product(
        sess_opts, (2, 3, 5, 7, 14), (10, 15, 20, 25, 30, 35), (65, 70, 75, 80, 85)
    ):
        b = {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}
        kinds.append(("rsi_cross", dict(b)))
        kinds.append(("rsi_level", dict(b)))
        kinds.append(("rsi_recover", {"rsi": rsi, "os": os, "ob": ob}))
    for sess, (per, k) in itertools.product(
        sess_opts, [(10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5)]
    ):
        kinds.append(("bb_fade", {"period": per, "k": k, "sessions": sess}))
        kinds.append(
            ("bb_rsi_fade", {"period": per, "k": k, "rsi": 5, "os": 25, "ob": 75, "sessions": sess})
        )
        kinds.append(
            ("bb_rsi_fade", {"period": per, "k": k, "rsi": 3, "os": 20, "ob": 80, "sessions": sess})
        )
    for sess, per, os, ob in itertools.product(sess_opts[:5], (5, 9, 14), (15, 20, 25), (75, 80, 85)):
        kinds.append(("stoch_cross", {"period": per, "os": os, "ob": ob, "sessions": sess}))
    for z, th in itertools.product((20, 40), (1.0, 1.5, 2.0, 2.5)):
        kinds.append(("zscore_fade", {"period": z, "th": th}))
    for per, th in itertools.product((14, 20), (100, 150, 200)):
        kinds.append(("cci_fade", {"period": per, "th": th}))
    for rsi, os, ob in itertools.product((3, 5, 7), (20, 25, 30), (70, 75, 80)):
        kinds.append(("micro_reclaim", {"rsi": rsi, "os": os, "ob": ob}))
        kinds.append(("range_fade", {"look": 5, "rsi": rsi, "os": os, "ob": ob}))
        kinds.append(("range_fade", {"look": 8, "rsi": rsi, "os": os, "ob": ob}))
    for wick, body in itertools.product((0.45, 0.55, 0.65), (0.3,)):
        kinds.append(("wick_reject", {"wick": wick, "body": body}))
    kinds.append(("micro_bounce", {}))

    # Exit expand grid
    exits = []
    for lot, tp, sm, mh, side in itertools.product(
        (0.02, 0.05, 0.1),
        (1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0),
        (1.5, 1.75, 2.0),
        (8, 12, 20, 30, 48),
        (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        ),
    ):
        if net_on_tp(tp, lot) < MIN_NET_TP:
            continue
        exits.append(
            dict(tp_dist=tp, sl_dist=tp * sm, max_hold=mh, lot=lot, sl_mult=sm, net_tp=net_on_tp(tp, lot), **side)
        )
    log(f"kinds={len(kinds)} exits={len(exits)}")

    tested = hits = 0
    best80 = None
    best = None  # best min WR at tpd≥1
    board = []
    t0 = time.time()

    # Fixed IS probe exit (WR-biased)
    probe_tp = 2.0
    while net_on_tp(probe_tp, 0.05) < MIN_NET_TP:
        probe_tp += 0.3

    for tf, df in frames.items():
        warm, split, end = windows[tf]
        for kind, base in kinds:
            for invert in (False, True):
                try:
                    if kind in ("rsi_recover", "wick_reject", "micro_bounce"):
                        L, S = local_signals(df, kind, base)
                    else:
                        L, S = gen_signals(df, kind, base)
                        L, S = np.asarray(L, bool), np.asarray(S, bool)
                except Exception:
                    continue
                if invert:
                    L, S = S, L
                nsig = int(L[warm:end].sum() + S[warm:end].sum())
                if nsig < 500 or nsig > 18000:
                    continue

                # IS multi probe
                is_pr = simulate_multi(
                    df, L, S,
                    tp_dist=probe_tp, sl_dist=2 * probe_tp, max_hold=20, lot=0.05,
                    allow_long=True, allow_short=True, start_i=warm, end_i=split,
                )
                if is_pr["tpd"] < 1.0 or is_pr["trades"] < 200:
                    continue
                if is_pr["winrate"] < 0.70:
                    continue
                log(
                    f"IS-PROBE {is_pr['winrate']:.1%} tpd={is_pr['tpd']:.2f} n={is_pr['trades']} "
                    f"| {kind} {tf} inv={invert} sess={base.get('sessions')}"
                )

                for ex in exits:
                    # Skip exits unlikely to keep frequency (very large TP / short hold mismatch)
                    for sim, mode in ((simulate_multi, "multi"), (simulate_one, "one")):
                        tested += 1
                        is_m = sim(
                            df, L, S,
                            tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                            lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                            start_i=warm, end_i=split,
                        )
                        if is_m["tpd"] < 1.0 or is_m["trades"] < 200 or is_m["net"] <= 0:
                            continue
                        if is_m["winrate"] < 0.72:
                            continue
                        oos = sim(
                            df, L, S,
                            tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                            lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                            start_i=split, end_i=end,
                        )
                        if oos["tpd"] < 0.9 or oos["trades"] < 80 or oos["net"] <= 0:
                            continue
                        gap = abs(is_m["winrate"] - oos["winrate"])
                        full = sim(
                            df, L, S,
                            tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                            lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                            start_i=warm, end_i=end,
                        )
                        if full["tpd"] < 1.0:
                            continue
                        side = (
                            "LS"
                            if ex["allow_long"] and ex["allow_short"]
                            else ("L" if ex["allow_long"] else "S")
                        )
                        name = (
                            f"{'inv|' if invert else ''}{kind}|tp{ex['tp_dist']:g}/sl{ex['sl_dist']:g}|"
                            f"h{ex['max_hold']}|{tf}|{side}|lot{ex['lot']:g}|{mode}"
                        )
                        row = dict(
                            name=name, kind=kind, params={**base, **({"invert": True} if invert else {})},
                            exit=ex, invert=invert, mode=mode, tf=tf,
                            is_wr=is_m["winrate"], oos_wr=oos["winrate"], full_wr=full["winrate"],
                            wr_gap=gap, is_n=is_m["trades"], oos_n=oos["trades"], full_n=full["trades"],
                            is_tpd=is_m["tpd"], oos_tpd=oos["tpd"], full_tpd=full["tpd"],
                            is_net=is_m["net"], oos_net=oos["net"], full_net=full["net"],
                            is_dd=is_m["dd"], oos_dd=oos["dd"], full_dd=full["dd"],
                            pf=full["pf"], avg_win=full["avg_win"], max_open=full.get("max_concurrent", 1),
                            net_tp_design=ex["net_tp"],
                            time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        mn = min(row["is_wr"], row["oos_wr"])
                        if best is None or (mn, -gap, row["full_tpd"], row["oos_net"]) > (
                            min(best["is_wr"], best["oos_wr"]), -best["wr_gap"], best["full_tpd"], best["oos_net"]
                        ):
                            best = row
                            log(
                                f"BEST min={mn:.1%} IS={row['is_wr']:.1%} OOS={row['oos_wr']:.1%} "
                                f"tpd={row['full_tpd']:.2f} gap={gap:.1%} | {name}"
                            )
                            CEILING.write_text(
                                json.dumps(
                                    {
                                        "best_at_tpd1": best,
                                        "hits80": hits,
                                        "tested": tested,
                                        "elapsed_s": time.time() - t0,
                                    },
                                    indent=2,
                                    default=str,
                                )
                            )
                        if mn >= 0.70 and gap <= 0.08:
                            board.append(row)
                            board.sort(
                                key=lambda r: (min(r["is_wr"], r["oos_wr"]), r["full_tpd"], -r["wr_gap"]),
                                reverse=True,
                            )
                            board[:] = board[:50]

                        if (
                            mn >= 0.80
                            and gap <= 0.05
                            and is_m["avg_win"] >= MIN_NET_TP * 0.75
                            and oos["avg_win"] >= MIN_NET_TP * 0.75
                        ):
                            with open(LEADER, "a") as f:
                                f.write(json.dumps(row, default=str) + "\n")
                            hits += 1
                            sk = (mn, row["full_tpd"], -gap, row["oos_net"], -row["full_dd"])
                            if best80 is None or sk > best80.get("_sk", (0, 0, 0, 0, 0)):
                                best80 = {**row, "_sk": sk}
                                CHAMP.write_text(
                                    json.dumps(
                                        {
                                            "rule": "2y IS&OOS≥80% gap≤5pp AND tpd≥1, TP≥$2, SL≤2×TP",
                                            "champion": {k: v for k, v in row.items()},
                                            "time": row["time"],
                                        },
                                        indent=2,
                                        default=str,
                                    )
                                )
                            log(
                                f"HIT80+1TPD IS {is_m['winrate']:.1%} OOS {oos['winrate']:.1%} "
                                f"tpd={full['tpd']:.2f} | {name}"
                            )

                if tested % 2000 == 0 and tested:
                    bm = f"{min(best['is_wr'], best['oos_wr']):.1%}@{best['full_tpd']:.2f}tpd" if best else "n/a"
                    log(f"… tested={tested} hits={hits} best={bm}")

    # Dedup board
    seen = set()
    top = []
    for r in board:
        key = r["name"]
        if key in seen:
            continue
        seen.add(key)
        top.append(r)
        if len(top) >= 25:
            break

    verdict = (
        f"FOUND: {best80['name']}"
        if best80
        else (
            f"NO 80% at tpd≥1. Best stable: IS {best['is_wr']:.1%} / OOS {best['oos_wr']:.1%} "
            f"(gap {best['wr_gap']:.1%}) tpd={best['full_tpd']:.2f} | {best['name']}"
            if best
            else "No tpd≥1 candidates with IS WR≥70% expanded"
        )
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits_80_1tpd": hits,
                "best80": best80,
                "best_at_tpd1": best,
                "top70": top,
                "elapsed_s": time.time() - t0,
                "verdict": verdict,
            },
            indent=2,
            default=str,
        )
    )
    CEILING.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits_80_1tpd": hits,
                "best_at_tpd1": best,
                "top70": top[:15],
                "elapsed_s": time.time() - t0,
                "verdict": verdict,
            },
            indent=2,
            default=str,
        )
    )
    log(f"DONE {verdict}")


if __name__ == "__main__":
    main()
