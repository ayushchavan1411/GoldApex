#!/usr/bin/env python3
"""Fast ceiling finder: max min(IS,OOS) WR subject to full tpd≥1 on 2y."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from rr1_hf80_10d_hunt import MIN_NET_TP, local_signals, net_on_tp, simulate_multi
from rr1_hf80_2y_hunt import simulate_one
from rr1_search import add_indicators, gen_signals, load_frames

OUT = Path("rr1_search_results")


def main() -> None:
    print("1TPD ceiling load…", flush=True)
    raw = load_frames(include_resampled=True)
    frames = {tf: add_indicators(raw[tf]) for tf in ("5m", "15m", "30m") if tf in raw}

    best = None
    hits80 = []
    board70 = []
    tested = 0
    t0 = time.time()

    lots = (0.02, 0.05, 0.1)
    tps = (1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    holds = (8, 12, 20, 30, 48)
    sides = [(True, True), (True, False), (False, True)]
    sess_opts = [None, ("london", "overlap"), ("overlap", "ny"), ("london",), ("ny",), ("asian",)]

    for tf, df in frames.items():
        n = len(df)
        warm = 400 if tf == "5m" else 250
        split = int(n * 0.7)
        end = n - 2
        print(f"{tf} bars={n} split={df.index[split]}", flush=True)

        kinds = []
        for sess, rsi, os, ob in itertools.product(
            sess_opts, (2, 3, 5, 7), (15, 20, 25, 30), (70, 75, 80, 85)
        ):
            kinds.append(("rsi_cross", {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}))
            kinds.append(("rsi_level", {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}))
            kinds.append(("rsi_recover", {"rsi": rsi, "os": os, "ob": ob}))
        for sess, (per, k) in itertools.product(sess_opts, [(10, 1.5), (20, 2.0), (20, 2.5)]):
            kinds.append(("bb_fade", {"period": per, "k": k, "sessions": sess}))
            kinds.append(
                ("bb_rsi_fade", {"period": per, "k": k, "rsi": 5, "os": 25, "ob": 75, "sessions": sess})
            )
        for sess, per, os, ob in itertools.product(sess_opts[:4], (5, 14), (20, 25), (75, 80)):
            kinds.append(("stoch_cross", {"period": per, "os": os, "ob": ob, "sessions": sess}))
        for z, th in itertools.product((20, 40), (1.5, 2.0)):
            kinds.append(("zscore_fade", {"period": z, "th": th}))
        kinds.append(("micro_bounce", {}))
        for wick in (0.5, 0.6):
            kinds.append(("wick_reject", {"wick": wick, "body": 0.3}))

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
                if nsig < 500 or nsig > 20000:
                    continue

                # probe multi
                ptp = 1.5 if net_on_tp(1.5, 0.05) >= MIN_NET_TP else 2.0
                pr = simulate_multi(
                    df, L, S, tp_dist=ptp, sl_dist=2 * ptp, max_hold=20, lot=0.05,
                    allow_long=True, allow_short=True, start_i=warm, end_i=end,
                )
                if pr["tpd"] < 1.0 or pr["trades"] < 400 or pr["winrate"] < 0.58:
                    continue
                print(
                    f"PROBE {pr['winrate']:.1%} tpd={pr['tpd']:.2f} n={pr['trades']} "
                    f"| {kind} {tf} inv={invert} sess={base.get('sessions')}",
                    flush=True,
                )

                for lot, tp, mh, (al, ash) in itertools.product(lots, tps, holds, sides):
                    if net_on_tp(tp, lot) < MIN_NET_TP:
                        continue
                    sl = 2.0 * tp
                    for sim, mode in ((simulate_multi, "multi"),):
                        # add one-pos only when probe WR already decent
                        pass
                    sims = [(simulate_multi, "multi")]
                    if pr["winrate"] >= 0.68:
                        sims.append((simulate_one, "one"))
                    for sim, mode in sims:
                        tested += 1
                        full = sim(
                            df, L, S, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                            allow_long=al, allow_short=ash, start_i=warm, end_i=end,
                        )
                        if full["tpd"] < 1.0 or full["net"] <= 0 or full["trades"] < 400:
                            continue
                        is_m = sim(
                            df, L, S, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                            allow_long=al, allow_short=ash, start_i=warm, end_i=split,
                        )
                        if is_m["trades"] < 200 or is_m["winrate"] < 0.65 or is_m["net"] <= 0:
                            continue
                        if is_m["tpd"] < 0.85:
                            continue
                        oos = sim(
                            df, L, S, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                            allow_long=al, allow_short=ash, start_i=split, end_i=end,
                        )
                        if oos["trades"] < 80 or oos["net"] <= 0 or oos["tpd"] < 0.85:
                            continue
                        gap = abs(is_m["winrate"] - oos["winrate"])
                        if gap > 0.08:  # slightly looser for ceiling scan
                            continue
                        side = "LS" if al and ash else ("L" if al else "S")
                        row = dict(
                            kind=kind, tf=tf, invert=invert, mode=mode, params=base,
                            tp=tp, sl=sl, hold=mh, lot=lot, side=side,
                            is_wr=is_m["winrate"], oos_wr=oos["winrate"], full_wr=full["winrate"],
                            wr_gap=gap, is_n=is_m["trades"], oos_n=oos["trades"], full_n=full["trades"],
                            is_tpd=is_m["tpd"], oos_tpd=oos["tpd"], full_tpd=full["tpd"],
                            is_net=is_m["net"], oos_net=oos["net"], full_net=full["net"],
                            full_dd=full["dd"], pf=full["pf"], avg_win=full["avg_win"],
                            max_open=full.get("max_concurrent", 1),
                        )
                        mn = min(row["is_wr"], row["oos_wr"])
                        if best is None or (mn, row["full_tpd"], row["oos_net"]) > (
                            min(best["is_wr"], best["oos_wr"]), best["full_tpd"], best["oos_net"]
                        ):
                            best = row
                            print(
                                f"BEST min={mn:.1%} IS={row['is_wr']:.1%} OOS={row['oos_wr']:.1%} "
                                f"tpd={row['full_tpd']:.2f} gap={gap:.1%} net=${row['full_net']:.0f} | "
                                f"{kind} {tf} tp{tp}/sl{sl} h{mh} lot{lot} {side} {mode} inv={invert}",
                                flush=True,
                            )
                            OUT.joinpath("HF80_1TPD_FAST_BEST.json").write_text(
                                json.dumps({"best": best, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2, default=str)
                            )
                        if mn >= 0.70:
                            board70.append(row)
                        if mn >= 0.80 and gap <= 0.05 and row["avg_win"] >= MIN_NET_TP * 0.75:
                            hits80.append(row)
                            print(f"HIT80+1TPD {row}", flush=True)
                            OUT.joinpath("HF80_1TPD_CHAMPION.json").write_text(
                                json.dumps(
                                    {"rule": "≥80% IS+OOS, tpd≥1, TP≥$2, SL≤2×TP", "champion": row,
                                     "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                                    indent=2, default=str,
                                )
                            )
                if tested % 3000 == 0 and tested:
                    print(f"… tested={tested} best_min="
                          f"{min(best['is_wr'], best['oos_wr']):.1%} tpd={best['full_tpd']:.2f}" if best else f"… {tested}",
                          flush=True)

    board70 = sorted(board70, key=lambda r: (min(r["is_wr"], r["oos_wr"]), r["full_tpd"]), reverse=True)
    # dedupe names roughly
    seen = set()
    top = []
    for r in board70:
        key = (r["kind"], r["tf"], r["tp"], r["sl"], r["hold"], r["lot"], r["side"], r["mode"], r["invert"])
        if key in seen:
            continue
        seen.add(key)
        top.append(r)
        if len(top) >= 25:
            break

    OUT.joinpath("HF80_1TPD_FAST_SUMMARY.json").write_text(
        json.dumps(
            {
                "tested": tested,
                "hits80": len(hits80),
                "best": best,
                "top70": top,
                "elapsed_s": time.time() - t0,
                "verdict": (
                    f"FOUND 80%+1tpd"
                    if hits80
                    else (
                        f"NO 80% at tpd≥1. Best: IS {best['is_wr']:.1%} OOS {best['oos_wr']:.1%} "
                        f"tpd={best['full_tpd']:.2f}"
                        if best
                        else "none"
                    )
                ),
            },
            indent=2,
            default=str,
        )
    )
    print("DONE", tested, "hits80", len(hits80), flush=True)
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
