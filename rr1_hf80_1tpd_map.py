#!/usr/bin/env python3
"""Map WR ceiling at tpd≥1: probe-only."""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
from rr1_search import load_frames, add_indicators, gen_signals
from rr1_hf80_10d_hunt import local_signals, simulate_multi, net_on_tp, MIN_NET_TP
from rr1_hf80_2y_hunt import simulate_one

def main():
    print("MAP load", flush=True)
    raw = load_frames(include_resampled=True)
    frames = {tf: add_indicators(raw[tf]) for tf in ("15m", "30m", "5m") if tf in raw}
    probe_tp = 2.0
    while net_on_tp(probe_tp, 0.05) < MIN_NET_TP:
        probe_tp += 0.3
    print("probe_tp", probe_tp, flush=True)
    board = []
    sess_opts = [None, ("london",), ("ny",), ("overlap",), ("asian",), ("london", "overlap"), ("overlap", "ny"), ("london", "overlap", "ny")]
    for tf in ("15m", "30m", "5m"):
        if tf not in frames:
            continue
        df = frames[tf]
        n = len(df)
        warm = 400 if tf == "5m" else 250
        split = int(n * 0.7)
        end = n - 2
        print("TF", tf, flush=True)
        kinds = []
        for sess, rsi, os, ob in itertools.product(sess_opts, (2, 3, 5, 7, 14), (15, 20, 25, 30), (70, 75, 80, 85)):
            kinds.append(("rsi_cross", {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}))
            kinds.append(("rsi_level", {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}))
            kinds.append(("rsi_recover", {"rsi": rsi, "os": os, "ob": ob}))
        for sess, (per, k) in itertools.product(sess_opts, [(10, 1.5), (20, 2.0), (20, 2.5)]):
            kinds.append(("bb_fade", {"period": per, "k": k, "sessions": sess}))
            kinds.append(("bb_rsi_fade", {"period": per, "k": k, "rsi": 5, "os": 25, "ob": 75, "sessions": sess}))
        for sess, per, os, ob in itertools.product(sess_opts[:4], (5, 14), (20, 25), (75, 80)):
            kinds.append(("stoch_cross", {"period": per, "os": os, "ob": ob, "sessions": sess}))
        for z, th in itertools.product((20, 40), (1.5, 2.0)):
            kinds.append(("zscore_fade", {"period": z, "th": th}))
        kinds.append(("micro_bounce", {}))
        for wick in (0.55, 0.65):
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
                if nsig < 500 or nsig > 18000:
                    continue
                # one-pos only for speed on map
                is_m = simulate_one(
                    df, L, S, tp_dist=probe_tp, sl_dist=2 * probe_tp, max_hold=20, lot=0.05,
                    allow_long=True, allow_short=True, start_i=warm, end_i=split,
                )
                if is_m["tpd"] < 1.0 or is_m["trades"] < 200:
                    is_m = simulate_multi(
                        df, L, S, tp_dist=probe_tp, sl_dist=2 * probe_tp, max_hold=20, lot=0.05,
                        allow_long=True, allow_short=True, start_i=warm, end_i=split,
                    )
                    mode = "multi"
                else:
                    mode = "one"
                if is_m["tpd"] < 1.0 or is_m["trades"] < 200:
                    continue
                sim = simulate_one if mode == "one" else simulate_multi
                oos = sim(
                    df, L, S, tp_dist=probe_tp, sl_dist=2 * probe_tp, max_hold=20, lot=0.05,
                    allow_long=True, allow_short=True, start_i=split, end_i=end,
                )
                if oos["tpd"] < 0.85 or oos["trades"] < 80:
                    continue
                full = sim(
                    df, L, S, tp_dist=probe_tp, sl_dist=2 * probe_tp, max_hold=20, lot=0.05,
                    allow_long=True, allow_short=True, start_i=warm, end_i=end,
                )
                if full["tpd"] < 1.0:
                    continue
                gap = abs(is_m["winrate"] - oos["winrate"])
                row = dict(
                    kind=kind, tf=tf, invert=invert, mode=mode, params=base,
                    is_wr=is_m["winrate"], oos_wr=oos["winrate"], full_wr=full["winrate"], wr_gap=gap,
                    is_tpd=is_m["tpd"], oos_tpd=oos["tpd"], full_tpd=full["tpd"],
                    is_n=is_m["trades"], oos_n=oos["trades"], full_n=full["trades"],
                    is_net=is_m["net"], oos_net=oos["net"], full_net=full["net"], full_dd=full["dd"],
                )
                board.append(row)
                mn = min(row["is_wr"], row["oos_wr"])
                if mn >= 0.65:
                    print(
                        f"MAP {mn:.1%} IS={row['is_wr']:.1%} OOS={row['oos_wr']:.1%} "
                        f"tpd={row['full_tpd']:.2f} gap={gap:.1%} {mode} | {kind} {tf} inv={invert}",
                        flush=True,
                    )

    board.sort(key=lambda r: (min(r["is_wr"], r["oos_wr"]), r["full_tpd"], -r["wr_gap"]), reverse=True)
    seen = set()
    top = []
    for r in board:
        k = (r["kind"], r["tf"], r["invert"], r["mode"], str(r["params"]))
        if k in seen:
            continue
        seen.add(k)
        top.append(r)
        if len(top) >= 30:
            break
    best = top[0] if top else None
    hits = [r for r in top if min(r["is_wr"], r["oos_wr"]) >= 0.80 and r["wr_gap"] <= 0.05]
    Path("rr1_search_results/HF80_1TPD_MAP.json").write_text(
        json.dumps(
            {
                "probe": "tp$2+/sl=2x/h20/lot0.05",
                "n_mapped": len(board),
                "best": best,
                "top30": top,
                "n_ge80_gap5": len(hits),
                "hits80": hits[:10],
                "n_ge75": sum(1 for r in top if min(r["is_wr"], r["oos_wr"]) >= 0.75),
                "n_ge70": sum(1 for r in top if min(r["is_wr"], r["oos_wr"]) >= 0.70),
                "verdict": (
                    "PROBE FOUND 80%+1tpd"
                    if hits
                    else (
                        f"PROBE ceiling at tpd≥1: IS {best['is_wr']:.1%} OOS {best['oos_wr']:.1%} tpd={best['full_tpd']:.2f}"
                        if best
                        else "none"
                    )
                ),
            },
            indent=2,
            default=str,
        )
    )
    print("DONE mapped", len(board), flush=True)
    if best:
        print(
            f"CEILING min={min(best['is_wr'], best['oos_wr']):.1%} "
            f"IS={best['is_wr']:.1%} OOS={best['oos_wr']:.1%} tpd={best['full_tpd']:.2f}",
            flush=True,
        )

if __name__ == "__main__":
    main()
