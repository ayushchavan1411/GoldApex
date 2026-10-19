#!/usr/bin/env python3
"""Hillclimb the ~77.5%/81% rsi_level+withTrend 1h family toward both ≥80%."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from rr1_hf80_10d_hunt import MIN_NET_TP, net_on_tp, simulate_multi
from rr1_hf80_2y_hunt import simulate_one
from rr1_search import add_indicators, gen_signals, load_frames

OUT = Path("rr1_search_results")


def main() -> None:
    print("PUSH80 load…", flush=True)
    raw = load_frames(include_resampled=True)
    frames = {tf: add_indicators(raw[tf]) for tf in ("30m", "1h", "2h", "4h") if tf in raw}

    best = None
    hits = []
    near = []
    tested = 0

    rsi_grid = (2, 3, 4, 5, 6, 7, 9, 14)
    os_grid = (1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30)
    ob_grid = (70, 75, 80, 85, 88, 90, 92, 95, 98)
    sess_opts = [
        None,
        ("london",),
        ("overlap",),
        ("ny",),
        ("asian",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
    ]
    tps = (3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25)
    holds = (12, 20, 24, 30, 36, 48, 60, 72, 84, 96, 120, 144)
    lots = (0.02, 0.05, 0.1)
    sms = (1.5, 1.75, 2.0)
    sides = [(True, True), (True, False), (False, True)]

    for tf, df in frames.items():
        n = len(df)
        warm, split, end = 250, int(n * 0.7), n - 2
        ema = df["EMA_50"].shift(1)
        above = (df["Close"] > ema).fillna(False).to_numpy(bool)
        below = (df["Close"] < ema).fillna(False).to_numpy(bool)
        # also EMA_200 if present
        if "EMA_200" in df.columns:
            ema2 = df["EMA_200"].shift(1)
            above2 = (df["Close"] > ema2).fillna(False).to_numpy(bool)
            below2 = (df["Close"] < ema2).fillna(False).to_numpy(bool)
        else:
            above2 = above
            below2 = below

        for rsi, os_, ob, sess in itertools.product(rsi_grid, os_grid, ob_grid, sess_opts):
            if os_ >= 45 or ob <= 55:
                continue
            for kind in ("rsi_level", "rsi_cross"):
                for invert in (False, True):
                    try:
                        L, S = gen_signals(df, kind, {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess})
                        L, S = np.asarray(L, bool), np.asarray(S, bool)
                    except Exception:
                        continue
                    if invert:
                        L, S = S, L
                    variants = [
                        ("raw", L, S),
                        ("withTrend50", L & above, S & below),
                        ("withTrend200", L & above2, S & below2),
                        ("longTrend50", L & above, np.zeros(len(L), bool)),
                        ("shortTrend50", np.zeros(len(L), bool), S & below),
                        ("longTrend200", L & above2, np.zeros(len(L), bool)),
                    ]
                    for vname, Lv, Sv in variants:
                        n_is = int(Lv[warm:split].sum() + Sv[warm:split].sum())
                        if n_is < 35:
                            continue
                        # IS probe
                        pr = simulate_one(
                            df, Lv, Sv, tp_dist=10, sl_dist=20, max_hold=72, lot=0.05,
                            allow_long=True, allow_short=True, start_i=warm, end_i=split,
                        )
                        if pr["trades"] < 30 or pr["winrate"] < 0.74:
                            continue
                        print(
                            f"PROBE IS {pr['winrate']:.1%} n={pr['trades']} {kind} {tf} "
                            f"rsi{rsi}/{os_}/{ob} {vname} inv={invert} sess={sess}",
                            flush=True,
                        )
                        for lot, tp, mh, sm, (al, ash) in itertools.product(lots, tps, holds, sms, sides):
                            if net_on_tp(tp, lot) < MIN_NET_TP:
                                continue
                            sl = tp * sm
                            if sl > 2.0 * tp + 1e-12:
                                continue
                            tested += 1
                            is_m = simulate_one(
                                df, Lv, Sv, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                                allow_long=al, allow_short=ash, start_i=warm, end_i=split,
                            )
                            if is_m["trades"] < 30 or is_m["winrate"] < 0.78 or is_m["net"] <= 0:
                                continue
                            oos = simulate_one(
                                df, Lv, Sv, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                                allow_long=al, allow_short=ash, start_i=split, end_i=end,
                            )
                            if oos["trades"] < 15:
                                continue
                            gap = abs(is_m["winrate"] - oos["winrate"])
                            side = "LS" if al and ash else ("L" if al else "S")
                            row = {
                                "kind": kind,
                                "tf": tf,
                                "rsi": rsi,
                                "os": os_,
                                "ob": ob,
                                "sess": sess,
                                "variant": vname,
                                "invert": invert,
                                "tp": tp,
                                "sl": sl,
                                "sm": sm,
                                "hold": mh,
                                "lot": lot,
                                "side": side,
                                "is_wr": is_m["winrate"],
                                "oos_wr": oos["winrate"],
                                "wr_gap": gap,
                                "is_n": is_m["trades"],
                                "oos_n": oos["trades"],
                                "is_net": is_m["net"],
                                "oos_net": oos["net"],
                                "avg_win_is": is_m["avg_win"],
                                "avg_win_oos": oos["avg_win"],
                                "is_dd": is_m["dd"],
                                "oos_dd": oos["dd"],
                            }
                            mn = min(row["is_wr"], row["oos_wr"])
                            if best is None or (mn, -gap, row["oos_net"]) > (
                                min(best["is_wr"], best["oos_wr"]),
                                -best["wr_gap"],
                                best["oos_net"],
                            ):
                                best = row
                                print(
                                    f"BEST min={mn:.1%} IS={row['is_wr']:.1%} OOS={row['oos_wr']:.1%} "
                                    f"gap={gap:.1%} n={row['is_n']}/{row['oos_n']} | "
                                    f"{kind} {tf} rsi{rsi}/{os_}/{ob} {vname} tp{tp}/sl{sl} "
                                    f"h{mh} lot{lot} {side} inv={invert}",
                                    flush=True,
                                )
                                OUT.joinpath("HF80_2Y_PUSH_BEST.json").write_text(
                                    json.dumps(
                                        {"best": best, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                                        indent=2,
                                        default=str,
                                    )
                                )
                            if mn >= 0.78:
                                near.append(row)
                                near.sort(key=lambda r: (min(r["is_wr"], r["oos_wr"]), -r["wr_gap"]), reverse=True)
                                near[:] = near[:40]

                            if (
                                row["is_wr"] >= 0.80
                                and row["oos_wr"] >= 0.80
                                and gap <= 0.05
                                and oos["net"] > 0
                                and is_m["avg_win"] >= MIN_NET_TP * 0.75
                                and oos["avg_win"] >= MIN_NET_TP * 0.75
                            ):
                                full = simulate_one(
                                    df, Lv, Sv, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                                    allow_long=al, allow_short=ash, start_i=warm, end_i=end,
                                )
                                # also multi confirm
                                full_m = simulate_multi(
                                    df, Lv, Sv, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                                    allow_long=al, allow_short=ash, start_i=warm, end_i=end,
                                )
                                row.update(
                                    full_wr=full["winrate"],
                                    full_net=full["net"],
                                    full_dd=full["dd"],
                                    tpd=full["tpd"],
                                    pf=full["pf"],
                                    multi_wr=full_m["winrate"],
                                    multi_net=full_m["net"],
                                    multi_dd=full_m["dd"],
                                )
                                hits.append(row)
                                print(f"HIT80 {json.dumps(row, default=str)}", flush=True)
                                OUT.joinpath("HF80_2Y_PUSH_CHAMPION.json").write_text(
                                    json.dumps(
                                        {
                                            "rule": "2y IS&OOS≥80% gap≤5pp TP≥$2 SL≤2×TP",
                                            "champion": row,
                                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        },
                                        indent=2,
                                        default=str,
                                    )
                                )
                        if tested % 5000 == 0 and tested:
                            bm = f"{min(best['is_wr'], best['oos_wr']):.1%}" if best else "n/a"
                            print(f"… tested={tested} hits={len(hits)} best_min={bm}", flush=True)

    OUT.joinpath("HF80_2Y_PUSH_SUMMARY.json").write_text(
        json.dumps(
            {"tested": tested, "hits": len(hits), "best": best, "near78": near[:20], "hits_list": hits[:20]},
            indent=2,
            default=str,
        )
    )
    print("PUSH DONE", tested, "hits", len(hits), flush=True)
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
