#!/usr/bin/env python3
"""Ultra-focused 2y 80% hunt: extreme RSI + trend filters, SL=2×TP."""
from __future__ import annotations
import itertools, json, time
from pathlib import Path
import numpy as np
from rr1_search import load_frames, add_indicators, gen_signals
from rr1_hf80_10d_hunt import simulate_multi, net_on_tp, min_tp_dist, MIN_NET_TP
from rr1_hf80_2y_hunt import simulate_one

OUT = Path("rr1_search_results")

def main():
    print("FOCUS load…", flush=True)
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("15m", "30m", "1h", "2h"):
        frames[tf] = add_indicators(raw[tf])
        print(tf, len(frames[tf]), flush=True)

    hits = []
    best = None
    tested = 0
    lots = (0.02, 0.05, 0.1)
    tps = (1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0)
    holds = (12, 20, 30, 48, 72, 96)
    sides = [(True, True), (True, False), (False, True)]

    for tf, df in frames.items():
        n = len(df)
        warm, split, end = 300, int(n * 0.7), n - 2
        ema = df["EMA_50"].shift(1)
        above = (df["Close"] > ema).fillna(False).to_numpy(bool)
        below = (df["Close"] < ema).fillna(False).to_numpy(bool)
        for rsi, os, ob in itertools.product((3, 5, 7, 14), (5, 8, 10, 12, 15, 20), (80, 85, 88, 90, 92, 95)):
            for kind in ("rsi_cross", "rsi_level"):
                for invert in (False, True):
                    try:
                        L, S = gen_signals(df, kind, {"rsi": rsi, "os": os, "ob": ob})
                        L, S = np.asarray(L, bool), np.asarray(S, bool)
                    except Exception:
                        continue
                    if invert:
                        L, S = S, L
                    variants = [
                        ("raw", L, S),
                        ("longEMA", L & above, np.zeros(len(L), bool)),
                        ("shortEMA", np.zeros(len(L), bool), S & below),
                        ("withTrend", L & above, S & below),
                        ("counterEMA", L & below, S & above),
                    ]
                    for vname, Lv, Sv in variants:
                        if int(Lv[warm:end].sum() + Sv[warm:end].sum()) < 60:
                            continue
                        ptp = max(min_tp_dist(0.05), 1.0)
                        for sim, mode in ((simulate_multi, "multi"), (simulate_one, "one")):
                            pr = sim(
                                df, Lv, Sv, tp_dist=ptp, sl_dist=2 * ptp, max_hold=30, lot=0.05,
                                allow_long=True, allow_short=True, start_i=warm, end_i=end,
                            )
                            if pr["trades"] < 50 or pr["winrate"] < 0.72:
                                continue
                            print(
                                f"PROBE {pr['winrate']:.1%} {mode} {kind} {tf} rsi{rsi}/{os}/{ob} "
                                f"{vname} inv={invert}",
                                flush=True,
                            )
                            for lot, tp, mh, (al, ash) in itertools.product(lots, tps, holds, sides):
                                if net_on_tp(tp, lot) < MIN_NET_TP:
                                    continue
                                sl = 2.0 * tp
                                tested += 1
                                is_m = sim(
                                    df, Lv, Sv, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                                    allow_long=al, allow_short=ash, start_i=warm, end_i=split,
                                )
                                if is_m["trades"] < 80 or is_m["winrate"] < 0.80 or is_m["net"] <= 0:
                                    continue
                                oos = sim(
                                    df, Lv, Sv, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                                    allow_long=al, allow_short=ash, start_i=split, end_i=end,
                                )
                                if oos["trades"] < 35 or oos["winrate"] < 0.80 or oos["net"] <= 0:
                                    continue
                                gap = abs(is_m["winrate"] - oos["winrate"])
                                if gap > 0.05:
                                    continue
                                if is_m["avg_win"] < MIN_NET_TP * 0.75:
                                    continue
                                full = sim(
                                    df, Lv, Sv, tp_dist=tp, sl_dist=sl, max_hold=mh, lot=lot,
                                    allow_long=al, allow_short=ash, start_i=warm, end_i=end,
                                )
                                side = "LS" if al and ash else ("L" if al else "S")
                                row = dict(
                                    kind=kind, tf=tf, rsi=rsi, os=os, ob=ob, variant=vname,
                                    invert=invert, mode=mode, tp=tp, sl=sl, hold=mh, lot=lot, side=side,
                                    is_wr=is_m["winrate"], oos_wr=oos["winrate"], full_wr=full["winrate"],
                                    wr_gap=gap, is_n=is_m["trades"], oos_n=oos["trades"],
                                    is_net=is_m["net"], oos_net=oos["net"], full_net=full["net"],
                                    full_dd=full["dd"], tpd=full["tpd"], pf=full["pf"],
                                    avg_win=full["avg_win"], max_open=full.get("max_concurrent", 1),
                                )
                                hits.append(row)
                                print(
                                    f"HIT80 IS {is_m['winrate']:.1%} OOS {oos['winrate']:.1%} "
                                    f"gap={gap:.1%} net=${full['net']:.0f} dd={full['dd']:.1%} | "
                                    f"{kind} {tf} {vname} {mode} tp{tp}/sl{sl} lot{lot} {side}",
                                    flush=True,
                                )
                                sk = (min(row["is_wr"], row["oos_wr"]), -gap, row["oos_net"])
                                if best is None or sk > (
                                    min(best["is_wr"], best["oos_wr"]), -best["wr_gap"], best["oos_net"]
                                ):
                                    best = row
                                    OUT.joinpath("HF80_2Y_FOCUS_CHAMPION.json").write_text(
                                        json.dumps(
                                            {"champion": best, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                                            indent=2,
                                        )
                                    )
                            if tested % 2000 == 0:
                                print(f"… tested={tested} hits={len(hits)}", flush=True)

    print("FOCUS DONE tested", tested, "hits", len(hits), flush=True)
    top = sorted(
        hits,
        key=lambda r: (min(r["is_wr"], r["oos_wr"]), -r["wr_gap"], r["oos_net"]),
        reverse=True,
    )[:20]
    OUT.joinpath("HF80_2Y_FOCUS_SUMMARY.json").write_text(
        json.dumps({"tested": tested, "hits": len(hits), "best": best, "top": top}, indent=2, default=str)
    )
    print("best", best, flush=True)

if __name__ == "__main__":
    main()
