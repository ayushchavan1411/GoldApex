#!/usr/bin/env python3
"""2y hunt: ≥80% WR IS+OOS AND ≥1 trade/day (user hard ask).

Same cost rules: TP nets ≥$2 after spread/commission/slip, SL ≤ 2× TP,
unlimited concurrent positions preferred for frequency.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np

from rr1_hf80_10d_hunt import (
    LOCAL_KINDS,
    MIN_NET_TP,
    local_signals,
    min_tp_dist,
    net_on_tp,
    simulate_multi,
)
from rr1_hf80_2y_hunt import simulate_one
from rr1_search import add_indicators, gen_signals, load_frames, log

RNG = random.Random(1911)
OUT = Path("rr1_search_results")
LEADER = OUT / "HF80_1TPD_LEADERBOARD.jsonl"
SUMMARY = OUT / "HF80_1TPD_SUMMARY.json"
CHAMP = OUT / "HF80_1TPD_CHAMPION.json"
CEILING = OUT / "HF80_1TPD_CEILING.json"

TARGET_WR = 0.80
MAX_GAP = 0.05
MIN_TPD = 1.0


def specs() -> list[tuple[str, dict]]:
    """Denser (higher-fire) mean-reversion grid for tpd≥1."""
    out: list[tuple[str, dict]] = []
    sess_opts = [
        None,
        ("london",),
        ("ny",),
        ("overlap",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
        ("asian",),
    ]
    # Milder RSI extremes = more trades
    for sess, rsi, os, ob in itertools.product(
        sess_opts, (2, 3, 5, 7, 14), (15, 20, 25, 30, 35), (65, 70, 75, 80, 85)
    ):
        base = {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}
        out.append(("rsi_cross", dict(base)))
        out.append(("rsi_level", dict(base)))
        out.append(("rsi_recover", {k: v for k, v in base.items() if k != "sessions"}))
        out.append(("rsi_stay", {k: v for k, v in base.items() if k != "sessions"}))
    for sess, (per, k) in itertools.product(
        sess_opts, [(10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5)]
    ):
        out.append(("bb_fade", {"period": per, "k": k, "sessions": sess}))
        out.append(
            ("bb_rsi_fade", {"period": per, "k": k, "rsi": 5, "os": 25, "ob": 75, "sessions": sess})
        )
        out.append(
            ("bb_rsi_fade", {"period": per, "k": k, "rsi": 7, "os": 30, "ob": 70, "sessions": sess})
        )
    for sess, per, os, ob in itertools.product(sess_opts[:5], (5, 9, 14), (20, 25, 30), (70, 75, 80)):
        out.append(("stoch_cross", {"period": per, "os": os, "ob": ob, "sessions": sess}))
    for z, th in itertools.product((20, 40), (1.0, 1.5, 2.0)):
        out.append(("zscore_fade", {"period": z, "th": th}))
    for per, th in itertools.product((14, 20), (100, 150, 200)):
        out.append(("cci_fade", {"period": per, "th": th}))
    for rsi, os, ob in itertools.product((3, 5, 7), (20, 25, 30), (70, 75, 80)):
        out.append(("micro_reclaim", {"rsi": rsi, "os": os, "ob": ob}))
        out.append(("range_fade", {"look": 5, "rsi": rsi, "os": os, "ob": ob}))
        out.append(("range_fade", {"look": 8, "rsi": rsi, "os": os, "ob": ob}))
    for wick, body in itertools.product((0.45, 0.55, 0.65), (0.3, 0.4)):
        out.append(("wick_reject", {"wick": wick, "body": body}))
    out.append(("micro_bounce", {}))
    return out


def exits() -> list[dict]:
    out = []
    lots = (0.02, 0.05, 0.1)
    # Smaller TPs → faster fills → higher tpd, still ≥$2 net
    tps = (0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)
    sl_mults = (1.5, 1.75, 2.0)
    holds = (5, 8, 12, 16, 20, 30, 48)
    sides = (
        dict(allow_long=True, allow_short=True),
        dict(allow_long=True, allow_short=False),
        dict(allow_long=False, allow_short=True),
    )
    for lot, tp, sm, mh, side in itertools.product(lots, tps, sl_mults, holds, sides):
        if net_on_tp(tp, lot) < MIN_NET_TP:
            continue
        out.append(
            dict(
                tp_dist=tp,
                sl_dist=tp * sm,
                max_hold=mh,
                lot=lot,
                sl_mult=sm,
                net_tp=net_on_tp(tp, lot),
                **side,
            )
        )
    return out


def make_signals(df, kind, pp):
    base = {k: v for k, v in pp.items() if k != "invert"}
    if kind in LOCAL_KINDS:
        return local_signals(df, kind, base)
    L, S = gen_signals(df, kind, base)
    return np.asarray(L, bool), np.asarray(S, bool)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    LEADER.write_text("")
    for lot in (0.02, 0.05, 0.1):
        log(f"lot={lot}: min TP$={min_tp_dist(lot):.3f}")

    raw = load_frames(include_resampled=True)
    frames = {}
    # Frequency lives on lower TFs
    for tf in ("5m", "15m", "30m"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])

    windows = {}
    for tf, df in frames.items():
        n = len(df)
        warm = 400 if tf == "5m" else 250
        split = int(n * 0.70)
        windows[tf] = (warm, split, n - 2)
        log(f"{tf}: bars={n} split={df.index[split]}")

    sp = specs()
    exs = exits()
    log(f"specs={len(sp)} exits={len(exs)}")

    cache = []
    for kind, base in sp:
        for invert in (False, True):
            pp = {**base, **({"invert": True} if invert else {})}
            for tf, df in frames.items():
                warm, split, end = windows[tf]
                try:
                    L, S = make_signals(df, kind, pp)
                except Exception:
                    continue
                if invert:
                    L, S = S, L
                # Need enough IS signals to support ≥1 tpd over ~IS days (~500+)
                nsig = int(L[warm:end].sum() + S[warm:end].sum())
                # Need density for ≥1 tpd, but skip every-bar spam (WR collapses + too slow)
                if nsig < 500:
                    continue
                if nsig > 25000 and tf == "5m":
                    continue
                if kind == "rsi_stay" and nsig > 15000:
                    continue
                cache.append((kind, pp, tf, L, S, invert, nsig))
    RNG.shuffle(cache)
    # Prefer mid-density (enough for 1tpd, not noise)
    cache.sort(key=lambda x: abs(x[6] - 3000))
    log(f"signal cache={len(cache)}")

    tested = hits = 0
    best80 = None
    ceiling = {"best_min_wr": None, "best_tpd_at_75": None, "ge70": [], "ge75": []}
    t0 = time.time()

    def consider(row):
        mn = min(row["is_wr"], row["oos_wr"])
        prev = ceiling["best_min_wr"]
        if prev is None or (mn, row["full_tpd"], row["oos_net"]) > (
            min(prev["is_wr"], prev["oos_wr"]),
            prev["full_tpd"],
            prev["oos_net"],
        ):
            ceiling["best_min_wr"] = row
        if mn >= 0.75 and row["full_tpd"] >= MIN_TPD:
            prev = ceiling["best_tpd_at_75"]
            if prev is None or (mn, row["full_tpd"]) > (min(prev["is_wr"], prev["oos_wr"]), prev["full_tpd"]):
                ceiling["best_tpd_at_75"] = row
        if mn >= 0.70 and row["full_tpd"] >= MIN_TPD:
            ceiling["ge70"].append(row)
            ceiling["ge70"] = sorted(
                ceiling["ge70"],
                key=lambda r: (min(r["is_wr"], r["oos_wr"]), r["full_tpd"], r["oos_net"]),
                reverse=True,
            )[:40]
        if mn >= 0.75 and row["full_tpd"] >= MIN_TPD:
            ceiling["ge75"].append(row)
            ceiling["ge75"] = sorted(
                ceiling["ge75"],
                key=lambda r: (min(r["is_wr"], r["oos_wr"]), r["full_tpd"], r["oos_net"]),
                reverse=True,
            )[:40]

    # Probe: need FULL tpd roughly possible — use multi, SL=2x, dual side
    for idx, (kind, pp, tf, L, S, invert, nsig) in enumerate(cache):
        df = frames[tf]
        warm, split, end = windows[tf]
        p_tp = max(min_tp_dist(0.05), 1.3)
        probe = simulate_multi(
            df, L, S,
            tp_dist=p_tp, sl_dist=2 * p_tp, max_hold=20, lot=0.05,
            allow_long=True, allow_short=True, start_i=warm, end_i=end,
        )
        if probe["trades"] < 400 or probe["tpd"] < 0.85:
            continue
        # Need a WR floor on probe or expand is wasted — 80% at 1tpd is hard
        if probe["winrate"] < 0.62:
            continue
        log(
            f"PROBE tpd={probe['tpd']:.2f} WR={probe['winrate']:.1%} n={probe['trades']} "
            f"| {kind} {tf} inv={invert} nsig={nsig}"
        )

        # Expand: multi first (frequency); one-pos only if probe already strong
        prefer = [e for e in exs if e["sl_mult"] >= 1.75]
        sample = RNG.sample(prefer, min(len(prefer), 100))
        modes = (("multi", simulate_multi),)
        if probe["winrate"] >= 0.70:
            modes = (("multi", simulate_multi), ("one", simulate_one))
        for ex in sample:
            for mode, sim in modes:
                tested += 1
                full = sim(
                    df, L, S,
                    tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                    lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                    start_i=warm, end_i=end,
                )
                if full["tpd"] < MIN_TPD or full["trades"] < 400 or full["net"] <= 0:
                    continue
                is_m = sim(
                    df, L, S,
                    tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                    lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                    start_i=warm, end_i=split,
                )
                if is_m["trades"] < 200 or is_m["net"] <= 0:
                    continue
                # Soft WR floor before OOS
                if is_m["winrate"] < 0.70:
                    continue
                oos = sim(
                    df, L, S,
                    tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                    lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                    start_i=split, end_i=end,
                )
                if oos["trades"] < 80 or oos["net"] <= 0:
                    continue
                # OOS tpd also roughly ≥0.85
                if oos["tpd"] < 0.85 or is_m["tpd"] < 0.85:
                    continue
                gap = abs(is_m["winrate"] - oos["winrate"])
                side = (
                    "LS"
                    if ex["allow_long"] and ex["allow_short"]
                    else ("L" if ex["allow_long"] else "S")
                )
                name = (
                    f"{'inv|' if invert else ''}{kind}|tp{ex['tp_dist']:g}/sl{ex['sl_dist']:g}|"
                    f"h{ex['max_hold']}|{tf}|{side}|lot{ex['lot']:g}|{mode}"
                )
                row = {
                    "name": name,
                    "kind": kind,
                    "params": pp,
                    "exit": ex,
                    "invert": invert,
                    "mode": mode,
                    "tf": tf,
                    "is_wr": is_m["winrate"],
                    "oos_wr": oos["winrate"],
                    "full_wr": full["winrate"],
                    "wr_gap": gap,
                    "is_n": is_m["trades"],
                    "oos_n": oos["trades"],
                    "full_n": full["trades"],
                    "is_tpd": is_m["tpd"],
                    "oos_tpd": oos["tpd"],
                    "full_tpd": full["tpd"],
                    "is_net": is_m["net"],
                    "oos_net": oos["net"],
                    "full_net": full["net"],
                    "is_dd": is_m["dd"],
                    "oos_dd": oos["dd"],
                    "full_dd": full["dd"],
                    "pf": full["pf"],
                    "avg_win": full["avg_win"],
                    "max_open": full.get("max_concurrent", 1),
                    "net_tp_design": ex["net_tp"],
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                consider(row)

                if (
                    min(row["is_wr"], row["oos_wr"]) >= TARGET_WR
                    and gap <= MAX_GAP
                    and row["full_tpd"] >= MIN_TPD
                    and is_m["avg_win"] >= MIN_NET_TP * 0.75
                    and oos["avg_win"] >= MIN_NET_TP * 0.75
                ):
                    with open(LEADER, "a") as f:
                        f.write(json.dumps(row, default=str) + "\n")
                    hits += 1
                    sk = (
                        min(row["is_wr"], row["oos_wr"]),
                        row["full_tpd"],
                        -gap,
                        row["oos_net"],
                        -row["full_dd"],
                    )
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
                        f"tpd={full['tpd']:.2f} gap={gap:.1%} net=${full['net']:.0f} | {name}"
                    )

        if (idx + 1) % 25 == 0:
            b = ceiling["best_min_wr"]
            msg = (
                f"ceil minWR={min(b['is_wr'], b['oos_wr']):.1%} tpd={b['full_tpd']:.2f} "
                f"IS={b['is_wr']:.1%} OOS={b['oos_wr']:.1%}"
                if b
                else "ceil n/a"
            )
            log(f"progress {idx+1}/{len(cache)} tested={tested} hits={hits} {msg}")
            CEILING.write_text(
                json.dumps(
                    {
                        "tested": tested,
                        "hits_80_1tpd": hits,
                        "best_min_wr_at_tpd1": ceiling["best_min_wr"],
                        "best_75_plus": ceiling["best_tpd_at_75"],
                        "n_ge70_tpd1": len(ceiling["ge70"]),
                        "n_ge75_tpd1": len(ceiling["ge75"]),
                        "ge75_top": ceiling["ge75"][:15],
                        "ge70_top": ceiling["ge70"][:15],
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
                "hits_80_1tpd": hits,
                "best80": best80,
                "ceiling": {
                    "best_min_wr_at_tpd1": ceiling["best_min_wr"],
                    "best_75_plus": ceiling["best_tpd_at_75"],
                    "ge75_top": ceiling["ge75"][:20],
                    "ge70_top": ceiling["ge70"][:20],
                },
                "gate": "IS&OOS≥80% gap≤5pp full_tpd≥1",
                "elapsed_s": time.time() - t0,
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
                "best_min_wr_at_tpd1": ceiling["best_min_wr"],
                "best_75_plus": ceiling["best_tpd_at_75"],
                "n_ge70_tpd1": len(ceiling["ge70"]),
                "n_ge75_tpd1": len(ceiling["ge75"]),
                "ge75_top": ceiling["ge75"][:20],
                "ge70_top": ceiling["ge70"][:20],
                "elapsed_s": time.time() - t0,
                "verdict": (
                    f"FOUND 80%+1tpd: {best80['name']}"
                    if best80
                    else (
                        f"NO 80% at tpd≥1. Best stable: "
                        f"IS {ceiling['best_min_wr']['is_wr']:.1%} / "
                        f"OOS {ceiling['best_min_wr']['oos_wr']:.1%} "
                        f"tpd={ceiling['best_min_wr']['full_tpd']:.2f}"
                        if ceiling["best_min_wr"]
                        else "NO tpd≥1 profitable candidates"
                    )
                ),
            },
            indent=2,
            default=str,
        )
    )
    log(f"DONE hits={hits}")
    if best80:
        log(f"CHAMP {best80['name']}")
    elif ceiling["best_min_wr"]:
        b = ceiling["best_min_wr"]
        log(
            f"CEILING minWR={min(b['is_wr'], b['oos_wr']):.1%} "
            f"IS={b['is_wr']:.1%} OOS={b['oos_wr']:.1%} tpd={b['full_tpd']:.2f} | {b['name']}"
        )


if __name__ == "__main__":
    main()
