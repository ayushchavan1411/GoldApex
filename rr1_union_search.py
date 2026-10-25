#!/usr/bin/env python3
"""Multi-OR union hunter: combine complementary high-WR selective signals to push tpd→1 while holding WR.

Strategy:
  1) Build a diverse library of selective signals (many families × sessions × hours).
  2) Score singles at strict 1:1 exits; keep those with min(IS,OOS) WR ≥ 60%.
  3) Greedy OR union: repeatedly add the partner that best preserves minWR while lifting tpd.
  4) Random k-way OR (k=2..6) with session/kind diversity.
Never stops until user kills the process.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np

from rr1_search import (
    OUT,
    add_indicators,
    evaluate,
    gen_signals,
    load_frames,
    log,
    qualifies,
    save_champion,
    write_bot,
    year_stable,
)

RNG = random.Random(91)
BOARD = OUT / "UNION_LEADERBOARD.jsonl"
PROG = OUT / "union_progress.json"


def _tuplify(p: dict) -> dict:
    p = dict(p)
    for k in ("sessions", "hours", "dows"):
        if p.get(k) is not None and not isinstance(p[k], tuple):
            p[k] = tuple(p[k])
    return p


def _clean(L, S):
    both = L & S
    return L & ~both, S & ~both


def _append_board(res, tag: str):
    with open(BOARD, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "name": res.name,
                    "wr": res.winrate,
                    "oos_wr": res.oos_winrate,
                    "tpd": res.trades_per_day,
                    "oos_tpd": res.oos_trades_per_day,
                    "n": res.trades,
                    "oos_n": res.oos_trades,
                    "pf": res.profit_factor,
                    "oos_pf": res.oos_profit_factor,
                    "pnl": res.net_pnl,
                    "oos_pnl": res.oos_net_pnl,
                    "score": res.score,
                    "params": res.params,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            + "\n"
        )


def _jitter_params(kind: str, p: dict) -> dict:
    """Randomize numeric thresholds so later passes explore new ground."""
    p = dict(p)
    def jinti(key, lo, hi, step=1):
        if key in p and isinstance(p[key], (int, float)):
            p[key] = int(max(lo, min(hi, int(p[key]) + RNG.choice([-2, -1, 0, 0, 1, 2]) * step)))
    def jfloat(key, lo, hi, step=0.05):
        if key in p and isinstance(p[key], (int, float)):
            p[key] = float(max(lo, min(hi, float(p[key]) + RNG.choice([-2, -1, 0, 0, 1, 2]) * step)))
    if kind in ("rsi_macd", "rsi_level", "fvg_rsi", "ema_pull_rsi", "bb_rsi_fade"):
        jinti("os", 15, 55)
        jinti("ob", 50, 90)
        jinti("band", 2, 12)
        jinti("rsi", 3, 21)
    if kind in ("liq_sweep", "liq_sweep_fvg"):
        jfloat("wick", 0.25, 0.7, 0.05)
        jinti("swing", 10, 60, 5)
    if kind in ("keltner_fade", "supertrend_flip"):
        jfloat("mult", 1.0, 4.0, 0.25)
    if kind in ("zscore_fade", "cci_fade"):
        jfloat("th", 1.0, 3.5, 0.25)
    if kind == "turtle_break":
        jinti("n", 5, 40, 5)
    if kind == "ob_retest":
        jfloat("atr_mult", 0.1, 0.8, 0.05)
    if kind in ("fib_bounce", "fib_discount", "bos_pullback", "smc_confluence"):
        jinti("swing", 10, 80, 5)
    if RNG.random() < 0.35:
        # random hour slice
        a = RNG.randint(0, 18)
        b = min(24, a + RNG.choice([4, 6, 8, 10, 12]))
        p["hours"] = tuple(range(a, b))
    if RNG.random() < 0.2:
        p["dows"] = tuple(sorted(RNG.sample(range(5), k=RNG.randint(2, 5))))  # Mon-Fri subset
    return p


def build_library(df, mutate: bool = False):
    """Diverse selective signal library across families."""
    sess_opts = [
        ("asian",),
        ("london",),
        ("overlap",),
        ("ny",),
        ("asian", "london"),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("asian", "london", "overlap"),
        ("london", "overlap", "ny"),
        None,
    ]
    hour_opts = [
        None,
        tuple(range(0, 8)),
        tuple(range(7, 12)),
        tuple(range(8, 17)),
        tuple(range(12, 20)),
        tuple(range(0, 12)),
        tuple(range(12, 24)),
    ]
    specs = []
    # RSI+MACD family (known ~71% peak)
    for rsi, os_, ob, sess, hours in itertools.product(
        [7, 14], [35, 40, 45, 50], [55, 60, 65, 70], sess_opts, hour_opts
    ):
        p = {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
        if hours:
            p["hours"] = hours
        specs.append(("rsi_macd", p))
    # SMC / ICT
    for kind, extra in [
        ("fvg_reclaim", {}),
        ("fvg_rsi", {"rsi": 7, "os": 35, "ob": 65}),
        ("liq_sweep", {"swing": 20, "wick": 0.45}),
        ("liq_sweep_fvg", {"swing": 20, "wick": 0.4}),
        ("ob_retest", {"atr_mult": 0.3}),
        ("bos_pullback", {"swing": 20}),
        ("fib_bounce", {"swing": 40, "level": "FIB618"}),
        ("fib_discount", {"swing": 40}),
        ("smc_confluence", {"swing": 20}),
    ]:
        for sess in sess_opts:
            p = {**extra, "sessions": sess}
            specs.append((kind, p))
    # Public playbooks
    for kind, grid in [
        ("ichimoku_tk", [{}]),
        ("ichimoku_kumo", [{}]),
        ("heikin_ema", [{"ema": e} for e in (21, 50, 100)]),
        ("keltner_fade", [{"period": 20, "mult": m} for m in (1.5, 2.0, 2.5)]),
        ("zscore_fade", [{"period": p, "th": t} for p in (20, 40) for t in (1.5, 2.0, 2.5)]),
        ("cci_fade", [{"period": 20, "th": t} for t in (100, 150, 200)]),
        ("supertrend_flip", [{"mult": m} for m in (2.0, 3.0)]),
        ("turtle_break", [{"n": n} for n in (10, 20)]),
        ("orb_fade", [{"orb": o} for o in ("LONDON", "NY", "ASIAN")]),
        ("orb_break", [{"orb": o} for o in ("LONDON", "NY")]),
        ("rsi_level", [{"rsi": r, "os": o, "ob": b, "band": 5} for r in (7, 14) for o in (25, 30, 35) for b in (65, 70, 75)]),
        ("bb_rsi_fade", [{"period": 20, "k": 2.0, "rsi": 7, "os": o, "ob": b} for o in (25, 30) for b in (70, 75)]),
        ("ema_pull_rsi", [{"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65}]),
        ("adx_di_cross", [{"th": t} for t in (20, 25)]),
    ]:
        for base in grid:
            for sess in (("asian",), ("london",), ("overlap",), ("ny",), ("london", "overlap"), None):
                specs.append((kind, {**base, "sessions": sess}))

    # Extra mutated copies each pass
    if mutate:
        extras = []
        for kind, p in RNG.sample(specs, min(400, len(specs))):
            extras.append((kind, _jitter_params(kind, p)))
        specs.extend(extras)

    lib = []
    seen = set()
    for kind, p in specs:
        key = (kind, json.dumps(p, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        try:
            L, S = gen_signals(df, kind, p)
            L, S = _clean(L, S)
        except Exception:
            continue
        nsig = int(L.sum() + S.sum())
        # selective but not empty; allow very selective for OR stacking
        if 30 <= nsig <= 8000:
            lib.append((kind, p, L, S, nsig))
    return lib


def score_singles(df, lib, exits):
    singles = []
    for kind, p, L, S, nsig in lib:
        best = None
        for ex in exits:
            res = evaluate(f"u|{kind}", "5m", df, L, S, ex, {"kind": kind, **p})
            if not res:
                continue
            if best is None or min(res.winrate, res.oos_winrate) > min(best.winrate, best.oos_winrate):
                best = res
        if best and min(best.winrate, best.oos_winrate) >= 0.60 and best.trades >= 25:
            singles.append(
                {
                    "res": best,
                    "kind": kind,
                    "p": p,
                    "L": L,
                    "S": S,
                    "sess": p.get("sessions"),
                    "min_wr": min(best.winrate, best.oos_winrate),
                    "tpd": min(best.trades_per_day, best.oos_trades_per_day),
                }
            )
    singles.sort(key=lambda x: (x["min_wr"], x["tpd"]), reverse=True)
    return singles


def consider(res, L, S, ex, tested_box, quals_box, best_box, tag: str, df):
    tested_box[0] += 1
    if not res:
        return
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
    min_wr = min(res.winrate, res.oos_winrate)
    min_tpd = min(res.trades_per_day, res.oos_trades_per_day)
    useful = min_wr >= 0.65 and min_tpd >= 0.4
    near = min_wr >= 0.70
    if q or useful or near:
        _append_board(res, tag)
        key = (min_wr, min_tpd, res.score)
        b = best_box[0]
        bkey = (
            (min(b.winrate, b.oos_winrate), min(b.trades_per_day, b.oos_trades_per_day), b.score)
            if b
            else (-1, -1, -1)
        )
        if q or key > bkey:
            if save_champion(res, force=q):
                write_bot(res)
                best_box[0] = res
                if q:
                    quals_box[0] += 1
                log(
                    f"{'QUAL ' if q else ''}UNION {tag} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                    f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                    f"n={res.trades}/{res.oos_trades}"
                )


def greedy_union(df, singles, exits, tested_box, quals_box, best_box, max_parts=8):
    """Greedy OR: start with best single, add partners that lift tpd with least WR damage."""
    if not singles:
        return
    pool = singles[:80]
    base = pool[0]
    L = base["L"].copy()
    S = base["S"].copy()
    used = {0}
    parts = [base["kind"]]
    for ex in exits[:4]:
        res = evaluate(f"ug|{'+'.join(parts)}", "5m", df, L, S, ex, {"kind": "union", "parts": parts})
        consider(res, L, S, ex, tested_box, quals_box, best_box, "greedy0", df)

    for _ in range(max_parts - 1):
        best_add = None
        best_score = None
        last = pool[list(used)[-1]]
        for i, cand in enumerate(pool):
            if i in used:
                continue
            # skip near-duplicates (same kind + same session as last add)
            if cand["kind"] == last["kind"] and cand["sess"] == last["sess"]:
                continue
            Ln, Sn = _clean(L | cand["L"], S | cand["S"])
            if int(Ln.sum() + Sn.sum()) <= int(L.sum() + S.sum()) + 5:
                continue
            ex = exits[0]
            res = evaluate("ugprobe", "5m", df, Ln, Sn, ex, {"kind": "union"})
            if not res:
                continue
            score = (min(res.winrate, res.oos_winrate), min(res.trades_per_day, res.oos_trades_per_day))
            if best_score is None or score > best_score:
                best_score = score
                best_add = (i, cand, Ln, Sn, res)
        if best_add is None:
            break
        i, cand, L, S, _ = best_add
        used.add(i)
        parts.append(cand["kind"])
        for ex in exits:
            res = evaluate(
                f"ug|{'+'.join(parts)}",
                "5m",
                df,
                L,
                S,
                ex,
                {"kind": "union", "parts": list(parts)},
            )
            consider(res, L, S, ex, tested_box, quals_box, best_box, f"greedy{len(parts)}", df)
        # stop early if already at target tpd with high WR
        if best_score and best_score[0] >= 0.75 and best_score[1] >= 1.0:
            break


def random_unions(df, singles, exits, tested_box, quals_box, best_box, rounds=400):
    pool = singles[:100]
    if len(pool) < 3:
        return
    for _ in range(rounds):
        k = RNG.randint(2, min(6, len(pool)))
        # diversify: try unique kinds when possible
        picks = []
        kinds_used = set()
        shuffled = pool[:]
        RNG.shuffle(shuffled)
        for cand in shuffled:
            if cand["kind"] in kinds_used and len(picks) < k - 1:
                # allow same kind only if session differs and we still need slots
                if cand["sess"] in {p["sess"] for p in picks if p["kind"] == cand["kind"]}:
                    continue
            picks.append(cand)
            kinds_used.add(cand["kind"])
            if len(picks) >= k:
                break
        if len(picks) < 2:
            continue
        L = picks[0]["L"].copy()
        S = picks[0]["S"].copy()
        for p in picks[1:]:
            L |= p["L"]
            S |= p["S"]
        L, S = _clean(L, S)
        if L.sum() + S.sum() < 80:
            continue
        parts = [p["kind"] for p in picks]
        for ex in RNG.sample(exits, min(3, len(exits))):
            res = evaluate(
                f"ur{k}|{'+'.join(parts)}",
                "5m",
                df,
                L,
                S,
                ex,
                {"kind": "union", "parts": parts, "k": k},
            )
            consider(res, L, S, ex, tested_box, quals_box, best_box, f"rand{k}", df)


def session_partition_or(df, exits, tested_box, quals_box, best_box):
    """OR the same high-WR family across complementary sessions to raise tpd."""
    base_grid = []
    for rsi, os_, ob in itertools.product([7, 14], [40, 45, 50], [55, 60, 65]):
        base_grid.append({"rsi": rsi, "os": os_, "ob": ob})
    partitions = [
        [("asian",), ("london",), ("ny",)],
        [("asian",), ("london", "overlap"), ("ny",)],
        [("asian", "london"), ("overlap", "ny")],
        [tuple(range(0, 8)), tuple(range(8, 16)), tuple(range(16, 24))],  # hours via hours key
    ]
    for base in base_grid:
        for part in partitions:
            L = np.zeros(len(df), dtype=bool)
            S = np.zeros(len(df), dtype=bool)
            labels = []
            for piece in part:
                p = dict(base)
                if piece and isinstance(piece[0], int):
                    p["hours"] = piece
                    p["sessions"] = None
                    labels.append(f"h{piece[0]}-{piece[-1]}")
                else:
                    p["sessions"] = piece
                    labels.append("+".join(piece) if piece else "all")
                try:
                    Li, Si = gen_signals(df, "rsi_macd", p)
                except Exception:
                    continue
                L |= Li
                S |= Si
            L, S = _clean(L, S)
            if L.sum() + S.sum() < 60:
                continue
            for ex in exits:
                res = evaluate(
                    f"upart|{'|'.join(labels)}",
                    "5m",
                    df,
                    L,
                    S,
                    ex,
                    {"kind": "union_partition", "base": base, "parts": labels},
                )
                consider(res, L, S, ex, tested_box, quals_box, best_box, "partition", df)


def main():
    OUT.mkdir(exist_ok=True)
    log("=== UNION multi-OR hunter START ===")
    df = add_indicators(load_frames()["5m"])
    exits = [
        {"rr": rr, "max_hold": mh, "atr": atr, "allow_long": al, "allow_short": ash}
        for rr in (1.0, 1.2)
        for mh in (4, 6, 8, 12)
        for atr in ("ATR_14", "ATR_7")
        for al, ash in ((True, True), (True, False), (False, True))
    ]
    tested_box = [0]
    quals_box = [0]
    best_box = [None]

    while True:
        log("union: building library…")
        lib = build_library(df, mutate=True)
        log(f"union library={len(lib)}")
        singles = score_singles(df, lib, exits[:6])
        log(
            f"union strong singles={len(singles)} "
            + (
                f"top={singles[0]['min_wr']:.1%} tpd{singles[0]['tpd']:.2f} {singles[0]['kind']}"
                if singles
                else "none"
            )
        )
        # keep top near-miss singles as champ if better
        for s in singles[:5]:
            consider(s["res"], s["L"], s["S"], s["res"].params["exit"], tested_box, quals_box, best_box, "single", df)

        session_partition_or(df, exits[:12], tested_box, quals_box, best_box)
        greedy_union(df, singles, exits[:8], tested_box, quals_box, best_box, max_parts=8)
        random_unions(df, singles, exits, tested_box, quals_box, best_box, rounds=500)

        PROG.write_text(
            json.dumps(
                {
                    "tested": tested_box[0],
                    "quals": quals_box[0],
                    "singles": len(singles),
                    "best": (
                        f"{best_box[0].winrate:.1%}/{best_box[0].oos_winrate:.1%} "
                        f"tpd{best_box[0].trades_per_day:.2f}"
                        if best_box[0]
                        else None
                    ),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
        log(f"union pass done tested={tested_box[0]} quals={quals_box[0]}; loop")
        RNG.seed(int(time.time()) % 100000)


if __name__ == "__main__":
    main()
