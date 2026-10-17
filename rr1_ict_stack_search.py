#!/usr/bin/env python3
"""ICT / FVG / SMC systematic stack hunter (user request).

Concept definitions used (causal, no look-ahead):
  FVG  — 3-candle imbalance; trade reclaim of prior filled gap edges.
  ICT  — liquidity sweep (raid swing then close back), MSS/BOS, OTE fibs,
         discount/premium vs equilibrium, displacement→OB.
  SMC  — order-block retest, BOS pullback, sweep+FVG+discount confluence.

Stages (loops forever):
  A — Pure concept singles (ICT / FVG / SMC families alone)
  B — Each concept AND each indicator filter (RSI levels, MACD, EMA, BB, Stoch)
  C — Concept×concept stacks: ICT∧FVG, FVG∧SMC, ICT∧SMC, ICT∧FVG∧SMC (+ OR variants)
  D — Those stacks AND indicator filters again

Strict RR ≥ 1.0×ATR, ≥75% WR + ≥1 tpd target, 2y local data. 5m primary (+15m).
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

RNG = random.Random(42)
PROG = OUT / "ict_stack_progress.json"
BOARD = OUT / "ICT_STACK_LEADERBOARD.jsonl"

SESS = [
    None,
    ("london",),
    ("overlap",),
    ("ny",),
    ("london", "overlap"),
    ("overlap", "ny"),
    ("london", "overlap", "ny"),
    ("asian", "london", "overlap", "ny"),
]

EXITS = [
    {"rr": rr, "max_hold": mh, "atr": atr, "allow_long": al, "allow_short": ash}
    for rr in (1.0, 1.2, 1.5)
    for mh in (4, 6, 8, 12)
    for atr in ("ATR_14", "ATR_7")
    for al, ash in ((True, True), (True, False), (False, True))
]


# ---------- concept libraries ----------

def fvg_concepts():
    out = []
    for sess in SESS:
        out.append(("fvg_reclaim", {"sessions": sess}, "FVG"))
        for rsi, os_, ob in itertools.product([5, 7, 14], [20, 25, 30, 35, 40], [60, 65, 70, 75, 80]):
            out.append(("fvg_rsi", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}, "FVG"))
    return out


def ict_concepts():
    out = []
    for sw, wick, sess in itertools.product([10, 20, 40], [0.4, 0.45, 0.55], SESS):
        out.append(("liq_sweep", {"swing": sw, "wick": wick, "sessions": sess}, "ICT"))
        out.append(("liq_sweep_fvg", {"swing": sw, "wick": wick, "sessions": sess}, "ICT"))
        out.append(("mss_entry", {"swing": sw, "sessions": sess}, "ICT"))
        out.append(("sweep_ob", {"swing": sw, "sessions": sess}, "ICT"))
        out.append(("eq_fade", {"swing": sw, "sessions": sess}, "ICT"))
    for sw, fib, wick, sess in itertools.product(
        [20, 40], ["382", "500", "618", "786"], [0.4, 0.5], SESS[:6]
    ):
        out.append(("fib_bounce", {"swing": sw, "fib": fib, "wick": wick, "sessions": sess}, "ICT"))
        out.append(("fib_discount", {"swing": sw, "fib": fib, "sessions": sess}, "ICT"))
    return out


def smc_concepts():
    out = []
    for sess in SESS:
        out.append(("ob_retest", {"sessions": sess}, "SMC"))
        out.append(("ob_fvg", {"sessions": sess}, "SMC"))
    for sw, sess in itertools.product([10, 20, 40], SESS):
        out.append(("smc_confluence", {"swing": sw, "sessions": sess}, "SMC"))
        out.append(("discount_ob", {"swing": sw, "sessions": sess}, "SMC"))
    for sw, ema, look, sess in itertools.product([20, 40], [9, 21, 50], [8, 12, 20], SESS[:5]):
        out.append(("bos_pullback", {"swing": sw, "ema": ema, "bos_look": look, "sessions": sess}, "SMC"))
    return out


# ---------- indicator filters (AND masks) ----------

def indicator_filters(df):
    """Return list of (name, long_mask, short_mask, meta)."""
    filters = []
    c = df["Close"]
    # RSI level / cross-style gates
    for rsi_p, os_, ob in itertools.product([5, 7, 14], [20, 25, 30, 35, 40], [60, 65, 70, 75, 80]):
        rsi = df[f"RSI_{rsi_p}"]
        filters.append((f"rsi{rsi_p}_{os_}_{ob}", (rsi < os_).to_numpy(), (rsi > ob).to_numpy(),
                        {"ind": "rsi", "rsi": rsi_p, "os": os_, "ob": ob}))
        # RSI mid (momentum) vs extreme
        filters.append((f"rsi{rsi_p}_mid", ((rsi > 40) & (rsi < 60)).to_numpy(),
                        ((rsi > 40) & (rsi < 60)).to_numpy(),
                        {"ind": "rsi_mid", "rsi": rsi_p}))
    # MACD hist / line
    mh = df["MACD_HIST"]
    filters.append(("macd_hist+", (mh > 0).to_numpy(), (mh < 0).to_numpy(), {"ind": "macd_hist"}))
    filters.append(("macd_flip+", ((mh > 0) & (mh.shift(1) <= 0)).to_numpy(),
                    ((mh < 0) & (mh.shift(1) >= 0)).to_numpy(), {"ind": "macd_flip"}))
    macd, sig = df["MACD"], df["MACD_SIG"]
    filters.append(("macd_line+", (macd > sig).to_numpy(), (macd < sig).to_numpy(), {"ind": "macd_line"}))
    # EMAs
    for fast, slow in ((9, 21), (21, 50), (8, 34), (12, 26)):
        f, s = df[f"EMA_{fast}"], df[f"EMA_{slow}"]
        filters.append((f"ema{fast}>{slow}", (f > s).to_numpy(), (f < s).to_numpy(),
                        {"ind": "ema_trend", "fast": fast, "slow": slow}))
        filters.append((f"ema_pull{fast}", ((c > s) & (df["Low"] <= f) & (c > f)).to_numpy(),
                        ((c < s) & (df["High"] >= f) & (c < f)).to_numpy(),
                        {"ind": "ema_pull", "fast": fast, "slow": slow}))
    # Bollinger
    for period, k in ((20, 2.0), (20, 2.5), (10, 2.0)):
        lo, up = df[f"BB_LO_{period}_{k}"], df[f"BB_UP_{period}_{k}"]
        filters.append((f"bb{period}_{k}_fade",
                        (df["Low"] <= lo).to_numpy(), (df["High"] >= up).to_numpy(),
                        {"ind": "bb_fade", "period": period, "k": k}))
        mid = df[f"BB_MID_{period}_{k}"]
        filters.append((f"bb{period}_{k}_inside",
                        (c > mid).to_numpy(), (c < mid).to_numpy(),
                        {"ind": "bb_side", "period": period, "k": k}))
    # Stoch
    for period, os_, ob in itertools.product([5, 14], [20, 25], [75, 80]):
        sk = df[f"STOCH_K_{period}"]
        filters.append((f"stoch{period}_{os_}_{ob}", (sk < os_).to_numpy(), (sk > ob).to_numpy(),
                        {"ind": "stoch", "period": period, "os": os_, "ob": ob}))
    # ADX regime
    adx = df["ADX"]
    filters.append(("adx20", (adx > 20).to_numpy(), (adx > 20).to_numpy(), {"ind": "adx", "th": 20}))
    filters.append(("adx25", (adx > 25).to_numpy(), (adx > 25).to_numpy(), {"ind": "adx", "th": 25}))
    return filters


def _as_bool(x):
    if hasattr(x, "fillna"):
        return x.fillna(False).to_numpy(dtype=bool)
    return np.asarray(x, dtype=bool)


def _clean(L, S):
    L, S = _as_bool(L), _as_bool(S)
    both = L & S
    return L & ~both, S & ~both


def _append(row):
    with open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def consider(res, L, S, ex, df, tested, quals, best, tag):
    tested[0] += 1
    if not res:
        return
    q = qualifies(
        {"trades": res.trades, "winrate": res.winrate, "profit_factor": res.profit_factor,
         "net_pnl": res.net_pnl, "trades_per_day": res.trades_per_day, "max_dd": res.max_dd},
        {"trades": res.oos_trades, "winrate": res.oos_winrate, "profit_factor": res.oos_profit_factor,
         "net_pnl": res.oos_net_pnl, "trades_per_day": res.oos_trades_per_day},
    )
    if q:
        q = year_stable(df, L, S, ex)
    min_wr = min(res.winrate, res.oos_winrate)
    min_tpd = min(res.trades_per_day, res.oos_trades_per_day)
    if q or (min_wr >= 0.62 and res.trades >= 25) or (min_wr >= 0.70) or (
        min_wr >= 0.58 and min_tpd >= 0.9 and res.trades >= 80
    ):
        _append({
            "tag": tag, "name": res.name, "wr": res.winrate, "oos_wr": res.oos_winrate,
            "tpd": res.trades_per_day, "oos_tpd": res.oos_trades_per_day,
            "n": res.trades, "oos_n": res.oos_trades, "pf": res.profit_factor,
            "oos_pf": res.oos_profit_factor, "params": res.params, "tf": res.timeframe,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    key = (min_wr, min_tpd, res.trades + res.oos_trades)
    b = best[0]
    bkey = (min(b.winrate, b.oos_winrate), min(b.trades_per_day, b.oos_trades_per_day),
            b.trades + b.oos_trades) if b else (-1, -1, -1)
    useful = (min_wr >= 0.68) or (min_wr >= 0.60 and min_tpd >= 0.85)
    if q or (useful and key > bkey):
        if save_champion(res, force=q):
            write_bot(res)
            best[0] = res
            if q:
                quals[0] += 1
            log(f"{'QUAL ' if q else ''}ICTSTACK {tag} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}")


def eval_grid(name, tf, df, L, S, params, tested, quals, best, tag, exits=None):
    L, S = _clean(L, S)
    if int(L.sum() + S.sum()) < 40:
        return
    for ex in (exits or EXITS):
        res = evaluate(name, tf, df, L, S, ex, params)
        consider(res, L, S, ex, df, tested, quals, best, tag)


def build_sig(df, kind, p):
    try:
        return gen_signals(df, kind, p)
    except Exception:
        return None


def cache_concepts(df, concepts, min_n=40, max_keep=180):
    cached = []
    for kind, p, family in concepts:
        sig = build_sig(df, kind, p)
        if sig is None:
            continue
        L, S = _clean(*sig)
        n = int(L.sum() + S.sum())
        if n < min_n:
            continue
        cached.append((kind, p, family, L, S, n))
    cached.sort(key=lambda x: -x[5])
    return cached[:max_keep]


def write_prog(stage, tested, quals, best, extra=None):
    row = {
        "stage": stage,
        "tested": tested[0],
        "quals": quals[0],
        "best": None if not best[0] else {
            "wr": best[0].winrate, "oos_wr": best[0].oos_winrate,
            "tpd": best[0].trades_per_day, "oos_tpd": best[0].oos_trades_per_day,
            "n": best[0].trades, "oos_n": best[0].oos_trades, "name": best[0].name,
        },
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        row.update(extra)
    PROG.write_text(json.dumps(row, indent=2))


def stage_a_singles(frames, tested, quals, best):
    log("=== ICT STACK STAGE A: pure FVG / ICT / SMC singles ===")
    for tf in ("5m", "15m"):
        if tf not in frames:
            continue
        df = frames[tf]
        concepts = fvg_concepts() + ict_concepts() + smc_concepts()
        RNG.shuffle(concepts)
        log(f"stageA {tf} concepts={len(concepts)}")
        for i, (kind, p, family) in enumerate(concepts):
            sig = build_sig(df, kind, p)
            if sig is None:
                continue
            L, S = sig
            eval_grid(
                f"A|{family}|{kind}|{tf}", tf, df, L, S,
                {"kind": kind, "family": family, "stage": "A", **p},
                tested, quals, best, f"A-{family}",
            )
            if (i + 1) % 50 == 0:
                write_prog("A", tested, quals, best, {"tf": tf, "i": i + 1})
                log(f"stageA {tf} i={i+1}/{len(concepts)} tested={tested[0]} quals={quals[0]}")
    write_prog("A-done", tested, quals, best)


def stage_b_concept_x_ind(frames, tested, quals, best):
    log("=== ICT STACK STAGE B: concept ∧ indicator (RSI/MACD/EMA/BB/Stoch) ===")
    tf = "5m"
    df = frames[tf]
    concepts = cache_concepts(df, fvg_concepts() + ict_concepts() + smc_concepts(), min_n=50, max_keep=120)
    filters = indicator_filters(df)
    RNG.shuffle(filters)
    filters = filters[:80]
    log(f"stageB concepts={len(concepts)} filters={len(filters)}")
    for ci, (kind, p, family, L0, S0, n) in enumerate(concepts):
        for fname, mL, mS, meta in filters:
            L, S = _clean(L0 & mL, S0 & mS)
            if int(L.sum() + S.sum()) < 50:
                continue
            eval_grid(
                f"B|{family}|{kind}+{fname}|{tf}", tf, df, L, S,
                {"kind": "concept_x_ind", "concept": kind, "family": family,
                 "filter": fname, "stage": "B", **p, **meta},
                tested, quals, best, f"B-{family}xIND",
            )
        if (ci + 1) % 10 == 0:
            write_prog("B", tested, quals, best, {"ci": ci + 1, "n_concepts": len(concepts)})
            log(f"stageB ci={ci+1}/{len(concepts)} tested={tested[0]}")
    write_prog("B-done", tested, quals, best)


def stage_c_concept_stacks(frames, tested, quals, best):
    log("=== ICT STACK STAGE C: ICT∧FVG, FVG∧SMC, ICT∧SMC, ICT∧FVG∧SMC (+OR) ===")
    tf = "5m"
    df = frames[tf]
    fvg = cache_concepts(df, fvg_concepts(), min_n=40, max_keep=40)
    ict = cache_concepts(df, ict_concepts(), min_n=40, max_keep=50)
    smc = cache_concepts(df, smc_concepts(), min_n=40, max_keep=40)
    log(f"stageC pools FVG={len(fvg)} ICT={len(ict)} SMC={len(smc)}")

    pairs = [
        ("ICT+FVG", ict[:25], fvg[:20]),
        ("FVG+SMC", fvg[:20], smc[:20]),
        ("ICT+SMC", ict[:25], smc[:20]),
    ]
    for label, a_pool, b_pool in pairs:
        for (ka, pa, fa, La, Sa, _), (kb, pb, fb, Lb, Sb, _) in itertools.product(a_pool, b_pool):
            for mode in ("and", "or"):
                if mode == "and":
                    L, S = La & Lb, Sa & Sb
                else:
                    L, S = La | Lb, Sa | Sb
                L, S = _clean(L, S)
                if int(L.sum() + S.sum()) < 55:
                    continue
                eval_grid(
                    f"C|{label}|{mode}|{ka}+{kb}|{tf}", tf, df, L, S,
                    {"kind": "concept_stack", "stack": label, "mode": mode,
                     "a": ka, "b": kb, "pa": pa, "pb": pb, "stage": "C"},
                    tested, quals, best, f"C-{label}-{mode}",
                )

    # Triple ICT ∧ FVG ∧ SMC
    for (ka, pa, _, La, Sa, _), (kb, pb, _, Lb, Sb, _), (kc, pc, _, Lc, Sc, _) in itertools.product(
        ict[:12], fvg[:10], smc[:10]
    ):
        L, S = _clean(La & Lb & Lc, Sa & Sb & Sc)
        if int(L.sum() + S.sum()) < 40:
            continue
        eval_grid(
            f"C|ICT+FVG+SMC|and|{ka}+{kb}+{kc}|{tf}", tf, df, L, S,
            {"kind": "concept_triple", "a": ka, "b": kb, "c": kc,
             "pa": pa, "pb": pb, "pc": pc, "stage": "C"},
            tested, quals, best, "C-TRIPLE",
        )
        # also (ICT|FVG) ∧ SMC
        L2, S2 = _clean((La | Lb) & Lc, (Sa | Sb) & Sc)
        if int(L2.sum() + S2.sum()) >= 55:
            eval_grid(
                f"C|(ICT|FVG)&SMC|{ka}+{kb}+{kc}|{tf}", tf, df, L2, S2,
                {"kind": "concept_mixed", "a": ka, "b": kb, "c": kc, "stage": "C"},
                tested, quals, best, "C-MIX",
            )
    write_prog("C-done", tested, quals, best)
    log(f"stageC done tested={tested[0]} quals={quals[0]}")


def stage_d_stacks_x_ind(frames, tested, quals, best):
    log("=== ICT STACK STAGE D: best concept stacks ∧ indicators ===")
    tf = "5m"
    df = frames[tf]
    fvg = cache_concepts(df, fvg_concepts(), min_n=40, max_keep=15)
    ict = cache_concepts(df, ict_concepts(), min_n=40, max_keep=20)
    smc = cache_concepts(df, smc_concepts(), min_n=40, max_keep=15)
    filters = indicator_filters(df)
    RNG.shuffle(filters)
    filters = filters[:40]

    stacks = []
    for (ka, pa, _, La, Sa, _), (kb, pb, _, Lb, Sb, _) in itertools.product(ict[:12], fvg[:10]):
        stacks.append(("ICT+FVG", ka, kb, La & Lb, Sa & Sb, {"a": ka, "b": kb, "pa": pa, "pb": pb}))
    for (ka, pa, _, La, Sa, _), (kb, pb, _, Lb, Sb, _) in itertools.product(fvg[:10], smc[:10]):
        stacks.append(("FVG+SMC", ka, kb, La & Lb, Sa & Sb, {"a": ka, "b": kb, "pa": pa, "pb": pb}))
    for (ka, pa, _, La, Sa, _), (kb, pb, _, Lb, Sb, _) in itertools.product(ict[:12], smc[:10]):
        stacks.append(("ICT+SMC", ka, kb, La & Lb, Sa & Sb, {"a": ka, "b": kb, "pa": pa, "pb": pb}))
    for (ka, pa, _, La, Sa, _), (kb, pb, _, Lb, Sb, _), (kc, pc, _, Lc, Sc, _) in itertools.product(
        ict[:8], fvg[:6], smc[:6]
    ):
        stacks.append(("ICT+FVG+SMC", f"{ka}+{kb}", kc, La & Lb & Lc, Sa & Sb & Sc,
                       {"a": ka, "b": kb, "c": kc, "pa": pa, "pb": pb, "pc": pc}))

    RNG.shuffle(stacks)
    stacks = stacks[:80]
    log(f"stageD stacks={len(stacks)} filters={len(filters)}")
    for si, (label, a, b, L0, S0, meta) in enumerate(stacks):
        L0, S0 = _clean(L0, S0)
        if int(L0.sum() + S0.sum()) < 40:
            continue
        for fname, mL, mS, imeta in filters:
            L, S = _clean(L0 & mL, S0 & mS)
            if int(L.sum() + S.sum()) < 45:
                continue
            eval_grid(
                f"D|{label}+{fname}|{tf}", tf, df, L, S,
                {"kind": "stack_x_ind", "stack": label, "filter": fname,
                 "stage": "D", **meta, **imeta},
                tested, quals, best, f"D-{label}",
            )
        if (si + 1) % 10 == 0:
            write_prog("D", tested, quals, best, {"si": si + 1})
            log(f"stageD si={si+1}/{len(stacks)} tested={tested[0]}")
    write_prog("D-done", tested, quals, best)


def main():
    OUT.mkdir(exist_ok=True)
    log("=== RR1 ICT/FVG/SMC STACK SEARCH START ===")
    log("Concepts: FVG reclaim/RSI | ICT sweep/MSS/OTE/EQ/sweep-OB | SMC OB/BOS/confluence/discount-OB")
    # 5m + 15m only to limit RAM under multi-hunter load
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("5m", "15m"):
        if tf in raw:
            frames[tf] = add_indicators(raw[tf])
            log(f"loaded {tf} bars={len(frames[tf])}")
    tested = [0]
    quals = [0]
    best = [None]
    while True:
        stage_a_singles(frames, tested, quals, best)
        stage_b_concept_x_ind(frames, tested, quals, best)
        stage_c_concept_stacks(frames, tested, quals, best)
        stage_d_stacks_x_ind(frames, tested, quals, best)
        log(f"ICT STACK full pass done tested={tested[0]} quals={quals[0]}; reshuffle+loop")
        RNG.seed(int(time.time()) % 100000)


if __name__ == "__main__":
    main()
