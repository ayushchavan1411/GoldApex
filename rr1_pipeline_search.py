#!/usr/bin/env python3
"""Structured pipeline hunter (user request):

  Stage A — Exhaustive SINGLE indicator sweep on every TF (5m→4h):
            RSI levels, Fib levels, MACD, EMAs, BB, Stoch, etc.
  Stage B — Take best singles, combine with ICT/SMC/FVG/OB/BOS.
  Stage C — Multi-timeframe combos, e.g. EMA on 15m + RSI on 5m + FVG on 5m.

Native 1m is used only if present; otherwise noted and skipped.
Never stops. Strict 1:1 rr≥1.0×ATR_14.
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
    align_signal_to_index,
    evaluate,
    gen_signals,
    load_frames,
    log,
    qualifies,
    save_champion,
    write_bot,
    year_stable,
)

RNG = random.Random(14)
PROG = OUT / "pipeline_progress.json"
SINGLES = OUT / "PIPELINE_SINGLES.jsonl"
COMBOS = OUT / "PIPELINE_COMBOS.jsonl"
BEST_SINGLES = OUT / "PIPELINE_BEST_SINGLES.json"

TFS = ["5m", "15m", "30m", "1h", "2h", "4h"]  # 1m appended if available
EXITS = [
    {"rr": rr, "max_hold": mh, "atr": "ATR_14", "allow_long": al, "allow_short": ash}
    for rr in (1.0, 1.2)
    for mh in (4, 6, 8, 12)
    for al, ash in ((True, True), (True, False), (False, True))
]
SESS = [
    None,
    ("asian",),
    ("london",),
    ("overlap",),
    ("ny",),
    ("london", "overlap"),
    ("overlap", "ny"),
    ("london", "overlap", "ny"),
    ("asian", "london", "overlap", "ny"),
]


def _clean(L, S):
    both = L & S
    return L & ~both, S & ~both


def _append(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def consider(res, L, S, ex, df, tested, quals, best, tag, board: Path):
    tested[0] += 1
    if not res:
        return None
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
    row = {
        "tag": tag, "name": res.name, "wr": res.winrate, "oos_wr": res.oos_winrate,
        "tpd": res.trades_per_day, "oos_tpd": res.oos_trades_per_day,
        "n": res.trades, "oos_n": res.oos_trades, "pf": res.profit_factor,
        "oos_pf": res.oos_profit_factor, "params": res.params, "tf": res.timeframe,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # log near-misses and anything with decent sample
    if q or (min_wr >= 0.58 and res.trades >= 30 and res.oos_trades >= 15) or (min_wr >= 0.68):
        _append(board, row)
    key = (min_wr, min_tpd, res.score)
    b = best[0]
    bkey = (min(b.winrate, b.oos_winrate), min(b.trades_per_day, b.oos_trades_per_day), b.score) if b else (-1, -1, -1)
    if q or key > bkey:
        if save_champion(res, force=q):
            write_bot(res)
            best[0] = res
            if q:
                quals[0] += 1
            log(f"{'QUAL ' if q else ''}PIPE {tag} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades} {res.name}")
    return row if res.trades >= 25 else None


# --------------- Stage A: singles grids ---------------

def single_specs():
    """Exhaustive-ish single indicator parameter grids."""
    specs = []
    # RSI levels / crosses / RSI+MACD
    for rsi, os_, ob, sess in itertools.product(
        [2, 3, 5, 7, 9, 14, 21],
        [10, 15, 20, 25, 30, 35, 40, 45],
        [55, 60, 65, 70, 75, 80, 85, 90],
        SESS,
    ):
        if os_ >= ob:
            continue
        specs.append(("rsi_level", {"rsi": rsi, "os": os_, "ob": ob, "band": 5, "sessions": sess}))
        specs.append(("rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}))
        if rsi in (7, 14) and os_ in (30, 35, 40, 45) and ob in (55, 60, 65, 70):
            specs.append(("rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}))
    # EMA cross / pull
    for fast, slow, sess in itertools.product([5, 8, 9, 12, 21], [21, 34, 50, 100, 200], SESS[:6]):
        if fast >= slow:
            continue
        specs.append(("ema_cross", {"fast": fast, "slow": slow, "sessions": sess}))
        specs.append(("ema_pull_rsi", {"fast": fast, "slow": slow, "rsi": 7, "os": 35, "ob": 65, "sessions": sess}))
    # BB / Stoch / MACD / Donchian
    for period, k, sess in itertools.product([10, 20, 50], [1.5, 2.0, 2.5], SESS[:6]):
        specs.append(("bb_fade", {"period": period, "k": k, "sessions": sess}))
        specs.append(("bb_mid_reject", {"period": period, "k": k, "sessions": sess}))
    for period, os_, ob, sess in itertools.product([5, 9, 14], [15, 20, 25, 30], [70, 75, 80, 85], SESS[:5]):
        specs.append(("stoch_cross", {"period": period, "os": os_, "ob": ob, "sessions": sess}))
    for sess in SESS:
        specs.append(("macd_cross", {"sessions": sess}))
        specs.append(("donch_fade", {"period": 20, "sessions": sess}))
        specs.append(("donch_fade", {"period": 40, "sessions": sess}))
    # Fib / ICT singles (also swept alone before combining)
    for swing, fib, sess in itertools.product(
        [10, 20, 40, 60],
        ["382", "500", "618", "786"],
        SESS[:6],
    ):
        specs.append(("fib_bounce", {"swing": swing, "fib": fib, "sessions": sess}))
    for swing, sess in itertools.product([20, 40], SESS[:6]):
        specs.append(("fib_discount", {"swing": swing, "sessions": sess}))
        specs.append(("fvg_reclaim", {"sessions": sess}))
        specs.append(("fvg_rsi", {"rsi": 7, "os": 35, "ob": 65, "sessions": sess}))
        specs.append(("liq_sweep", {"swing": swing, "wick": 0.45, "sessions": sess}))
        specs.append(("ob_retest", {"atr_mult": 0.3, "sessions": sess}))
        specs.append(("bos_pullback", {"swing": swing, "ema": 21, "sessions": sess}))
        specs.append(("smc_confluence", {"swing": swing, "sessions": sess}))
    # Public
    for sess in SESS[:6]:
        specs.append(("ichimoku_tk", {"sessions": sess}))
        specs.append(("ichimoku_kumo", {"sessions": sess}))
        specs.append(("heikin_ema", {"ema": 50, "sessions": sess}))
        specs.append(("keltner_fade", {"period": 20, "mult": 2.0, "sessions": sess}))
        specs.append(("zscore_fade", {"period": 20, "th": 2.0, "sessions": sess}))
        specs.append(("supertrend_flip", {"mult": 3.0, "sessions": sess}))
        specs.append(("turtle_break", {"n": 20, "sessions": sess}))
        specs.append(("orb_fade", {"orb": "LONDON", "sessions": sess}))
        specs.append(("orb_break", {"orb": "LONDON", "sessions": sess}))
    return specs


def stage_a_singles(raw_frames, tested, quals, best):
    """Process one TF at a time to avoid OOM on 1m (910k bars × fat indicator frame)."""
    import gc
    log("=== PIPE STAGE A: exhaustive singles all TFs ===")
    specs = single_specs()
    RNG.shuffle(specs)
    tfs = [t for t in TFS if t in raw_frames]
    log(f"stageA specs={len(specs)} tfs={tfs}")
    leaders = []  # keep top singles for stage B/C
    for tf in tfs:
        log(f"stageA loading indicators for {tf} ({len(raw_frames[tf])} bars)…")
        df = add_indicators(raw_frames[tf]).copy()
        # fewer exits on non-5m; even fewer on 1m (heavy)
        if tf == "1m":
            exs = [e for e in EXITS if e["max_hold"] in (8, 12) and e["allow_long"] and e["allow_short"]][:4]
            # subsample specs on 1m for memory/time — still broad, not all 9k×exits
            use_specs = specs[::2]  # half
        elif tf == "5m":
            exs = EXITS
            use_specs = specs
        else:
            exs = [e for e in EXITS if e["max_hold"] in (6, 8, 12)][:8]
            use_specs = specs
        for kind, p in use_specs:
            try:
                L, S = gen_signals(df, kind, p)
                L, S = _clean(L, S)
            except Exception:
                continue
            if L.sum() + S.sum() < 40:
                continue
            local_best = None
            for ex in exs:
                res = evaluate(f"A|{kind}|{tf}", tf, df, L, S, ex, {"kind": kind, **p, "pipe": "A"})
                consider(res, L, S, ex, df, tested, quals, best, f"A|{tf}", SINGLES)
                if res and (local_best is None or min(res.winrate, res.oos_winrate) > min(local_best.winrate, local_best.oos_winrate)):
                    local_best = res
            if local_best and local_best.trades >= 30 and local_best.oos_trades >= 15:
                leaders.append({
                    "kind": kind, "p": p, "tf": tf,
                    "wr": local_best.winrate, "oos_wr": local_best.oos_winrate,
                    "tpd": local_best.trades_per_day, "oos_tpd": local_best.oos_trades_per_day,
                    "n": local_best.trades, "oos_n": local_best.oos_trades,
                    "exit": local_best.params["exit"],
                    "score": min(local_best.winrate, local_best.oos_winrate),
                })
            if tested[0] % 400 == 0:
                _prog(tested, quals, best, "A")
        del df
        gc.collect()
        log(f"stageA finished TF={tf} leaders_so_far={len(leaders)}")
    leaders.sort(key=lambda x: (x["score"], min(x["tpd"], x["oos_tpd"])), reverse=True)
    top = leaders[:80]
    BEST_SINGLES.write_text(json.dumps(top, indent=2, default=str))
    log(f"stageA done leaders={len(top)} top={top[0]['kind']}@{top[0]['tf']} "
        f"WR{top[0]['wr']:.1%}/{top[0]['oos_wr']:.1%} tpd{top[0]['tpd']:.2f}" if top else "stageA no leaders")
    return top


# --------------- Stage B: best singles × ICT/SMC ---------------

SMC_KINDS = [
    ("fvg_reclaim", {}),
    ("fvg_rsi", {"rsi": 7, "os": 35, "ob": 65}),
    ("liq_sweep", {"swing": 20, "wick": 0.45}),
    ("liq_sweep_fvg", {"swing": 20, "wick": 0.4}),
    ("ob_retest", {"atr_mult": 0.3}),
    ("bos_pullback", {"swing": 20, "ema": 21}),
    ("fib_bounce", {"swing": 40, "fib": "618"}),
    ("fib_discount", {"swing": 40}),
    ("smc_confluence", {"swing": 20}),
]


def stage_b_smc_combos(raw_frames, leaders, tested, quals, best):
    log("=== PIPE STAGE B: best singles × ICT/SMC/FVG ===")
    # Focus on 5m for execution; HTF leaders aligned in stage C
    df5 = add_indicators(raw_frames["5m"])
    # rebuild 5m leader signals
    base = []
    for lead in leaders:
        if lead["tf"] != "5m":
            continue
        try:
            L, S = gen_signals(df5, lead["kind"], lead["p"])
            L, S = _clean(L, S)
        except Exception:
            continue
        if L.sum() + S.sum() >= 40:
            base.append((lead, L, S))
    log(f"stageB bases={len(base)}")
    smc_lib = []
    for kind, extra in SMC_KINDS:
        for sess in SESS[:6]:
            p = {**extra, "sessions": sess}
            try:
                L, S = gen_signals(df5, kind, p)
                L, S = _clean(L, S)
            except Exception:
                continue
            if L.sum() + S.sum() >= 40:
                smc_lib.append((kind, p, L, S))
    log(f"stageB smc_lib={len(smc_lib)}")
    for lead, Lb, Sb in base:
        for kind, p, Ls, Ss in smc_lib:
            # AND
            L, S = _clean(Lb & Ls, Sb & Ss)
            if L.sum() + S.sum() >= 40:
                for ex in RNG.sample(EXITS, 4):
                    res = evaluate(
                        f"B|AND|{lead['kind']}+{kind}|5m", "5m", df5, L, S, ex,
                        {"kind": "combo_and", "a": lead["kind"], "b": kind, "pa": lead["p"], "pb": p, "pipe": "B"},
                    )
                    consider(res, L, S, ex, df5, tested, quals, best, "B-and", COMBOS)
            # OR
            L, S = _clean(Lb | Ls, Sb | Ss)
            if L.sum() + S.sum() >= 80:
                for ex in RNG.sample(EXITS, 3):
                    res = evaluate(
                        f"B|OR|{lead['kind']}+{kind}|5m", "5m", df5, L, S, ex,
                        {"kind": "combo_or", "a": lead["kind"], "b": kind, "pa": lead["p"], "pb": p, "pipe": "B"},
                    )
                    consider(res, L, S, ex, df5, tested, quals, best, "B-or", COMBOS)
            if tested[0] % 400 == 0:
                _prog(tested, quals, best, "B")
    log(f"stageB done tested={tested[0]}")


# --------------- Stage C: multi-timeframe ---------------

def stage_c_mtf(raw_frames, leaders, tested, quals, best):
    import gc
    log("=== PIPE STAGE C: multi-timeframe combos ===")
    df5 = add_indicators(raw_frames["5m"])
    idx5 = df5.index
    # Precompute HTF signal banks from leaders + common HTF filters
    htf_bank = []  # (label, L5, S5)
    for lead in leaders:
        tf = lead["tf"]
        if tf not in raw_frames or tf == "1m":
            continue  # skip aligning 1m→5m noise in MTF bank
        dft = add_indicators(raw_frames[tf]) if tf != "5m" else df5
        try:
            L, S = gen_signals(dft, lead["kind"], lead["p"])
            L, S = _clean(L, S)
        except Exception:
            if tf != "5m":
                del dft
            continue
        if tf == "5m":
            L5, S5 = L, S
        else:
            L5 = align_signal_to_index(L, dft.index, idx5)
            S5 = align_signal_to_index(S, dft.index, idx5)
            del dft
            gc.collect()
        if L5.sum() + S5.sum() < 30:
            continue
        htf_bank.append((f"{lead['kind']}@{tf}", L5, S5, lead))
    # Also add pure HTF trend masks from EMA on 15m/1h/4h
    for tf in ("15m", "1h", "4h"):
        if tf not in raw_frames:
            continue
        dft = add_indicators(raw_frames[tf])
        for kind, p in [
            ("ema_cross", {"fast": 9, "slow": 21, "sessions": None}),
            ("ema_cross", {"fast": 21, "slow": 50, "sessions": None}),
            ("ichimoku_tk", {"sessions": None}),
            ("smc_confluence", {"swing": 20, "sessions": None}),
        ]:
            try:
                L, S = gen_signals(dft, kind, p)
                L, S = _clean(L, S)
            except Exception:
                continue
            L5 = align_signal_to_index(L, dft.index, idx5)
            S5 = align_signal_to_index(S, dft.index, idx5)
            htf_bank.append((f"{kind}@{tf}", L5, S5, {"kind": kind, "p": p, "tf": tf}))
        del dft
        gc.collect()
    log(f"stageC htf_bank={len(htf_bank)}")

    # 5m execution signals (RSI / FVG / SMC)
    ltf_bank = []
    for kind, p in [
        ("rsi_level", {"rsi": 7, "os": 30, "ob": 70, "band": 5, "sessions": None}),
        ("rsi_macd", {"rsi": 14, "os": 40, "ob": 60, "sessions": ("asian",)}),
        ("rsi_macd", {"rsi": 14, "os": 45, "ob": 55, "sessions": ("asian",), "hours": tuple(range(7, 22))}),
        ("fvg_reclaim", {"sessions": ("london", "overlap", "ny")}),
        ("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": None}),
        ("smc_confluence", {"swing": 20, "sessions": ("overlap", "ny")}),
        ("bb_rsi_fade", {"period": 20, "k": 2.0, "rsi": 7, "os": 30, "ob": 70, "sessions": None}),
        ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": None}),
    ]:
        try:
            L, S = gen_signals(df5, kind, p)
            L, S = _clean(L, S)
        except Exception:
            continue
        if L.sum() + S.sum() >= 40:
            ltf_bank.append((kind, p, L, S))
    # also top 5m leaders
    for lead in leaders:
        if lead["tf"] != "5m":
            continue
        try:
            L, S = gen_signals(df5, lead["kind"], lead["p"])
            L, S = _clean(L, S)
        except Exception:
            continue
        if L.sum() + S.sum() >= 40:
            ltf_bank.append((lead["kind"], lead["p"], L, S))
    log(f"stageC ltf_bank={len(ltf_bank)}")

    # Combos: HTF filter AND LTF entry; also HTF1 AND HTF2 AND LTF
    for hlab, HL, HS, _ in htf_bank:
        for kind, p, LL, LS in ltf_bank:
            L, S = _clean(HL & LL, HS & LS)
            if L.sum() + S.sum() < 40:
                continue
            for ex in RNG.sample(EXITS, 3):
                res = evaluate(
                    f"C|MTF|{hlab}&{kind}|5m", "5m", df5, L, S, ex,
                    {"kind": "mtf_and", "htf": hlab, "ltf": kind, "ltf_p": p, "pipe": "C"},
                )
                consider(res, L, S, ex, df5, tested, quals, best, "C-mtf", COMBOS)
        if tested[0] % 400 == 0:
            _prog(tested, quals, best, "C")

    # Triple: HTF trend + 5m RSI + 5m FVG/SMC
    triples = [x for x in ltf_bank if x[0] in ("fvg_reclaim", "smc_confluence", "liq_sweep", "rsi_macd", "rsi_level")]
    for hlab, HL, HS, _ in RNG.sample(htf_bank, min(40, len(htf_bank))):
        for a, b in itertools.combinations(triples[:12], 2):
            L, S = _clean(HL & a[2] & b[2], HS & a[3] & b[3])
            if L.sum() + S.sum() < 35:
                continue
            for ex in RNG.sample(EXITS, 2):
                res = evaluate(
                    f"C|TRI|{hlab}&{a[0]}&{b[0]}|5m", "5m", df5, L, S, ex,
                    {"kind": "mtf_tri", "htf": hlab, "a": a[0], "b": b[0], "pipe": "C"},
                )
                consider(res, L, S, ex, df5, tested, quals, best, "C-tri", COMBOS)
    log(f"stageC done tested={tested[0]}")


def _prog(tested, quals, best, stage):
    b = best[0]
    PROG.write_text(json.dumps({
        "tested": tested[0], "quals": quals[0], "stage": stage,
        "best": (f"{b.winrate:.1%}/{b.oos_winrate:.1%} tpd{b.trades_per_day:.2f}" if b else None),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    log(f"pipeline[{stage}] tested={tested[0]} quals={quals[0]} "
        f"best={(f'{b.winrate:.1%}/{b.oos_winrate:.1%} tpd{b.trades_per_day:.2f}' if b else 'n/a')}")


def main():
    OUT.mkdir(exist_ok=True)
    log("=== PIPELINE singles→SMC→MTF START ===")
    # Keep RAW frames only — indicators built per-TF to avoid OOM on 1m
    raw = load_frames(include_resampled=True)
    global TFS
    # 5m first (execution TF), HTFs next, 1m last (heaviest)
    TFS = [t for t in ["5m", "15m", "30m", "1h", "2h", "4h", "1m"] if t in raw]
    log(f"pipeline TFs available: {TFS} sizes={{ {', '.join(f'{k}:{len(v)}' for k,v in raw.items())} }}")
    tested = [0]
    quals = [0]
    best = [None]

    while True:
        leaders = stage_a_singles(raw, tested, quals, best)
        if BEST_SINGLES.exists():
            try:
                leaders = json.loads(BEST_SINGLES.read_text())
            except Exception:
                pass
        if leaders:
            stage_b_smc_combos(raw, leaders, tested, quals, best)
            stage_c_mtf(raw, leaders, tested, quals, best)
        _prog(tested, quals, best, "loop")
        log(f"pipeline full loop done tested={tested[0]} quals={quals[0]}; restart stages")
        RNG.seed(int(time.time()) % 1000000)


if __name__ == "__main__":
    main()
