#!/usr/bin/env python3
"""Exhaustive AND/OR combination hunter across ALL strategy families.

Tries:
  - all 2-way AND of library signals
  - all 2-way OR of library signals
  - random 3/4/5-way AND and OR
  - mixed (A|B) & C patterns

Strict 1:1 (rr≥1.0×ATR), MIN_ATR floor, year_stable for qualifiers.
Never stops until killed.
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

RNG = random.Random(2026)
PROG = OUT / "allcombo_progress.json"
BOARD = OUT / "ALLCOMBO_LEADERBOARD.jsonl"

# Every gen_signals kind we support (excl. meta combo_and)
ALL_KINDS = [
    "rsi_cross", "rsi_level", "bb_fade", "bb_rsi_fade", "bb_mid_reject",
    "stoch_cross", "triple_fade", "ema_wick", "ema_stack_pull", "range_fade",
    "micro_reclaim", "pin_rsi", "engulf", "macd_cross", "donch_fade",
    "fvg_reclaim", "fvg_rsi", "liq_sweep", "liq_sweep_fvg", "ob_retest",
    "bos_pullback", "fib_bounce", "fib_discount", "smc_confluence",
    "ichimoku_tk", "ichimoku_kumo", "rsi_macd", "ema_cross", "ema_pull_rsi",
    "heikin_flip", "heikin_ema", "keltner_fade", "zscore_fade", "cci_fade",
    "adx_di_cross", "supertrend_flip", "turtle_break", "orb_break", "orb_fade",
]

SESS = [
    None,
    ("asian",),
    ("london",),
    ("overlap",),
    ("ny",),
    ("asian", "london"),
    ("london", "overlap"),
    ("overlap", "ny"),
    ("london", "overlap", "ny"),
    ("asian", "london", "overlap", "ny"),
]


def _clean(L, S):
    both = L & S
    return L & ~both, S & ~both


def default_params(kind: str, sess) -> dict | None:
    """One workable default param set per kind."""
    p = {"sessions": sess}
    if kind in ("rsi_cross", "rsi_level", "rsi_macd", "fvg_rsi", "pin_rsi", "ema_pull_rsi"):
        p.update(rsi=14, os=35, ob=65)
        if kind == "rsi_level":
            p["band"] = 5
        if kind == "ema_pull_rsi":
            p.update(fast=9, slow=21)
    elif kind in ("bb_fade", "bb_mid_reject"):
        p.update(period=20, k=2.0)
    elif kind == "bb_rsi_fade":
        p.update(period=20, k=2.0, rsi=7, os=30, ob=70)
    elif kind == "stoch_cross":
        p.update(period=14, os=20, ob=80)
    elif kind == "triple_fade":
        p.update(rsi=7, os=30, ob=70, bb=20, k=2.0, stoch=14)
    elif kind == "ema_wick":
        p.update(ema=21, wick=0.5)
    elif kind == "ema_stack_pull":
        p.update(fast=8, mid=21, slow=50)
    elif kind == "range_fade":
        p.update(look=20)
    elif kind == "micro_reclaim":
        p.update(look=3)
    elif kind == "engulf":
        p.update({})
    elif kind == "macd_cross":
        p.update({})
    elif kind == "donch_fade":
        p.update(period=20)
    elif kind in ("fvg_reclaim",):
        pass
    elif kind in ("liq_sweep", "liq_sweep_fvg"):
        p.update(swing=20, wick=0.45)
    elif kind == "ob_retest":
        p.update(atr_mult=0.3)
    elif kind == "bos_pullback":
        p.update(swing=20)
    elif kind == "fib_bounce":
        p.update(swing=40, level="FIB618")
    elif kind == "fib_discount":
        p.update(swing=40)
    elif kind == "smc_confluence":
        p.update(swing=20)
    elif kind in ("ichimoku_tk", "ichimoku_kumo", "heikin_flip", "macd_cross"):
        pass
    elif kind == "ema_cross":
        p.update(fast=9, slow=21)
    elif kind == "heikin_ema":
        p.update(ema=50)
    elif kind == "keltner_fade":
        p.update(period=20, mult=2.0)
    elif kind == "zscore_fade":
        p.update(period=20, th=2.0)
    elif kind == "cci_fade":
        p.update(period=20, th=100)
    elif kind == "adx_di_cross":
        p.update(th=25)
    elif kind == "supertrend_flip":
        p.update(mult=3.0)
    elif kind == "turtle_break":
        p.update(n=20)
    elif kind in ("orb_break", "orb_fade"):
        p.update(orb="LONDON")
    else:
        return None
    return p


def variant_params(kind: str, sess) -> list[dict]:
    """Several param variants per kind for broader coverage."""
    base = default_params(kind, sess)
    if base is None:
        return []
    out = [base]
    # mutate a few copies
    for _ in range(3):
        p = dict(base)
        if "os" in p:
            p["os"] = int(max(15, min(50, p["os"] + RNG.choice([-10, -5, 0, 5, 10]))))
        if "ob" in p:
            p["ob"] = int(max(55, min(90, p["ob"] + RNG.choice([-10, -5, 0, 5, 10]))))
        if "rsi" in p:
            p["rsi"] = RNG.choice([7, 14])
        if "swing" in p:
            p["swing"] = RNG.choice([10, 20, 40])
        if "mult" in p:
            p["mult"] = RNG.choice([1.5, 2.0, 2.5, 3.0])
        if "th" in p:
            p["th"] = RNG.choice([1.5, 2.0, 2.5, 100, 150]) if kind == "cci_fade" else RNG.choice([1.5, 2.0, 2.5])
        if "orb" in p:
            p["orb"] = RNG.choice(["ASIAN", "LONDON", "NY"])
        if "k" in p:
            p["k"] = RNG.choice([1.5, 2.0, 2.5])
        if "n" in p:
            p["n"] = RNG.choice([10, 20, 40])
        out.append(p)
    # dedupe
    seen = set()
    uniq = []
    for p in out:
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def build_library(df, max_per_kind=8):
    lib = []
    for kind in ALL_KINDS:
        count = 0
        for sess in SESS:
            if count >= max_per_kind:
                break
            for p in variant_params(kind, sess):
                if count >= max_per_kind:
                    break
                try:
                    L, S = gen_signals(df, kind, p)
                    L, S = _clean(L, S)
                except Exception:
                    continue
                n = int(L.sum() + S.sum())
                if 40 <= n <= 12000:
                    lib.append({"kind": kind, "p": p, "L": L, "S": S, "n": n})
                    count += 1
    return lib


def exits():
    return [
        {"rr": rr, "max_hold": mh, "atr": atr, "allow_long": al, "allow_short": ash}
        for rr in (1.0, 1.2)
        for mh in (6, 8, 12)
        for atr in ("ATR_14",)
        for al, ash in ((True, True), (True, False), (False, True))
    ]


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
    interesting = q or min_wr >= 0.68 or (min_wr >= 0.60 and min_tpd >= 0.8)
    if not interesting:
        return
    with open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "tag": tag, "name": res.name, "wr": res.winrate, "oos_wr": res.oos_winrate,
            "tpd": res.trades_per_day, "oos_tpd": res.oos_trades_per_day,
            "n": res.trades, "oos_n": res.oos_trades, "pf": res.profit_factor,
            "oos_pf": res.oos_profit_factor, "params": res.params,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, default=str) + "\n")
    key = (min_wr, min_tpd, res.score)
    b = best[0]
    bkey = (min(b.winrate, b.oos_winrate), min(b.trades_per_day, b.oos_trades_per_day), b.score) if b else (-1, -1, -1)
    if q or key > bkey:
        if save_champion(res, force=q):
            write_bot(res)
            best[0] = res
            if q:
                quals[0] += 1
            log(f"{'QUAL ' if q else ''}ALLCOMBO {tag} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}")


def eval_combo(df, parts, mode, exits_list, tested, quals, best, tag_prefix):
    """mode: 'and' or 'or'. parts = list of lib entries."""
    if mode == "and":
        L = parts[0]["L"].copy()
        S = parts[0]["S"].copy()
        for p in parts[1:]:
            L &= p["L"]
            S &= p["S"]
    else:
        L = parts[0]["L"].copy()
        S = parts[0]["S"].copy()
        for p in parts[1:]:
            L |= p["L"]
            S |= p["S"]
    L, S = _clean(L, S)
    nsig = int(L.sum() + S.sum())
    if nsig < (80 if mode == "or" else 40):
        return
    kinds = [p["kind"] for p in parts]
    name = f"{tag_prefix}|{'&' if mode=='and' else '|'}|{'+'.join(kinds)}"
    for ex in RNG.sample(exits_list, min(4, len(exits_list))):
        res = evaluate(
            name, "5m", df, L, S, ex,
            {"kind": f"combo_{mode}", "parts": kinds, "mode": mode,
             "part_params": [p["p"] for p in parts]},
        )
        consider(res, L, S, ex, df, tested, quals, best, f"{mode}{len(parts)}")


def main():
    OUT.mkdir(exist_ok=True)
    log("=== ALLCOMBO exhaustive AND/OR hunter START ===")
    log(f"kinds={len(ALL_KINDS)}")
    df = add_indicators(load_frames()["5m"])
    exs = exits()
    tested = [0]
    quals = [0]
    best = [None]
    pass_n = 0

    while True:
        pass_n += 1
        log(f"allcombo: building library (pass {pass_n})…")
        lib = build_library(df, max_per_kind=6 + (pass_n % 4))
        # diversify: shuffle and keep up to 120
        RNG.shuffle(lib)
        lib = lib[:120]
        log(f"allcombo library={len(lib)} kinds={len({x['kind'] for x in lib})}")

        # --- ALL 2-way AND (by kind diversity: different kinds preferred) ---
        pairs = list(itertools.combinations(range(len(lib)), 2))
        RNG.shuffle(pairs)
        for i, j in pairs:
            if lib[i]["kind"] == lib[j]["kind"] and RNG.random() < 0.7:
                continue  # skip most same-kind pairs
            eval_combo(df, [lib[i], lib[j]], "and", exs, tested, quals, best, "ac")
            if tested[0] % 500 == 0:
                _progress(tested, quals, best, lib)

        # --- ALL 2-way OR ---
        RNG.shuffle(pairs)
        for i, j in pairs[: min(len(pairs), 5000)]:
            if lib[i]["kind"] == lib[j]["kind"] and RNG.random() < 0.5:
                continue
            eval_combo(df, [lib[i], lib[j]], "or", exs, tested, quals, best, "ac")
            if tested[0] % 500 == 0:
                _progress(tested, quals, best, lib)

        # --- 3/4/5-way random AND and OR ---
        for k in (3, 4, 5):
            if len(lib) < k:
                continue
            for _ in range(800 if k == 3 else 400):
                # prefer unique kinds
                picks = []
                used = set()
                shuffled = lib[:]
                RNG.shuffle(shuffled)
                for item in shuffled:
                    if item["kind"] in used and len(picks) < k - 1:
                        continue
                    picks.append(item)
                    used.add(item["kind"])
                    if len(picks) >= k:
                        break
                if len(picks) < k:
                    continue
                mode = RNG.choice(["and", "or", "or", "and"])
                eval_combo(df, picks, mode, exs, tested, quals, best, "ac")

        # --- mixed: (A|B) & C ---
        for _ in range(600):
            if len(lib) < 3:
                break
            a, b, c = RNG.sample(lib, 3)
            L = (a["L"] | b["L"]) & c["L"]
            S = (a["S"] | b["S"]) & c["S"]
            L, S = _clean(L, S)
            if L.sum() + S.sum() < 50:
                continue
            for ex in RNG.sample(exs, 3):
                res = evaluate(
                    f"ac|mix|{a['kind']}|{b['kind']}&{c['kind']}",
                    "5m", df, L, S, ex,
                    {"kind": "combo_mix", "or": [a["kind"], b["kind"]], "and": c["kind"]},
                )
                consider(res, L, S, ex, df, tested, quals, best, "mix")

        _progress(tested, quals, best, lib)
        log(f"allcombo pass {pass_n} done tested={tested[0]} quals={quals[0]}; rebuild+loop")
        RNG.seed(int(time.time()) % 1000000)


def _progress(tested, quals, best, lib):
    b = best[0]
    PROG.write_text(json.dumps({
        "tested": tested[0],
        "quals": quals[0],
        "lib": len(lib),
        "best": (f"{b.winrate:.1%}/{b.oos_winrate:.1%} tpd{b.trades_per_day:.2f}" if b else None),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    if tested[0] % 500 == 0:
        log(f"allcombo tested={tested[0]} quals={quals[0]} "
            f"best={(f'{b.winrate:.1%}/{b.oos_winrate:.1%} tpd{b.trades_per_day:.2f}' if b else 'n/a')}")


if __name__ == "__main__":
    main()
