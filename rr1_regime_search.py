#!/usr/bin/env python3
"""Regime / MTF-proxy hunter: take high-frequency base signals and AND with trend/vol filters
to lift WR toward ≥75% while trying to keep tpd ≥ 1.0 on 2y 5m 1:1.

Uses only causal 5m proxies for higher-timeframe context (EMA stack, ADX, ATR percentile, MACD).
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
PROG = OUT / "regime_progress.json"
BOARD = OUT / "REGIME_LEADERBOARD.jsonl"


def _clean(L, S):
    both = L & S
    return L & ~both, S & ~both


def regime_masks(df):
    """Named boolean masks (causal via already-shifted indicators where needed)."""
    c = df["Close"]
    atr = df["ATR_14"]
    atr_pct = atr.rolling(100).rank(pct=True).shift(1)  # causal
    masks = {}
    # Trend: price vs EMAs (EMA is recursive on past closes — standard causal)
    masks["ema50_up"] = (c > df["EMA_50"]).to_numpy(dtype=bool)
    masks["ema50_dn"] = (c < df["EMA_50"]).to_numpy(dtype=bool)
    masks["ema200_up"] = (c > df["EMA_200"]).to_numpy(dtype=bool)
    masks["ema200_dn"] = (c < df["EMA_200"]).to_numpy(dtype=bool)
    masks["ema_stack_up"] = ((df["EMA_21"] > df["EMA_50"]) & (df["EMA_50"] > df["EMA_200"])).to_numpy(dtype=bool)
    masks["ema_stack_dn"] = ((df["EMA_21"] < df["EMA_50"]) & (df["EMA_50"] < df["EMA_200"])).to_numpy(dtype=bool)
    # ADX regime
    masks["adx_gt20"] = (df["ADX"] > 20).to_numpy(dtype=bool)
    masks["adx_gt25"] = (df["ADX"] > 25).to_numpy(dtype=bool)
    masks["adx_lt20"] = (df["ADX"] < 20).to_numpy(dtype=bool)  # range
    masks["di_bull"] = (df["PLUS_DI"] > df["MINUS_DI"]).to_numpy(dtype=bool)
    masks["di_bear"] = (df["PLUS_DI"] < df["MINUS_DI"]).to_numpy(dtype=bool)
    # Vol regime
    masks["atr_mid"] = ((atr_pct > 0.3) & (atr_pct < 0.8)).fillna(False).to_numpy(dtype=bool)
    masks["atr_high"] = (atr_pct > 0.6).fillna(False).to_numpy(dtype=bool)
    masks["atr_low"] = (atr_pct < 0.4).fillna(False).to_numpy(dtype=bool)
    # MACD
    masks["macd_up"] = (df["MACD_HIST"] > 0).to_numpy(dtype=bool)
    masks["macd_dn"] = (df["MACD_HIST"] < 0).to_numpy(dtype=bool)
    # Session busy hours
    masks["active_hrs"] = df["hour"].isin(range(7, 21)).to_numpy(dtype=bool)
    masks["london_ny"] = df["session"].isin(["london", "overlap", "ny"]).to_numpy(dtype=bool)
    return masks


def base_signal_specs():
    sess = [None, ("asian",), ("london",), ("overlap",), ("ny",), ("london", "overlap"), ("overlap", "ny"),
            ("asian", "london", "overlap", "ny")]
    specs = []
    for rsi, os_, ob, s in itertools.product([7, 14], [25, 30, 35, 40], [60, 65, 70, 75], sess):
        specs.append(("rsi_level", {"rsi": rsi, "os": os_, "ob": ob, "band": 5, "sessions": s}))
        specs.append(("rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": s}))
        specs.append(("rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": s}))
    for period, k, s in itertools.product([20], [2.0, 2.5], sess):
        specs.append(("bb_fade", {"period": period, "k": k, "sessions": s}))
        specs.append(("bb_rsi_fade", {"period": period, "k": k, "rsi": 7, "os": 30, "ob": 70, "sessions": s}))
    for s in sess:
        specs.append(("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": s}))
        specs.append(("keltner_fade", {"period": 20, "mult": 2.0, "sessions": s}))
        specs.append(("zscore_fade", {"period": 20, "th": 2.0, "sessions": s}))
        specs.append(("heikin_ema", {"ema": 50, "sessions": s}))
        specs.append(("fvg_reclaim", {"sessions": s}))
        specs.append(("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": s}))
        specs.append(("smc_confluence", {"swing": 20, "sessions": s}))
        specs.append(("supertrend_flip", {"mult": 3.0, "sessions": s}))
        specs.append(("ichimoku_tk", {"sessions": s}))
    return specs


def filter_pairs():
    """(long_mask_name, short_mask_name) — trend-aligned filters."""
    pairs = [
        ("ema50_up", "ema50_dn"),
        ("ema200_up", "ema200_dn"),
        ("ema_stack_up", "ema_stack_dn"),
        ("di_bull", "di_bear"),
        ("macd_up", "macd_dn"),
        ("ema50_up", "ema50_up"),  # long-only trend filter both sides same = long bias only useful with sides
    ]
    # combine with regime
    extras = []
    for a, b in pairs:
        for reg in ("adx_gt20", "adx_gt25", "adx_lt20", "atr_mid", "atr_high", "active_hrs", "london_ny", None):
            extras.append((a, b, reg))
    return extras


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
    interesting = (min_wr >= 0.62 and min_tpd >= 0.8) or (min_wr >= 0.70) or q
    if not interesting:
        return
    with open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "tag": tag, "name": res.name, "wr": res.winrate, "oos_wr": res.oos_winrate,
            "tpd": res.trades_per_day, "oos_tpd": res.oos_trades_per_day,
            "n": res.trades, "oos_n": res.oos_trades, "pf": res.profit_factor,
            "oos_pf": res.oos_profit_factor, "params": res.params,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }) + "\n")
    key = (min_wr, min_tpd, res.score)
    bkey = (min(best[0].winrate, best[0].oos_winrate), min(best[0].trades_per_day, best[0].oos_trades_per_day), best[0].score) if best[0] else (-1, -1, -1)
    if q or key > bkey:
        if save_champion(res, force=q):
            write_bot(res)
            best[0] = res
            if q:
                quals[0] += 1
            log(f"{'QUAL ' if q else ''}REGIME {tag} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}")


def main():
    OUT.mkdir(exist_ok=True)
    log("=== REGIME/MTF-proxy hunter START ===")
    df = add_indicators(load_frames()["5m"])
    masks = regime_masks(df)
    exits = [
        {"rr": rr, "max_hold": mh, "atr": "ATR_14", "allow_long": al, "allow_short": ash}
        for rr in (1.0, 1.2)
        for mh in (6, 8, 12)
        for al, ash in ((True, True), (True, False), (False, True))
    ]
    tested = [0]
    quals = [0]
    best = [None]

    while True:
        specs = base_signal_specs()
        RNG.shuffle(specs)
        # precompute high-freq bases
        bases = []
        for kind, p in specs[:180]:
            try:
                L, S = gen_signals(df, kind, p)
                L, S = _clean(L, S)
            except Exception:
                continue
            nsig = int(L.sum() + S.sum())
            if nsig < 200:
                continue
            bases.append((kind, p, L, S, nsig))
        log(f"regime bases={len(bases)} masks={len(masks)}")

        fpairs = filter_pairs()
        for kind, p, L0, S0, nsig in bases:
            # unfiltered baseline probe (few exits)
            for ex in exits[:3]:
                res = evaluate(f"rg|{kind}", "5m", df, L0, S0, ex, {"kind": kind, **p})
                consider(res, L0, S0, ex, df, tested, quals, best, "base")
            # apply regime filters
            for la, sa, reg in RNG.sample(fpairs, min(40, len(fpairs))):
                Lm = masks[la]
                Sm = masks[sa]
                L = L0 & Lm
                S = S0 & Sm
                if reg:
                    R = masks[reg]
                    L = L & R
                    S = S & R
                L, S = _clean(L, S)
                if L.sum() + S.sum() < 120:
                    continue
                for ex in RNG.sample(exits, 3):
                    res = evaluate(
                        f"rg|{kind}+{la}/{sa}+{reg}",
                        "5m",
                        df,
                        L,
                        S,
                        ex,
                        {"kind": "regime", "base": kind, "base_p": p, "long_m": la, "short_m": sa, "reg": reg},
                    )
                    consider(res, L, S, ex, df, tested, quals, best, f"{kind}|{la}|{reg}")

            if tested[0] % 400 == 0:
                b = best[0]
                log(f"regime tested={tested[0]} quals={quals[0]} "
                    f"best={(f'{b.winrate:.1%}/{b.oos_winrate:.1%} tpd{b.trades_per_day:.2f}' if b else 'n/a')}")
                PROG.write_text(json.dumps({
                    "tested": tested[0], "quals": quals[0],
                    "best": (f"{b.winrate:.1%}/{b.oos_winrate:.1%} tpd{b.trades_per_day:.2f}" if b else None),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, indent=2))

        log(f"regime pass done tested={tested[0]} quals={quals[0]}; loop")
        RNG.seed(int(time.time()) % 100000)


if __name__ == "__main__":
    main()
