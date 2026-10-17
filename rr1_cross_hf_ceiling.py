#!/usr/bin/env python3
"""For each cross-asset symbol, maximize min(WR_IS, WR_OOS) among setups with
min(tpd_IS, tpd_OOS) ≥ 1.0 under RR≥1.0 — answers whether the frequency gate
can coexist with the 75% WR gate.
"""
from __future__ import annotations

import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np

import rr1_search as R
from rr1_cross_asset import (
    ASSET_CLASS,
    SYMBOLS,
    adaptive_min_atr,
    find_csv,
    load_symbol_df,
    log,
)

OUT = Path("rr1_search_results")
SUMMARY = OUT / "CROSS_HF_CEILING.json"
MD = OUT / "CROSS_HF_CEILING.md"


def space():
    sessions = [
        None,
        ("asian",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
        ("asian", "london", "overlap", "ny"),
    ]
    for rsi, os_, ob, sess in itertools.product(
        (2, 3, 5, 7, 14), (20, 25, 30, 35, 40), (60, 65, 70, 75, 80), sessions
    ):
        if os_ >= ob - 15:
            continue
        yield "rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
        yield "rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
        yield "rsi_level", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess, "band": 5}
    for period, os_, ob, sess in itertools.product(
        (5, 9, 14), (15, 20, 25, 30), (70, 75, 80, 85), sessions
    ):
        yield "stoch_cross", {"period": period, "os": os_, "ob": ob, "sessions": sess}
    for (period, k), rsi, os_, ob, sess in itertools.product(
        ((10, 2.0), (20, 2.0), (20, 2.5)),
        (3, 7, 14),
        (25, 30, 35),
        (65, 70, 75),
        sessions[:4],
    ):
        yield "bb_rsi_fade", {
            "period": period, "k": k, "rsi": rsi, "os": os_, "ob": ob, "sessions": sess
        }
    for kind, p in [
        ("macd_cross", {"sessions": None}),
        ("macd_cross", {"sessions": ("london", "overlap", "ny")}),
        ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": None}),
        ("ema_cross", {"fast": 9, "slow": 21, "sessions": None}),
    ]:
        yield kind, p


def hunt(symbol: str) -> dict:
    path = find_csv(symbol)
    if not path:
        return {"symbol": symbol, "error": "no csv"}
    log(f"=== HF-CEILING {symbol} ===")
    df = load_symbol_df(path)
    min_atr = adaptive_min_atr(df)
    old = R.MIN_ATR
    R.MIN_ATR = min_atr
    try:
        exits = [
            {"rr": rr, "max_hold": mh, "atr": atr, "allow_long": al, "allow_short": ash}
            for rr in (1.0, 1.2)
            for mh in (4, 6, 8, 12)
            for atr in ("ATR_14",)
            for al, ash in ((True, True), (True, False), (False, True))
        ]
        best_hf = None  # among tpd>=1
        best_any = None
        tested = 0
        hf_n = 0
        for kind, p in space():
            try:
                L, S = R.gen_signals(df, kind, p)
            except Exception:
                continue
            L = np.asarray(L, dtype=bool)
            S = np.asarray(S, dtype=bool)
            both = L & S
            L, S = L & ~both, S & ~both
            if int(L.sum() + S.sum()) < 200:  # need frequency headroom
                continue
            for ex in exits:
                tested += 1
                res = R.evaluate(
                    f"{symbol}|{kind}", "5m", df, L, S, ex,
                    {"kind": kind, **p, "symbol": symbol},
                )
                if not res:
                    continue
                row = {
                    "symbol": symbol,
                    "kind": kind,
                    "params": p,
                    "exit": ex,
                    "wr": res.winrate,
                    "oos_wr": res.oos_winrate,
                    "tpd": res.trades_per_day,
                    "oos_tpd": res.oos_trades_per_day,
                    "n": res.trades,
                    "oos_n": res.oos_trades,
                    "pf": res.profit_factor,
                    "oos_pf": res.oos_profit_factor,
                    "min_wr": min(res.winrate, res.oos_winrate),
                    "min_tpd": min(res.trades_per_day, res.oos_trades_per_day),
                }
                if best_any is None or (row["min_wr"], row["min_tpd"]) > (
                    best_any["min_wr"], best_any["min_tpd"]
                ):
                    best_any = row
                if row["min_tpd"] >= 1.0 and res.trades >= 30 and res.oos_trades >= 15:
                    hf_n += 1
                    if best_hf is None or row["min_wr"] > best_hf["min_wr"] or (
                        row["min_wr"] == best_hf["min_wr"] and row["min_tpd"] > best_hf["min_tpd"]
                    ):
                        best_hf = row
                        log(
                            f"  HF {symbol} minWR={row['min_wr']:.1%} "
                            f"WR={row['wr']:.1%}/{row['oos_wr']:.1%} "
                            f"tpd={row['tpd']:.2f}/{row['oos_tpd']:.2f} "
                            f"n={row['n']}/{row['oos_n']} {kind}"
                        )
        goal = bool(
            best_hf
            and best_hf["min_wr"] >= 0.75
            and best_hf["min_tpd"] >= 1.0
        )
        log(
            f"{symbol}: tested={tested} hf_hits={hf_n} "
            f"best_hf_minWR={(best_hf['min_wr'] if best_hf else None)} "
            f"goal={goal}"
        )
        return {
            "symbol": symbol,
            "class": ASSET_CLASS.get(symbol),
            "tested": tested,
            "hf_hits": hf_n,
            "best_hf": best_hf,
            "best_any": best_any,
            "goal_reachable_in_grid": goal,
            "min_atr": min_atr,
        }
    finally:
        R.MIN_ATR = old


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    # Prefer FX (higher bar count / more tpd potential) then ETFs
    order = [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD",
        "SPY", "QQQ", "DIA", "IWM", "USO", "GLD", "SLV",
    ]
    symbols = args.symbols or order
    summaries = []
    for sym in symbols:
        try:
            summaries.append(hunt(sym))
            payload = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "any_goal": any(s.get("goal_reachable_in_grid") for s in summaries),
                "symbols": summaries,
            }
            SUMMARY.write_text(json.dumps(payload, indent=2, default=str))
            lines = [
                "# Cross-asset high-frequency WR ceiling (tpd≥1 gate)",
                "",
                f"Generated: {payload['time']}",
                "",
                f"**Any symbol hit ≥75% WR with tpd≥1 in this grid:** "
                f"{'YES' if payload['any_goal'] else 'NO'}",
                "",
                "| Symbol | HF hits | Best minWR @ tpd≥1 | WR IS/OOS | tpd IS/OOS | n | Goal? |",
                "|---|---:|---:|---|---|---:|---|",
            ]
            for s in summaries:
                b = s.get("best_hf") or {}
                lines.append(
                    f"| {s.get('symbol')} | {s.get('hf_hits', 0)} | "
                    f"{b.get('min_wr', 0):.1%} | "
                    f"{b.get('wr', 0):.1%}/{b.get('oos_wr', 0):.1%} | "
                    f"{b.get('tpd', 0):.2f}/{b.get('oos_tpd', 0):.2f} | "
                    f"{b.get('n', 0)}/{b.get('oos_n', 0)} | "
                    f"{'YES' if s.get('goal_reachable_in_grid') else 'no'} |"
                )
            MD.write_text("\n".join(lines) + "\n")
        except Exception as e:
            log(f"FAIL {sym}: {e}")
            traceback.print_exc()
            summaries.append({"symbol": sym, "error": str(e), "goal_reachable_in_grid": False})
    log(f"HF-CEILING DONE any_goal={any(s.get('goal_reachable_in_grid') for s in summaries)}")


if __name__ == "__main__":
    main()
