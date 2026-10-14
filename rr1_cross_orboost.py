#!/usr/bin/env python3
"""OR-union high-WR low-frequency seeds on cross-asset symbols to chase tpd≥1.

Collects fvg_rsi / rsi_macd / bb_rsi_fade / smc seeds with min(WR)≥0.62,
then OR-combines pairs/triples under RR≥1.0 with adaptive MIN_ATR.
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
    adaptive_min_atr,
    apply_filters,
    find_csv,
    load_symbol_df,
    log,
    near_miss,
    qualifies_strict,
)
from rr1_sl_learn_search import build_global_filters

OUT = Path("rr1_search_results")
BOARD = OUT / "CROSS_OR_LEADERBOARD.jsonl"
SUMMARY = OUT / "CROSS_OR_RESULTS.json"
MD = OUT / "CROSS_OR_RESULTS.md"

PRIORITY = [
    "USD/JPY", "GBP/USD", "USO", "SLV", "GLD", "EUR/USD", "USD/CHF",
    "USD/CAD", "AUD/USD", "NZD/USD", "SPY", "QQQ", "DIA", "IWM",
]


def seed_space():
    sessions = [None, ("london", "overlap"), ("overlap", "ny"), ("london", "overlap", "ny"),
                ("asian", "london", "overlap", "ny")]
    for rsi, os_, ob, sess in itertools.product((7, 14), (30, 35, 40), (65, 70, 75, 80), sessions):
        if os_ >= ob - 15:
            continue
        yield "fvg_rsi", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
        yield "rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    for rsi, os_, ob, sess in itertools.product((7, 14), (25, 30, 35), (70, 75, 80), sessions[:4]):
        yield "rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    for (period, k), rsi, os_, ob, sess in itertools.product(
        ((20, 2.0), (20, 2.5)), (7, 14), (30, 35), (65, 70, 75), sessions[:3]
    ):
        yield "bb_rsi_fade", {"period": period, "k": k, "rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    for kind, p in [
        ("smc_confluence", {"swing": 20, "sessions": ("overlap", "ny")}),
        ("smc_confluence", {"swing": 20, "sessions": None}),
        ("fvg_reclaim", {"sessions": ("london", "overlap", "ny")}),
        ("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": ("london", "overlap", "ny")}),
        ("ob_fvg", {"sessions": ("overlap", "ny")}),
        ("mss_entry", {"swing": 20, "sessions": ("london", "overlap", "ny")}),
        ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": None}),
    ]:
        yield kind, p


def eval_row(symbol, name, kind, filt, df, L, S, ex, p):
    res = R.evaluate(name, "5m", df, L, S, ex, {"kind": kind, "filter": filt, **p, "symbol": symbol})
    if not res:
        return None
    return {
        "symbol": symbol,
        "name": res.name,
        "kind": kind,
        "filter": filt,
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
        "exit": ex,
        "params": p,
        "qualifies": qualifies_strict(res),
        "near": near_miss(res),
        "min_wr": min(res.winrate, res.oos_winrate),
        "min_tpd": min(res.trades_per_day, res.oos_trades_per_day),
    }, res


def hunt(symbol: str) -> dict:
    path = find_csv(symbol)
    if not path:
        return {"symbol": symbol, "error": "no csv"}
    log(f"=== ORBOOST {symbol} {path.name} ===")
    df = load_symbol_df(path)
    min_atr = adaptive_min_atr(df)
    old = R.MIN_ATR
    R.MIN_ATR = min_atr
    try:
        filters, _ = build_global_filters(df, [])
        fmap = {n: (a, b) for n, a, b in filters}
        filter_opts = [(None, None, None)]
        for name in ("lon_ny", "overlap_ny", "atr_expanded", "skip_tiny_body", "rsi_not_chase",
                     "atr_not_expanded", "with_trend", "adx_min_18", "pack_trend_adx"):
            if name in fmap:
                filter_opts.append((name, *fmap[name]))
        for a, b in (("atr_expanded", "skip_tiny_body"), ("lon_ny", "skip_tiny_body"),
                     ("rsi_not_chase", "atr_not_expanded"), ("with_trend", "adx_min_18")):
            if a in fmap and b in fmap:
                filter_opts.append((f"{a}+{b}", fmap[a][0] & fmap[b][0], fmap[a][1] & fmap[b][1]))

        exits = [
            {"rr": rr, "max_hold": mh, "atr": "ATR_14", "allow_long": True, "allow_short": True}
            for rr in (1.0, 1.2) for mh in (6, 8, 12)
        ]
        seeds = []  # (score, kind, filt, L, S, row, p)
        tested = 0
        for kind, p in seed_space():
            try:
                L0, S0 = R.gen_signals(df, kind, p)
            except Exception:
                continue
            L0 = np.asarray(L0, dtype=bool)
            S0 = np.asarray(S0, dtype=bool)
            both = L0 & S0
            L0, S0 = L0 & ~both, S0 & ~both
            if int(L0.sum() + S0.sum()) < 50:
                continue
            for fname, fl, fs in filter_opts:
                L, S = (L0, S0) if fl is None else apply_filters(L0, S0, fl, fs)
                if int(L.sum() + S.sum()) < 40:
                    continue
                for ex in exits:
                    tested += 1
                    out = eval_row(symbol, f"{symbol}|seed|{kind}|{fname}", kind, fname, df, L, S, ex, p)
                    if not out:
                        continue
                    row, res = out
                    if row["min_wr"] >= 0.62 and res.trades >= 20 and res.oos_trades >= 8:
                        seeds.append((row["min_wr"], kind, fname, L.copy(), S.copy(), row, p, ex))
        seeds.sort(key=lambda x: (-x[0], -x[5]["n"]))
        # Dedup by (kind, filter, exit) keep best
        uniq = []
        seen = set()
        for s in seeds:
            key = (s[1], s[2], s[7]["rr"], s[7]["max_hold"], json.dumps(s[6], sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
        seeds = uniq[:40]
        log(f"{symbol}: seed pool={len(seeds)} tested_seeds={tested}")

        quals, nears, best = [], [], None
        best_key = (-1.0, -1.0, -1)

        def consider(row):
            nonlocal best, best_key
            key = (row["min_wr"], row["min_tpd"], row["n"] + row["oos_n"])
            if key > best_key:
                best_key, best = key, row
            if row["qualifies"] or row["near"] or row["min_wr"] >= 0.70 or (
                row["min_wr"] >= 0.65 and row["min_tpd"] >= 0.5
            ):
                with open(BOARD, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            if row["qualifies"]:
                quals.append(row)
                log(f"QUAL {symbol} {row['name']} WR{row['wr']:.1%}/{row['oos_wr']:.1%} "
                    f"tpd{row['tpd']:.2f}/{row['oos_tpd']:.2f} n{row['n']}/{row['oos_n']}")
                (OUT / f"CROSS_QUAL_{symbol.replace('/', '')}.json").write_text(
                    json.dumps(row, indent=2, default=str)
                )
            elif row["near"] or (row["min_wr"] >= 0.72 and row["oos_n"] >= 12):
                nears.append(row)
                log(f"NEAR {symbol} {row['name']} WR{row['wr']:.1%}/{row['oos_wr']:.1%} "
                    f"tpd{row['tpd']:.2f}/{row['oos_tpd']:.2f} n{row['n']}/{row['oos_n']}")

        for s in seeds:
            consider(s[5])

        # Pair OR
        or_tested = 0
        for (a, b) in itertools.combinations(seeds[:25], 2):
            L = a[3] | b[3]
            S = a[4] | b[4]
            both = L & S
            L, S = L & ~both, S & ~both
            if int(L.sum() + S.sum()) < 70:
                continue
            for ex in exits:
                or_tested += 1
                out = eval_row(
                    symbol,
                    f"{symbol}|OR|{a[1]}+{b[1]}|{a[2]}+{b[2]}|rr{ex['rr']}|h{ex['max_hold']}",
                    "or2", f"{a[2]}|{b[2]}", df, L, S, ex,
                    {"a": a[1], "b": b[1], "ap": a[6], "bp": b[6]},
                )
                if out:
                    consider(out[0])

        # Triple OR of top-12 by min_wr
        top = seeds[:12]
        for (a, b, c) in itertools.combinations(top, 3):
            L = a[3] | b[3] | c[3]
            S = a[4] | b[4] | c[4]
            both = L & S
            L, S = L & ~both, S & ~both
            if int(L.sum() + S.sum()) < 90:
                continue
            for ex in exits[::2]:
                or_tested += 1
                out = eval_row(
                    symbol,
                    f"{symbol}|OR3|{a[1]}+{b[1]}+{c[1]}|rr{ex['rr']}|h{ex['max_hold']}",
                    "or3", "triple", df, L, S, ex,
                    {"a": a[1], "b": b[1], "c": c[1]},
                )
                if out:
                    consider(out[0])

        log(f"{symbol}: OR done or_tested={or_tested} quals={len(quals)} nears={len(nears)} "
            f"best_minWR={(best['min_wr'] if best else 0):.1%} best_tpd={(best['min_tpd'] if best else 0):.2f}")
        return {
            "symbol": symbol,
            "class": ASSET_CLASS.get(symbol),
            "bars": len(df),
            "min_atr": min_atr,
            "seed_pool": len(seeds),
            "tested_seeds": tested,
            "or_tested": or_tested,
            "qualifiers": len(quals),
            "near_misses": len(nears),
            "best": best,
            "quals": quals[:20],
            "nears": sorted(nears, key=lambda r: (r["min_wr"], r["min_tpd"]), reverse=True)[:20],
        }
    finally:
        R.MIN_ATR = old


def write_md(summaries):
    any_q = any(s.get("qualifiers", 0) > 0 for s in summaries)
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "goal": "IS+OOS WR≥75%, tpd≥1.0, rr≥1.0, n≥30/15",
        "any_qualifier": any_q,
        "symbols": summaries,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, default=str))
    lines = [
        "# Cross-asset OR-boost hunt",
        "",
        f"Generated: {payload['time']}",
        "",
        f"**Any full qualifier:** {'YES' if any_q else 'NO'}",
        "",
        "| Symbol | Seeds | Best minWR | WR IS/OOS | tpd IS/OOS | n | Quals | Near |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for s in summaries:
        b = s.get("best") or {}
        lines.append(
            f"| {s.get('symbol')} | {s.get('seed_pool', 0)} | {b.get('min_wr', 0):.1%} | "
            f"{b.get('wr', 0):.1%}/{b.get('oos_wr', 0):.1%} | "
            f"{b.get('tpd', 0):.2f}/{b.get('oos_tpd', 0):.2f} | "
            f"{b.get('n', 0)}/{b.get('oos_n', 0)} | "
            f"{s.get('qualifiers', 0)} | {s.get('near_misses', 0)} |"
        )
    lines += ["", "## Qualifiers", ""]
    quals = [q for s in summaries for q in s.get("quals") or []]
    lines.append("_None._" if not quals else "")
    for q in quals:
        lines.append(
            f"- **{q['symbol']}** `{q['name']}` WR {q['wr']:.1%}/{q['oos_wr']:.1%} "
            f"tpd {q['tpd']:.2f}/{q['oos_tpd']:.2f} n={q['n']}/{q['oos_n']}"
        )
    lines += ["", "## Near / high-WR", ""]
    nears = [n for s in summaries for n in s.get("nears") or []]
    nears.sort(key=lambda r: (r["min_wr"], r["min_tpd"]), reverse=True)
    if not nears:
        lines.append("_None yet._")
    for n in nears[:40]:
        lines.append(
            f"- **{n['symbol']}** `{n['kind']}+{n['filter']}` "
            f"WR {n['wr']:.1%}/{n['oos_wr']:.1%} tpd {n['tpd']:.2f}/{n['oos_tpd']:.2f} "
            f"n={n['n']}/{n['oos_n']}"
        )
    MD.write_text("\n".join(lines) + "\n")
    log(f"wrote {MD} any_q={any_q}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    symbols = args.symbols or PRIORITY
    if BOARD.exists():
        BOARD.write_text("")
    summaries = []
    for sym in symbols:
        try:
            summaries.append(hunt(sym))
            write_md(summaries)
        except Exception as e:
            log(f"FAIL {sym}: {e}")
            traceback.print_exc()
            summaries.append({"symbol": sym, "error": str(e), "qualifiers": 0, "near_misses": 0, "best": None})
            write_md(summaries)
    write_md(summaries)
    log(f"ORBOOST DONE any_q={any(s.get('qualifiers',0)>0 for s in summaries)}")


if __name__ == "__main__":
    main()
