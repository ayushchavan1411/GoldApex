#!/usr/bin/env python3
"""Deeper cross-asset hunt: denser grids + SL filters on each downloaded symbol.

Goal: IS+OOS WR≥75%, tpd≥1.0, rr≥1.0, n≥30/15.
Prioritizes USD/JPY (already saw 75%/80% at low tpd) and high-frequency FX.
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
    apply_filters,
    find_csv,
    load_symbol_df,
    log,
    near_miss,
    qualifies_strict,
)
from rr1_sl_learn_search import build_global_filters

OUT = Path("rr1_search_results")
BOARD = OUT / "CROSS_DEEP_LEADERBOARD.jsonl"
PROG = OUT / "cross_deep_progress.json"
SUMMARY = OUT / "CROSS_DEEP_RESULTS.json"
MD = OUT / "CROSS_DEEP_RESULTS.md"


def exits_grid(dense: bool = False):
    holds = (4, 6, 8, 12) if dense else (6, 8, 12)
    atrs = ("ATR_14", "ATR_7") if dense else ("ATR_14",)
    sides_opts = (
        (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        )
        if dense
        else (dict(allow_long=True, allow_short=True),)
    )
    for rr, mh, atr, sides in itertools.product((1.0, 1.2), holds, atrs, sides_opts):
        yield {"rr": rr, "max_hold": mh, "atr": atr, **sides}


def kind_space():
    """Focused grid: gold winners + enough coverage to chase tpd on high-WR seeds."""
    sessions = [
        None,
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
        ("asian", "london", "overlap", "ny"),
    ]
    for rsi, os_, ob, sess in itertools.product(
        (7, 14), (30, 35, 40), (60, 65, 70, 75, 80), sessions
    ):
        if os_ >= ob - 15:
            continue
        yield "fvg_rsi", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
        yield "rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    for rsi, os_, ob, sess in itertools.product(
        (5, 7, 14), (25, 30, 35), (70, 75, 80), sessions[:4]
    ):
        yield "rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
    for period, os_, ob, sess in itertools.product(
        (5, 14), (20, 25, 30), (75, 80, 85), sessions[:4]
    ):
        yield "stoch_cross", {"period": period, "os": os_, "ob": ob, "sessions": sess}
    for (period, k), rsi, os_, ob, sess in itertools.product(
        ((20, 2.0), (20, 2.5)),
        (7, 14),
        (25, 30, 35),
        (65, 70, 75),
        sessions[:3],
    ):
        yield "bb_rsi_fade", {
            "period": period, "k": k, "rsi": rsi, "os": os_, "ob": ob, "sessions": sess
        }
    for kind, p in [
        ("fvg_reclaim", {"sessions": ("london", "overlap", "ny")}),
        ("fvg_reclaim", {"sessions": ("overlap", "ny")}),
        ("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": ("london", "overlap", "ny")}),
        ("smc_confluence", {"swing": 20, "sessions": ("overlap", "ny")}),
        ("smc_confluence", {"swing": 20, "sessions": None}),
        ("ob_fvg", {"sessions": ("overlap", "ny")}),
        ("mss_entry", {"swing": 20, "sessions": ("london", "overlap", "ny")}),
        ("bos_pullback", {"swing": 20, "ema": 21, "bos_look": 12, "sessions": ("london", "overlap")}),
        ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": None}),
        ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 14, "os": 40, "ob": 60, "sessions": ("overlap", "ny")}),
    ]:
        yield kind, p


def hunt_symbol(symbol: str) -> dict:
    path = find_csv(symbol)
    if not path:
        return {"symbol": symbol, "error": "no csv"}
    log(f"=== DEEP {symbol} ({ASSET_CLASS.get(symbol)}) {path.name} ===")
    df = load_symbol_df(path)
    min_atr = adaptive_min_atr(df)
    old = R.MIN_ATR
    R.MIN_ATR = min_atr
    try:
        filters, _ = build_global_filters(df, [])
        fmap = {n: (a, b) for n, a, b in filters}
        # Lean filter set: raw + strongest singles + dual packs that helped gold/JPY
        keep_singles = {
            "with_trend", "adx_min_18", "adx_min_22", "atr_expanded", "atr_not_expanded",
            "skip_tiny_body", "rsi_not_chase", "macd_align", "lon_ny", "overlap_ny",
            "pack_trend_adx", "pack_fade_safe",
        }
        filter_opts = [(None, None, None)]
        for name, fl, fs in filters:
            if name in keep_singles or name.startswith("avoid_hours"):
                filter_opts.append((name, fl, fs))
        for a, b in [
            ("atr_expanded", "skip_tiny_body"),
            ("with_trend", "adx_min_18"),
            ("lon_ny", "skip_tiny_body"),
            ("rsi_not_chase", "atr_not_expanded"),
            ("pack_trend_adx", "skip_tiny_body"),
            ("overlap_ny", "atr_expanded"),
            ("macd_align", "adx_min_18"),
        ]:
            if a in fmap and b in fmap:
                fl = fmap[a][0] & fmap[b][0]
                fs = fmap[a][1] & fmap[b][1]
                filter_opts.append((f"{a}+{b}", fl, fs))

        exits_dense = list(exits_grid(dense=True))
        exits_lite = list(exits_grid(dense=False))
        tested = 0
        quals = []
        nears = []
        best = None
        best_key = (-1.0, -1.0, -1)
        kinds = list(kind_space())
        log(
            f"{symbol}: bars={len(df)} kinds={len(kinds)} filters={len(filter_opts)} "
            f"exits={len(exits_lite)}/{len(exits_dense)} MIN_ATR={min_atr:.6g}"
        )

        for ki, (kind, p) in enumerate(kinds):
            try:
                L0, S0 = R.gen_signals(df, kind, p)
            except Exception:
                continue
            L0 = np.asarray(L0, dtype=bool)
            S0 = np.asarray(S0, dtype=bool)
            both = L0 & S0
            L0, S0 = L0 & ~both, S0 & ~both
            nsig0 = int(L0.sum() + S0.sum())
            if nsig0 < 40:
                continue
            for fname, fl, fs in filter_opts:
                if fl is None:
                    L, S = L0, S0
                else:
                    L, S = apply_filters(L0, S0, fl, fs)
                if int(L.sum() + S.sum()) < 35:
                    continue
                # Dense exits only for high-WR gold family + dual filters
                if kind in ("fvg_rsi", "rsi_macd") and (fname is None or "+" in str(fname or "")):
                    use_exits = exits_dense
                else:
                    use_exits = exits_lite
                for ex in use_exits:
                    tested += 1
                    name = f"{symbol}|{kind}|{fname or 'raw'}|rr{ex['rr']}|h{ex['max_hold']}"
                    res = R.evaluate(
                        name, "5m", df, L, S, ex,
                        {"kind": kind, **p, "filter": fname, "symbol": symbol},
                    )
                    if not res:
                        continue
                    row = {
                        "symbol": symbol,
                        "class": ASSET_CLASS.get(symbol),
                        "name": res.name,
                        "kind": kind,
                        "filter": fname,
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
                    }
                    min_wr = min(res.winrate, res.oos_winrate)
                    min_tpd = min(res.trades_per_day, res.oos_trades_per_day)
                    key = (min_wr, min_tpd, res.trades + res.oos_trades)
                    if key > best_key:
                        best_key = key
                        best = row
                    interesting = (
                        row["qualifies"]
                        or row["near"]
                        or (min_wr >= 0.72 and res.oos_trades >= 12)
                        or (min_wr >= 0.68 and min_tpd >= 0.5)
                        or (min_wr >= 0.65 and min_tpd >= 0.9)
                    )
                    if interesting:
                        with open(BOARD, "a", encoding="utf-8") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                    if row["qualifies"]:
                        quals.append(row)
                        log(
                            f"QUAL {symbol} {kind}+{fname} WR{res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd{res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                            f"n{res.trades}/{res.oos_trades}"
                        )
                        # persist champion-style note (do not overwrite gold CHAMPION)
                        (OUT / f"CROSS_QUAL_{symbol.replace('/', '')}.json").write_text(
                            json.dumps(row, indent=2, default=str)
                        )
                    elif row["near"] or (min_wr >= 0.74 and res.oos_trades >= 12):
                        nears.append(row)
                        log(
                            f"NEAR {symbol} {kind}+{fname} WR{res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd{res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                            f"n{res.trades}/{res.oos_trades}"
                        )
            if (ki + 1) % 80 == 0:
                PROG.write_text(json.dumps({
                    "symbol": symbol, "ki": ki + 1, "kinds": len(kinds),
                    "tested": tested, "quals": len(quals), "nears": len(nears),
                    "best": best, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, indent=2, default=str))
                log(f"{symbol} progress {ki+1}/{len(kinds)} tested={tested} quals={len(quals)} nears={len(nears)} "
                    f"best_minWR={(best_key[0] if best else 0):.1%}")

        return {
            "symbol": symbol,
            "class": ASSET_CLASS.get(symbol),
            "bars": len(df),
            "days": max(1, (df.index[-1] - df.index[0]).days + 1),
            "atr_med": float(df["ATR_14"].median()),
            "min_atr": min_atr,
            "tested": tested,
            "qualifiers": len(quals),
            "near_misses": len(nears),
            "best": best,
            "quals": quals[:20],
            "nears": sorted(
                nears,
                key=lambda r: (min(r["wr"], r["oos_wr"]), min(r["tpd"], r["oos_tpd"])),
                reverse=True,
            )[:20],
        }
    finally:
        R.MIN_ATR = old


def write_md(summaries: list[dict]):
    any_q = any(s.get("qualifiers", 0) > 0 for s in summaries)
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "goal": "IS+OOS WR≥75%, tpd≥1.0, rr≥1.0, n≥30/15",
        "any_qualifier": any_q,
        "symbols": summaries,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, default=str))
    lines = [
        "# Cross-asset deep hunt",
        "",
        f"Generated: {payload['time']}",
        "",
        f"**Any full qualifier:** {'YES' if any_q else 'NO'}",
        "",
        "| Symbol | Class | Tested | Best minWR | WR IS/OOS | tpd IS/OOS | n | Quals | Near |",
        "|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for s in summaries:
        b = s.get("best") or {}
        mw = min(b.get("wr", 0), b.get("oos_wr", 0)) if b else 0
        lines.append(
            f"| {s.get('symbol')} | {s.get('class')} | {s.get('tested', 0)} | {mw:.1%} | "
            f"{b.get('wr', 0):.1%}/{b.get('oos_wr', 0):.1%} | "
            f"{b.get('tpd', 0):.2f}/{b.get('oos_tpd', 0):.2f} | "
            f"{b.get('n', 0)}/{b.get('oos_n', 0)} | "
            f"{s.get('qualifiers', 0)} | {s.get('near_misses', 0)} |"
        )
    lines += ["", "## Qualifiers", ""]
    quals = [q for s in summaries for q in s.get("quals") or []]
    if not quals:
        lines.append("_None._")
    else:
        for q in quals:
            lines.append(
                f"- **{q['symbol']}** `{q['kind']}+{q['filter']}` "
                f"WR {q['wr']:.1%}/{q['oos_wr']:.1%} tpd {q['tpd']:.2f}/{q['oos_tpd']:.2f} "
                f"n={q['n']}/{q['oos_n']}"
            )
    lines += ["", "## Near / high-WR", ""]
    nears = [n for s in summaries for n in s.get("nears") or []]
    nears.sort(key=lambda r: (min(r["wr"], r["oos_wr"]), min(r["tpd"], r["oos_tpd"])), reverse=True)
    if not nears:
        lines.append("_None yet._")
    else:
        for n in nears[:40]:
            lines.append(
                f"- **{n['symbol']}** `{n['kind']}+{n['filter']}` "
                f"WR {n['wr']:.1%}/{n['oos_wr']:.1%} tpd {n['tpd']:.2f}/{n['oos_tpd']:.2f} "
                f"n={n['n']}/{n['oos_n']} PF {n['pf']:.2f}/{n['oos_pf']:.2f}"
            )
    MD.write_text("\n".join(lines) + "\n")
    log(f"wrote {MD} any_q={any_q}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    # Prioritize closest hits first
    default_order = [
        "USD/JPY", "GBP/USD", "USO", "SLV", "GLD", "USD/CHF", "EUR/USD",
        "USD/CAD", "AUD/USD", "NZD/USD", "SPY", "QQQ", "DIA", "IWM",
    ]
    args = ap.parse_args()
    symbols = args.symbols or default_order
    if BOARD.exists():
        BOARD.write_text("")  # fresh board this run
    summaries = []
    for sym in symbols:
        if sym not in SYMBOLS and find_csv(sym) is None:
            log(f"skip unknown {sym}")
            continue
        try:
            summaries.append(hunt_symbol(sym))
            write_md(summaries)
        except Exception as e:
            log(f"FAIL {sym}: {e}")
            traceback.print_exc()
            summaries.append({"symbol": sym, "error": str(e), "qualifiers": 0, "near_misses": 0, "best": None})
            write_md(summaries)
    write_md(summaries)
    log(f"DEEP DONE symbols={len(summaries)} any_q={any(s.get('qualifiers',0)>0 for s in summaries)}")


if __name__ == "__main__":
    main()
