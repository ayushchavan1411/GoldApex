#!/usr/bin/env python3
"""Cross-asset screen: download FX / oil-ETF / metal-ETFs / US-index ETFs,
then backtest our best gold RR=1:1 strategies with adaptive MIN_ATR.

Goal gate (same as gold): IS+OOS WR≥75%, tpd≥1.0, rr≥1.0, n≥30/15, PF≥1.0, net>0.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import rr1_search as R
from fetch_twelvedata import download
from rr1_sl_learn_search import build_global_filters

OUT = Path("rr1_search_results")
OUT.mkdir(exist_ok=True)
DATA = Path("data")

# Free-plan accessible proxies (cash metals / cash indices / crude futures locked).
SYMBOLS = [
    # FX majors
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CAD",
    "USD/CHF",
    "NZD/USD",
    # US index ETFs (SPX/NDX cash locked on free plan)
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    # Oil / metals proxies (USOIL / XAG cash locked)
    "USO",  # United States Oil Fund → crude proxy
    "GLD",  # gold ETF
    "SLV",  # silver ETF
]

ASSET_CLASS = {
    "EUR/USD": "fx",
    "GBP/USD": "fx",
    "USD/JPY": "fx",
    "AUD/USD": "fx",
    "USD/CAD": "fx",
    "USD/CHF": "fx",
    "NZD/USD": "fx",
    "SPY": "us_index_etf",
    "QQQ": "us_index_etf",
    "DIA": "us_index_etf",
    "IWM": "us_index_etf",
    "USO": "oil_etf",
    "GLD": "metal_etf",
    "SLV": "metal_etf",
}


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)


def find_csv(symbol: str, interval: str = "5min") -> Path | None:
    stem = f"{symbol.replace('/', '')}_{interval}_"
    cands = sorted(
        [p for p in DATA.glob(f"{stem}*.csv") if "PARTIAL" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def ensure_data(symbol: str, years: float, sleep_s: float, force: bool = False) -> Path:
    existing = find_csv(symbol)
    if existing and not force:
        df = pd.read_csv(existing, parse_dates=["datetime"])
        span = (df["datetime"].max() - df["datetime"].min()).days
        if span >= int(years * 365 * 0.85) and len(df) > 5000:
            log(f"reuse {existing.name} bars={len(df)} days={span}")
            return existing
        log(f"extend {symbol}: have days={span} bars={len(df)}")
    return download(symbol, "5min", years=years, sleep_s=sleep_s)


def load_symbol_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime").sort_index()
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df[~df.index.duplicated(keep="last")]
    return R.add_indicators(df)


def adaptive_min_atr(df: pd.DataFrame) -> float:
    """Reject near-flat quote noise relative to this asset's own ATR scale."""
    atr = df["ATR_14"].replace(0, np.nan).dropna()
    if atr.empty:
        return 1e-9
    med = float(atr.median())
    p10 = float(np.nanpercentile(atr.to_numpy(), 10))
    # Floor ~15% of median ATR, but not above 40% of p10 (keep most real bars).
    floor = max(med * 0.15, p10 * 0.35, 1e-12)
    return float(floor)


def apply_filters(L, S, fl_mask, fs_mask):
    L = np.asarray(L, dtype=bool) & np.asarray(fl_mask, dtype=bool)
    S = np.asarray(S, dtype=bool) & np.asarray(fs_mask, dtype=bool)
    both = L & S
    return L & ~both, S & ~both


def strategy_bank():
    """Best gold seeds + champion dual-filter path."""
    bank = []
    # Champion dual path
    bank.append(
        (
            "champ_fvg_rsi_dual",
            "fvg_rsi",
            {"rsi": 14, "os": 40, "ob": 80, "sessions": None},
            {"rr": 1.2, "max_hold": 12, "atr": "ATR_14", "allow_long": True, "allow_short": True},
            ["atr_expanded", "skip_tiny_body"],
        )
    )
    # High-WR FVG+RSI family
    for rsi, os_, ob, sess, rr, mh in [
        (14, 40, 80, None, 1.2, 12),
        (14, 40, 75, ("overlap", "ny"), 1.0, 8),
        (7, 35, 65, ("london", "overlap", "ny"), 1.0, 6),
        (14, 35, 70, ("london", "overlap"), 1.2, 8),
    ]:
        bank.append(
            (
                f"fvg_rsi_r{rsi}_{os_}_{ob}",
                "fvg_rsi",
                {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess},
                {"rr": rr, "max_hold": mh, "atr": "ATR_14", "allow_long": True, "allow_short": True},
                [],
            )
        )
    # RSI+MACD (often higher tpd on gold)
    for rsi, os_, ob, sess, rr, mh in [
        (14, 35, 65, None, 1.0, 6),
        (7, 25, 70, ("london", "overlap", "ny"), 1.0, 8),
        (14, 40, 60, ("overlap", "ny"), 1.2, 8),
        (7, 30, 70, None, 1.0, 4),
    ]:
        bank.append(
            (
                f"rsi_macd_r{rsi}_{os_}_{ob}",
                "rsi_macd",
                {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess},
                {"rr": rr, "max_hold": mh, "atr": "ATR_14", "allow_long": True, "allow_short": True},
                [],
            )
        )
    # Focused champion-style rsi_cross (higher tpd on gold, weaker OOS)
    bank.append(
        (
            "rsi_cross_focus",
            "rsi_cross",
            {"rsi": 7, "os": 30, "ob": 75, "sessions": ("overlap", "ny")},
            {"rr": 1.5, "max_hold": 24, "atr": "ATR_14", "allow_long": True, "allow_short": True},
            [],
        )
    )
    # ICT / SMC cores
    for name, kind, p, ex in [
        ("fvg_reclaim", "fvg_reclaim", {"sessions": ("london", "overlap", "ny")},
         {"rr": 1.0, "max_hold": 8, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
        ("liq_sweep", "liq_sweep", {"swing": 20, "wick": 0.45, "sessions": ("london", "overlap", "ny")},
         {"rr": 1.0, "max_hold": 8, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
        ("smc_confluence", "smc_confluence", {"swing": 20, "sessions": ("overlap", "ny")},
         {"rr": 1.2, "max_hold": 8, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
        ("ob_fvg", "ob_fvg", {"sessions": ("overlap", "ny")},
         {"rr": 1.0, "max_hold": 6, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
        ("mss_entry", "mss_entry", {"swing": 20, "sessions": ("london", "overlap", "ny")},
         {"rr": 1.0, "max_hold": 8, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
        ("bb_rsi_fade", "bb_rsi_fade",
         {"period": 20, "k": 2.0, "rsi": 14, "os": 30, "ob": 70, "sessions": ("overlap", "ny")},
         {"rr": 1.0, "max_hold": 6, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
        ("stoch_cross", "stoch_cross",
         {"period": 5, "os": 25, "ob": 80, "sessions": ("asian",)},
         {"rr": 1.0, "max_hold": 6, "atr": "ATR_14", "allow_long": True, "allow_short": False}),
        ("ema_pull_rsi", "ema_pull_rsi",
         {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": None},
         {"rr": 1.0, "max_hold": 8, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
    ]:
        bank.append((name, kind, p, ex, []))
    # Dual-filter variants of top kinds
    for base_name, kind, p, ex in [
        ("fvg_rsi", "fvg_rsi", {"rsi": 14, "os": 40, "ob": 80, "sessions": None},
         {"rr": 1.2, "max_hold": 12, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
        ("rsi_macd", "rsi_macd", {"rsi": 14, "os": 35, "ob": 65, "sessions": None},
         {"rr": 1.0, "max_hold": 6, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
        ("bb_rsi_fade", "bb_rsi_fade",
         {"period": 20, "k": 2.0, "rsi": 14, "os": 30, "ob": 70, "sessions": None},
         {"rr": 1.0, "max_hold": 6, "atr": "ATR_14", "allow_long": True, "allow_short": True}),
    ]:
        for fpair in (
            ["atr_expanded", "skip_tiny_body"],
            ["with_trend", "adx_min_18"],
            ["pack_trend_adx"],
            ["lon_ny", "skip_tiny_body"],
            ["rsi_not_chase", "atr_not_expanded"],
        ):
            bank.append((f"dual|{base_name}+{'+'.join(fpair)}", kind, p, ex, fpair))
    return bank


def qualifies_strict(res) -> bool:
    return (
        res.trades >= 30
        and res.oos_trades >= 15
        and res.winrate >= 0.75
        and res.oos_winrate >= 0.75
        and res.trades_per_day >= 1.0
        and res.oos_trades_per_day >= 1.0
        and res.profit_factor >= 1.0
        and res.oos_profit_factor >= 1.0
        and res.net_pnl > 0
        and res.oos_net_pnl > 0
        and res.params.get("exit", {}).get("rr", 0) >= 1.0
    )


def near_miss(res) -> bool:
    """Close: high WR both sides, or high tpd with WR≥70%."""
    wr_ok = res.winrate >= 0.70 and res.oos_winrate >= 0.70 and res.trades >= 25 and res.oos_trades >= 12
    tpd_ok = res.trades_per_day >= 0.8 and res.oos_trades_per_day >= 0.8
    wr_hi = res.winrate >= 0.75 and res.oos_winrate >= 0.75 and res.trades_per_day >= 0.3
    return (wr_ok and tpd_ok) or wr_hi


def backtest_symbol(symbol: str, path: Path) -> dict:
    log(f"=== backtest {symbol} ({ASSET_CLASS.get(symbol, '?')}) path={path.name} ===")
    df = load_symbol_df(path)
    min_atr = adaptive_min_atr(df)
    old = R.MIN_ATR
    R.MIN_ATR = min_atr
    try:
        atr_med = float(df["ATR_14"].median())
        days = max(1, (df.index[-1] - df.index[0]).days + 1)
        log(
            f"{symbol}: bars={len(df)} days={days} "
            f"{df.index[0]}→{df.index[-1]} ATR_med={atr_med:.6g} MIN_ATR={min_atr:.6g}"
        )
        # Build filter masks once (uses gold autopsy lessons for dual path)
        filters, _meta = build_global_filters(df, [])
        fmap = {name: (fl, fs) for name, fl, fs in filters}

        results = []
        quals = []
        nears = []
        for tag, kind, p, ex, flist in strategy_bank():
            try:
                L, S = R.gen_signals(df, kind, p)
            except Exception as e:
                log(f"  skip {tag}: gen_signals {e}")
                continue
            L = np.asarray(L, dtype=bool)
            S = np.asarray(S, dtype=bool)
            both = L & S
            L, S = L & ~both, S & ~both
            for fname in flist:
                if fname not in fmap:
                    continue
                fl, fs = fmap[fname]
                L, S = apply_filters(L, S, fl, fs)
            if int(L.sum() + S.sum()) < 25:
                continue
            # Also sweep a small exit grid around the seed
            exits = [ex]
            for rr in (1.0, 1.2):
                for mh in (6, 8, 12):
                    e2 = {**ex, "rr": rr, "max_hold": mh}
                    if e2 not in exits:
                        exits.append(e2)
            for e in exits:
                name = f"{symbol}|{tag}|rr{e['rr']}|h{e['max_hold']}"
                res = R.evaluate(name, "5m", df, L, S, e, {"kind": kind, **p, "filters": flist, "symbol": symbol})
                if not res:
                    continue
                row = {
                    "symbol": symbol,
                    "class": ASSET_CLASS.get(symbol),
                    "name": res.name,
                    "kind": kind,
                    "filters": flist,
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
                    "max_dd": res.max_dd,
                    "exit": e,
                    "min_atr": min_atr,
                    "atr_med": atr_med,
                    "days": days,
                    "bars": len(df),
                    "qualifies": qualifies_strict(res),
                    "near": near_miss(res),
                    "score": res.score,
                }
                results.append(row)
                if row["qualifies"]:
                    quals.append(row)
                    log(
                        f"  QUAL {symbol} {tag} WR{res.winrate:.1%}/{res.oos_winrate:.1%} "
                        f"tpd{res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                        f"n{res.trades}/{res.oos_trades}"
                    )
                elif row["near"]:
                    nears.append(row)
                    log(
                        f"  NEAR {symbol} {tag} WR{res.winrate:.1%}/{res.oos_winrate:.1%} "
                        f"tpd{res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                        f"n{res.trades}/{res.oos_trades}"
                    )

        results.sort(key=lambda r: (-(r["wr"] + r["oos_wr"]) / 2, -min(r["tpd"], r["oos_tpd"])))
        best = results[0] if results else None
        summary = {
            "symbol": symbol,
            "class": ASSET_CLASS.get(symbol),
            "path": str(path),
            "bars": len(df),
            "days": days,
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "atr_med": atr_med,
            "min_atr": min_atr,
            "tested": len(results),
            "qualifiers": len(quals),
            "near_misses": len(nears),
            "best": best,
            "top5": results[:5],
            "quals": quals[:10],
            "nears": sorted(nears, key=lambda r: (-(r["wr"] + r["oos_wr"]) / 2, -min(r["tpd"], r["oos_tpd"])))[:10],
        }
        return summary
    finally:
        R.MIN_ATR = old


def write_report(summaries: list[dict]):
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "goal": "IS+OOS WR≥75%, tpd≥1.0, rr≥1.0, n≥30/15",
        "note": (
            "Cash XAG/XPT/USOIL/SPX/NDX locked on free Twelve Data plan; "
            "using USO (oil ETF), SLV/GLD (metal ETFs), SPY/QQQ/DIA/IWM (index ETFs)."
        ),
        "symbols": summaries,
        "any_qualifier": any(s.get("qualifiers", 0) > 0 for s in summaries),
    }
    (OUT / "CROSS_ASSET_RESULTS.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# Cross-asset RR=1:1 screen",
        "",
        f"Generated: {payload['time']}",
        "",
        f"**Goal:** {payload['goal']}",
        "",
        f"**Data note:** {payload['note']}",
        "",
        f"**Any full qualifier:** {'YES' if payload['any_qualifier'] else 'NO'}",
        "",
        "| Symbol | Class | Days | ATR med | Best WR IS/OOS | Best tpd IS/OOS | n | Quals | Near |",
        "|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for s in summaries:
        b = s.get("best") or {}
        lines.append(
            f"| {s['symbol']} | {s.get('class')} | {s.get('days')} | "
            f"{s.get('atr_med', 0):.5g} | "
            f"{b.get('wr', 0):.1%}/{b.get('oos_wr', 0):.1%} | "
            f"{b.get('tpd', 0):.2f}/{b.get('oos_tpd', 0):.2f} | "
            f"{b.get('n', 0)}/{b.get('oos_n', 0)} | "
            f"{s.get('qualifiers', 0)} | {s.get('near_misses', 0)} |"
        )
    lines.append("")
    lines.append("## Qualifiers")
    lines.append("")
    quals = [q for s in summaries for q in s.get("quals") or []]
    if not quals:
        lines.append("_None hit the full ≥75% WR + ≥1 tpd IS+OOS gate._")
    else:
        for q in quals:
            lines.append(
                f"- **{q['symbol']}** `{q['name']}` WR {q['wr']:.1%}/{q['oos_wr']:.1%} "
                f"tpd {q['tpd']:.2f}/{q['oos_tpd']:.2f} n={q['n']}/{q['oos_n']}"
            )
    lines.append("")
    lines.append("## Near misses")
    lines.append("")
    nears = [n for s in summaries for n in s.get("nears") or []]
    nears.sort(key=lambda r: (-(r["wr"] + r["oos_wr"]) / 2, -min(r["tpd"], r["oos_tpd"])))
    if not nears:
        lines.append("_No near misses._")
    else:
        for n in nears[:25]:
            lines.append(
                f"- **{n['symbol']}** `{n['kind']}` filters={n['filters']} "
                f"WR {n['wr']:.1%}/{n['oos_wr']:.1%} tpd {n['tpd']:.2f}/{n['oos_tpd']:.2f} "
                f"n={n['n']}/{n['oos_n']} PF {n['pf']:.2f}/{n['oos_pf']:.2f}"
            )
    (OUT / "CROSS_ASSET_RESULTS.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {OUT / 'CROSS_ASSET_RESULTS.md'} quals={len(quals)} nears={len(nears)}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--sleep", type=float, default=8.5)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--backtest-only", action="store_true")
    args = ap.parse_args()
    symbols = args.symbols or SYMBOLS

    paths = {}
    if not args.backtest_only:
        for sym in symbols:
            try:
                if args.skip_download:
                    p = find_csv(sym)
                    if not p:
                        log(f"missing data for {sym}, downloading")
                        p = ensure_data(sym, args.years, args.sleep, force=True)
                else:
                    p = ensure_data(sym, args.years, args.sleep, force=args.force_download)
                paths[sym] = p
            except Exception as e:
                log(f"DOWNLOAD FAIL {sym}: {e}")
                traceback.print_exc()
        (OUT / "CROSS_ASSET_PATHS.json").write_text(
            json.dumps({k: str(v) for k, v in paths.items()}, indent=2)
        )
        if args.download_only:
            log("download-only done")
            return

    if args.backtest_only or args.skip_download:
        for sym in symbols:
            p = find_csv(sym)
            if p:
                paths[sym] = p
            else:
                log(f"no csv for {sym}")

    summaries = []
    for sym, path in paths.items():
        try:
            summaries.append(backtest_symbol(sym, path))
            # checkpoint after each symbol
            write_report(summaries)
        except Exception as e:
            log(f"BACKTEST FAIL {sym}: {e}")
            traceback.print_exc()
            summaries.append({"symbol": sym, "error": str(e), "qualifiers": 0, "near_misses": 0, "best": None})
            write_report(summaries)

    payload = write_report(summaries)
    log(f"DONE any_qualifier={payload['any_qualifier']} symbols={len(summaries)}")


if __name__ == "__main__":
    main()
