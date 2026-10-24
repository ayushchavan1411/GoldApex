#!/usr/bin/env python3
"""SL-hit learning hunter.

1) Replay top strategies with trade-level simulate_trades (SL|TP|TIME).
2) Autopsy why SLs hit (hour/session/RSI/ADX/trend/ATR/MFE).
3) Derive avoidance filters from SL vs TP feature gaps.
4) Re-backtest existing strategies WITH those filters (AND), and OR-unions of
   improved variants — hunt ≥75% WR + ≥1 tpd under RR≥1.0.

Never stops. Writes SL_AUTOPSY.json + SL_LEARN_LEADERBOARD.jsonl.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from collections import defaultdict
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
    simulate_trades,
    write_bot,
    year_stable,
)

RNG = random.Random(19)
AUTOPSY = OUT / "SL_AUTOPSY.json"
BOARD = OUT / "SL_LEARN_LEADERBOARD.jsonl"
PROG = OUT / "sl_learn_progress.json"

EXITS = [
    {"rr": rr, "max_hold": mh, "atr": atr, "allow_long": al, "allow_short": ash}
    for rr in (1.0, 1.2)
    for mh in (4, 6, 8, 12)
    for atr in ("ATR_14", "ATR_7")
    for al, ash in ((True, True), (True, False), (False, True))
]


def _clean(L, S):
    L = np.asarray(L, dtype=bool)
    S = np.asarray(S, dtype=bool)
    both = L & S
    return L & ~both, S & ~both


def strategy_bank(df5, df1=None):
    """Known strong / high-freq seeds to autopsy and improve."""
    bank = []
    # High-WR FVG+RSI family
    for rsi, os_, ob, sess in itertools.product(
        [7, 14], [35, 40], [65, 75, 80],
        [("overlap", "ny"), ("london", "overlap"), ("asian", "london", "overlap", "ny"), None],
    ):
        bank.append(("5m", "fvg_rsi", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}, df5))
    # RSI+MACD
    for rsi, os_, ob, sess in itertools.product(
        [7, 14], [25, 35, 40, 45], [55, 65, 70],
        [None, ("london", "overlap", "ny"), ("asian",), ("overlap", "ny")],
    ):
        bank.append(("5m", "rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}, df5))
    # BB+RSI fade
    for (period, k), rsi, os_, ob, sess in itertools.product(
        [(20, 2.0), (20, 2.5)], [7, 14], [25, 30], [70, 75],
        [None, ("overlap", "ny"), ("london", "overlap")],
    ):
        bank.append(("5m", "bb_rsi_fade", {"period": period, "k": k, "rsi": rsi, "os": os_, "ob": ob, "sessions": sess}, df5))
    # ICT / SMC cores
    for kind, p in [
        ("fvg_reclaim", {"sessions": ("london", "overlap", "ny")}),
        ("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": ("london", "overlap", "ny")}),
        ("smc_confluence", {"swing": 20, "sessions": ("overlap", "ny")}),
        ("ob_retest", {"sessions": ("london", "overlap")}),
        ("ob_fvg", {"sessions": ("overlap", "ny")}),
        ("mss_entry", {"swing": 20, "sessions": ("london", "overlap", "ny")}),
        ("bos_pullback", {"swing": 20, "ema": 21, "bos_look": 12, "sessions": ("london", "overlap")}),
        ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": None}),
        ("rsi_cross", {"rsi": 7, "os": 25, "ob": 75, "sessions": ("london", "overlap", "ny")}),
        ("stoch_cross", {"period": 14, "os": 20, "ob": 80, "sessions": None}),
    ]:
        bank.append(("5m", kind, p, df5))
    if df1 is not None:
        for rsi, os_, ob, sess in itertools.product(
            [14], [35, 40], [65, 70],
            [None, ("london", "overlap", "ny")],
        ):
            bank.append(("1m", "rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}, df1))
    return bank


def autopsy_strategy(name, df, L, S, exit_p):
    n = len(df)
    split = int(n * 0.7)
    warm = 500 if n > 50000 else 250
    common = dict(
        long_e=L, short_e=S, rr_atr=exit_p["rr"], max_hold=exit_p["max_hold"],
        atr_col=exit_p.get("atr", "ATR_14"),
        allow_long=exit_p.get("allow_long", True),
        allow_short=exit_p.get("allow_short", True),
    )
    trades = simulate_trades(df, start_i=warm, end_i=n - 2, **common)
    if len(trades) < 15:
        return None
    sl = [t for t in trades if t["reason"] == "SL"]
    tp = [t for t in trades if t["reason"] == "TP"]
    tm = [t for t in trades if t["reason"] == "TIME"]

    def avg(rows, key):
        vals = [t[key] for t in rows if t.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def rate(rows, pred):
        if not rows:
            return None
        return float(np.mean([1.0 if pred(t) else 0.0 for t in rows]))

    summary = {
        "name": name,
        "n": len(trades),
        "sl": len(sl),
        "tp": len(tp),
        "time": len(tm),
        "sl_rate": len(sl) / len(trades),
        "tp_rate": len(tp) / len(trades),
        "sl_avg_bars": avg(sl, "bars"),
        "tp_avg_bars": avg(tp, "bars"),
        "sl_avg_mfe": avg(sl, "mfe_r"),  # how far toward TP before dying
        "tp_avg_mae": avg(tp, "mae_r"),
        "sl_with_trend": rate(sl, lambda t: t.get("with_trend") is True),
        "tp_with_trend": rate(tp, lambda t: t.get("with_trend") is True),
        "sl_avg_adx": avg(sl, "adx"),
        "tp_avg_adx": avg(tp, "adx"),
        "sl_avg_rsi14": avg(sl, "rsi14"),
        "tp_avg_rsi14": avg(tp, "rsi14"),
        "sl_avg_atr_vs_ma": avg(sl, "atr_vs_ma"),
        "tp_avg_atr_vs_ma": avg(tp, "atr_vs_ma"),
        "sl_hour_hist": dict(sorted(
            ((h, sum(1 for t in sl if t["hour"] == h)) for h in range(24)),
            key=lambda x: -x[1],
        )[:8]),
        "tp_hour_hist": dict(sorted(
            ((h, sum(1 for t in tp if t["hour"] == h)) for h in range(24)),
            key=lambda x: -x[1],
        )[:8]),
        "sl_session": {s: sum(1 for t in sl if t["session"] == s) for s in sorted({t["session"] for t in sl})},
        "lessons": [],
    }

    # Derive lessons (feature gaps SL vs TP)
    lessons = []
    if summary["sl_with_trend"] is not None and summary["tp_with_trend"] is not None:
        if summary["sl_with_trend"] + 0.12 < summary["tp_with_trend"]:
            lessons.append({"rule": "require_with_trend", "why": "SL trades were more often against EMA21/50"})
        if summary["sl_with_trend"] > summary["tp_with_trend"] + 0.12:
            lessons.append({"rule": "prefer_countertrend", "why": "mean-reversion: SL more often with-trend"})
    if summary["sl_avg_adx"] and summary["tp_avg_adx"]:
        if summary["sl_avg_adx"] + 3 < summary["tp_avg_adx"]:
            lessons.append({"rule": "adx_min_22", "why": f"SL ADX {summary['sl_avg_adx']:.1f} < TP {summary['tp_avg_adx']:.1f}"})
        if summary["sl_avg_adx"] > summary["tp_avg_adx"] + 4:
            lessons.append({"rule": "adx_max_28", "why": f"SL in stronger trend ADX {summary['sl_avg_adx']:.1f}"})
    if summary["sl_avg_atr_vs_ma"] and summary["tp_avg_atr_vs_ma"]:
        if summary["sl_avg_atr_vs_ma"] > summary["tp_avg_atr_vs_ma"] + 0.15:
            lessons.append({"rule": "atr_not_expanded", "why": "SL more common when ATR > rolling mean (news/spike)"})
        if summary["sl_avg_atr_vs_ma"] + 0.12 < summary["tp_avg_atr_vs_ma"]:
            lessons.append({"rule": "atr_expanded", "why": "TP preferred expanded ATR"})
    if summary["sl_avg_mfe"] is not None and summary["sl_avg_mfe"] >= 0.55:
        lessons.append({"rule": "breakeven_style_note", "why": f"SL often after MFE {summary['sl_avg_mfe']:.2f}R — entry OK, hold/filter issue"})
    # Hour poison: hours where SL>>TP
    sl_h = defaultdict(int)
    tp_h = defaultdict(int)
    for t in sl:
        sl_h[t["hour"]] += 1
    for t in tp:
        tp_h[t["hour"]] += 1
    poison = []
    for h in range(24):
        s, p = sl_h[h], tp_h[h]
        if s >= 4 and s >= 1.6 * max(p, 1):
            poison.append(h)
    if poison:
        lessons.append({"rule": "avoid_hours", "hours": poison, "why": f"SL-heavy hours {poison}"})
    # Session poison
    sl_s = defaultdict(int)
    tp_s = defaultdict(int)
    for t in sl:
        sl_s[t["session"]] += 1
    for t in tp:
        tp_s[t["session"]] += 1
    bad_sess = [s for s in sl_s if sl_s[s] >= 4 and sl_s[s] >= 1.5 * max(tp_s[s], 1)]
    if bad_sess:
        lessons.append({"rule": "avoid_sessions", "sessions": bad_sess, "why": f"SL-heavy sessions {bad_sess}"})
    # RSI extremes for fades vs momentum
    if summary["sl_avg_rsi14"] and summary["tp_avg_rsi14"]:
        # shorts dying with lower RSI => chasing; longs dying with higher RSI
        lessons.append({
            "rule": "rsi_sanity",
            "sl_rsi": summary["sl_avg_rsi14"],
            "tp_rsi": summary["tp_avg_rsi14"],
            "why": "compare RSI at SL vs TP entries",
        })
    # Fast SL (<2 bars) => bad entry timing
    fast_sl = sum(1 for t in sl if t["bars"] <= 2)
    if sl and fast_sl / len(sl) >= 0.45:
        lessons.append({"rule": "skip_weak_impulse", "why": f"{fast_sl/len(sl):.0%} of SLs within 2 bars"})

    summary["lessons"] = lessons
    summary["sample_sl"] = sl[:8]
    summary["sample_tp"] = tp[:5]
    return summary


def build_global_filters(df, autopsy_rows):
    """Aggregate lessons → boolean mask factories (long_mask, short_mask)."""
    # Count rule popularity
    rule_votes = defaultdict(int)
    avoid_hours = defaultdict(int)
    avoid_sess = defaultdict(int)
    for row in autopsy_rows:
        for les in row.get("lessons") or []:
            rule_votes[les["rule"]] += 1
            if les["rule"] == "avoid_hours":
                for h in les.get("hours") or []:
                    avoid_hours[h] += 1
            if les["rule"] == "avoid_sessions":
                for s in les.get("sessions") or []:
                    avoid_sess[s] += 1

    filters = []
    # Always include structural filters suggested by aggregate vote
    ema_bull = (df["EMA_21"] > df["EMA_50"]).to_numpy()
    adx = df["ADX"].to_numpy()
    atr = df["ATR_14"]
    atr_ma = atr.rolling(50).mean()
    atr_ratio = (atr / atr_ma).fillna(1.0).to_numpy()
    hour = df.index.hour
    rsi = df["RSI_14"].to_numpy()
    macdh = df["MACD_HIST"].to_numpy()
    body = (df["Close"] - df["Open"]).abs().to_numpy()
    rng = (df["High"] - df["Low"]).replace(0, np.nan).to_numpy()
    body_frac = body / np.where(np.isfinite(rng) & (rng > 0), rng, 1.0)

    filters.append(("with_trend", ema_bull, ~ema_bull))
    filters.append(("adx_min_22", adx > 22, adx > 22))
    filters.append(("adx_min_18", adx > 18, adx > 18))
    filters.append(("adx_max_30", adx < 30, adx < 30))
    filters.append(("atr_not_expanded", atr_ratio < 1.35, atr_ratio < 1.35))
    filters.append(("atr_expanded", atr_ratio > 1.05, atr_ratio > 1.05))
    filters.append(("macd_align", macdh > 0, macdh < 0))
    filters.append(("rsi_long_os", rsi < 45, rsi > 55))
    filters.append(("rsi_not_chase", rsi < 60, rsi > 40))
    filters.append(("skip_tiny_body", body_frac > 0.25, body_frac > 0.25))
    filters.append(("skip_huge_body", body_frac < 0.85, body_frac < 0.85))

    # Poison hours (voted)
    poison_h = [h for h, c in avoid_hours.items() if c >= 2]
    if not poison_h:
        # gold typical: avoid thin Asian open + late NY if often SL
        poison_h = [0, 1, 2, 21, 22, 23]
    ok_h = ~np.isin(hour, poison_h)
    filters.append((f"avoid_hours_{poison_h}", ok_h, ok_h))

    # London/NY only (common good sessions)
    sess = df["session"].to_numpy()
    for name, allowed in [
        ("lon_ny", {"london", "overlap", "ny"}),
        ("overlap_ny", {"overlap", "ny"}),
        ("london_overlap", {"london", "overlap"}),
    ]:
        m = np.isin(sess, list(allowed))
        filters.append((name, m, m))

    # Combo packs from strongest lessons
    filters.append(("pack_trend_adx", ema_bull & (adx > 20) & (atr_ratio < 1.4),
                    (~ema_bull) & (adx > 20) & (atr_ratio < 1.4)))
    filters.append(("pack_fade_safe", (rsi < 40) & (atr_ratio < 1.3) & (adx < 32),
                    (rsi > 60) & (atr_ratio < 1.3) & (adx < 32)))
    filters.append(("pack_no_poison_h", ok_h & (adx > 18), ok_h & (adx > 18)))

    log(f"SL filters built={len(filters)} poison_hours={poison_h} rule_votes={dict(rule_votes)}")
    return filters, {"rule_votes": dict(rule_votes), "poison_hours": poison_h, "poison_sessions": dict(avoid_sess)}


def run_autopsy(frames):
    log("=== SL AUTOPSY: replaying top strategies for SL/TP context ===")
    df5 = frames["5m"]
    df1 = frames.get("1m")
    bank = strategy_bank(df5, df1)
    rows = []
    exits_probe = [
        {"rr": 1.0, "max_hold": 6, "atr": "ATR_14", "allow_long": True, "allow_short": True},
        {"rr": 1.2, "max_hold": 8, "atr": "ATR_7", "allow_long": True, "allow_short": True},
        {"rr": 1.2, "max_hold": 6, "atr": "ATR_7", "allow_long": False, "allow_short": True},
        {"rr": 1.2, "max_hold": 12, "atr": "ATR_14", "allow_long": True, "allow_short": True},
    ]
    for tf, kind, p, df in bank:
        try:
            L, S = gen_signals(df, kind, p)
        except Exception:
            continue
        L, S = _clean(L, S)
        if int(L.sum() + S.sum()) < 40:
            continue
        for ex in exits_probe:
            name = f"{kind}|{tf}|rr{ex['rr']}|h{ex['max_hold']}|{p}"
            summ = autopsy_strategy(name[:120], df, L, S, ex)
            if summ and summ["sl"] >= 5:
                rows.append(summ)
    rows.sort(key=lambda r: (-r["sl_rate"], -r["n"]))
    # Aggregate lessons
    filters, meta = build_global_filters(df5, rows[:80])
    report = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategies_autopsied": len(rows),
        "meta": meta,
        "top_sl_rate": [
            {k: r[k] for k in ("name", "n", "sl", "tp", "sl_rate", "sl_avg_mfe", "sl_with_trend",
                               "tp_with_trend", "sl_avg_adx", "tp_avg_adx", "lessons") if k in r}
            for r in rows[:25]
        ],
        "global_lessons": meta,
    }
    AUTOPSY.write_text(json.dumps(report, indent=2, default=str))
    log(f"autopsy done strategies={len(rows)} → {AUTOPSY}")
    # Print key lessons
    for r in rows[:8]:
        log(f"SL {r['sl_rate']:.0%} n={r['n']} mfe_before_SL={r.get('sl_avg_mfe')} "
            f"withTrend SL/TP={r.get('sl_with_trend')}/{r.get('tp_with_trend')} | {r['name'][:70]}")
        for les in r.get("lessons") or []:
            log(f"  lesson: {les.get('rule')} — {les.get('why')}")
    return rows, filters, meta


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
        min_wr >= 0.58 and min_tpd >= 0.9
    ):
        with open(BOARD, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "tag": tag, "name": res.name, "wr": res.winrate, "oos_wr": res.oos_winrate,
                "tpd": res.trades_per_day, "oos_tpd": res.oos_trades_per_day,
                "n": res.trades, "oos_n": res.oos_trades, "pf": res.profit_factor,
                "oos_pf": res.oos_profit_factor, "params": res.params, "tf": res.timeframe,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, default=str) + "\n")
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
            log(f"{'QUAL ' if q else ''}SLLEARN {tag} WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}")


def main():
    OUT.mkdir(exist_ok=True)
    log("=== RR1 SL-LEARN SEARCH START ===")
    # 5m only here to limit RAM under multi-hunter load (1m autopsy optional later)
    raw = load_frames(include_resampled=False)
    frames = {"5m": add_indicators(raw["5m"])}
    log(f"loaded 5m={len(frames['5m'])}")

    tested = [0]
    quals = [0]
    best = [None]

    while True:
        rows, filters, meta = run_autopsy(frames)
        df5 = frames["5m"]
        bank = strategy_bank(df5, frames.get("1m"))
        RNG.shuffle(bank)
        log(f"=== SL-LEARN re-backtest bank={len(bank)} filters={len(filters)} ===")

        # Phase 1: each strategy × each SL-avoidance filter
        for bi, (tf, kind, p, df) in enumerate(bank):
            try:
                L0, S0 = gen_signals(df, kind, p)
            except Exception:
                continue
            L0, S0 = _clean(L0, S0)
            if int(L0.sum() + S0.sum()) < 50:
                continue
            # baseline
            for ex in EXITS[::2]:
                res = evaluate(f"slbase|{kind}|{tf}", tf, df, L0, S0, ex, {"kind": kind, "stage": "base", **p})
                consider(res, L0, S0, ex, df, tested, quals, best, "BASE")
            # filtered — only 5m filters align; for 1m rebuild simple masks
            if tf == "5m":
                flist = filters
            else:
                # light 1m masks
                ema_bull = (df["EMA_21"] > df["EMA_50"]).to_numpy()
                adx = df["ADX"].to_numpy()
                hour = df.index.hour
                flist = [
                    ("with_trend", ema_bull, ~ema_bull),
                    ("adx_min_20", adx > 20, adx > 20),
                    ("avoid_thin_h", ~np.isin(hour, [0, 1, 2, 3, 22, 23]), ~np.isin(hour, [0, 1, 2, 3, 22, 23])),
                ]
            for fname, mL, mS in flist:
                L, S = _clean(L0 & mL, S0 & mS)
                if int(L.sum() + S.sum()) < 45:
                    continue
                for ex in EXITS:
                    res = evaluate(
                        f"sl|{kind}+{fname}|{tf}", tf, df, L, S, ex,
                        {"kind": "sl_filtered", "base": kind, "filter": fname, **p},
                    )
                    consider(res, L, S, ex, df, tested, quals, best, f"FILT-{fname}")
            # dual filter stacks
            for (f1, aL, aS), (f2, bL, bS) in itertools.combinations(flist[:10], 2):
                L, S = _clean(L0 & aL & bL, S0 & aS & bS)
                if int(L.sum() + S.sum()) < 40:
                    continue
                for ex in EXITS[::3]:
                    res = evaluate(
                        f"sl2|{kind}+{f1}+{f2}|{tf}", tf, df, L, S, ex,
                        {"kind": "sl_dual", "base": kind, "f1": f1, "f2": f2, **p},
                    )
                    consider(res, L, S, ex, df, tested, quals, best, "DUAL")

            if (bi + 1) % 10 == 0:
                PROG.write_text(json.dumps({
                    "tested": tested[0], "quals": quals[0], "bi": bi + 1,
                    "best": None if not best[0] else {
                        "wr": best[0].winrate, "oos_wr": best[0].oos_winrate,
                        "tpd": best[0].trades_per_day, "oos_tpd": best[0].oos_trades_per_day,
                        "name": best[0].name,
                    },
                    "meta": meta,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, indent=2))
                log(f"sl-learn bi={bi+1}/{len(bank)} tested={tested[0]} quals={quals[0]}")

        # Phase 2: OR-combine improved high-WR low-tpd filtered signals to lift frequency
        log("=== SL-LEARN OR-boost of filtered high-WR seeds ===")
        improved = []
        for tf, kind, p, df in bank:
            if tf != "5m":
                continue
            try:
                L0, S0 = gen_signals(df, kind, p)
            except Exception:
                continue
            for fname, mL, mS in filters[:8]:
                L, S = _clean(L0 & mL, S0 & mS)
                nsig = int(L.sum() + S.sum())
                if nsig < 60:
                    continue
                # quick score at 1:1
                res = evaluate(f"seed|{kind}+{fname}", "5m", df, L, S,
                               {"rr": 1.2, "max_hold": 8, "atr": "ATR_14", "allow_long": True, "allow_short": True},
                               {"kind": kind, "filter": fname})
                if res and min(res.winrate, res.oos_winrate) >= 0.62 and res.trades >= 25:
                    improved.append((kind, fname, L, S, res))
        improved.sort(key=lambda x: min(x[4].winrate, x[4].oos_winrate), reverse=True)
        improved = improved[:40]
        log(f"improved seeds={len(improved)}")
        for (a, fa, La, Sa, _), (b, fb, Lb, Sb, _) in itertools.combinations(improved[:25], 2):
            L, S = _clean(La | Lb, Sa | Sb)
            if int(L.sum() + S.sum()) < 70:
                continue
            for ex in EXITS[::2]:
                res = evaluate(f"slOR|{a}+{b}", "5m", df5, L, S, ex,
                               {"kind": "sl_or", "a": a, "b": b, "fa": fa, "fb": fb})
                consider(res, L, S, ex, df5, tested, quals, best, "OR")

        log(f"SL-LEARN pass done tested={tested[0]} quals={quals[0]}; re-autopsy+loop")
        PROG.write_text(json.dumps({
            "tested": tested[0], "quals": quals[0], "stage": "loop",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
        RNG.seed(int(time.time()) % 100000)


if __name__ == "__main__":
    main()
