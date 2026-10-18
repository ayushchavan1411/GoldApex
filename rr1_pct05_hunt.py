#!/usr/bin/env python3
"""1% risk / 1% reward hunt — NO ATR-based RR.

User (2026-10-18): remove ATR RR; use 1% SL = 1% TP (price), fixed 0.02 lot,
$500 capital, charges included. Goal still WR≥75% IS+OOS, tpd≥1, year_stable.
"""
from __future__ import annotations

import itertools
import json
import random
import time

from rr1_search import (
    COMMISSION_PER_LOT_RT,
    FIXED_LOT,
    OUT,
    PCT_RR,
    SPREAD_PRICE,
    START_CAP,
    TARGET_WR,
    add_indicators,
    evaluate,
    fee_pct_exits,
    gen_signals,
    load_frames,
    log,
    qualifies,
    save_champion,
    simulate,
    write_bot,
    year_stable,
    _exit_sim_kwargs,
)

RNG = random.Random(100)
LEADER = OUT / "PCT1_LEADERBOARD.jsonl"
PROGRESS = OUT / "pct1_progress.json"
SUMMARY = OUT / "PCT1_SUMMARY.json"
REPORT = OUT / "PCT1_CHARGES_REPORT.json"


def exits():
    # Strict 1% only (user: drop ATR RR)
    return fee_pct_exits(PCT_RR, holds=(12, 18, 24, 36, 48, 72, 96))


def specs():
    s = None
    out = []
    for kind, extra in (
        ("fvg_rsi", {"rsi": 14, "os": 40, "ob": 80}),
        ("fvg_rsi", {"rsi": 14, "os": 35, "ob": 65}),
        ("fvg_rsi", {"rsi": 7, "os": 30, "ob": 70}),
        ("fvg_reclaim", {}),
        ("fvg_ce", {}),
        ("turtle_soup", {"period": 20, "wick": 0.35}),
        ("turtle_soup", {"period": 10, "wick": 0.35}),
        ("wyckoff_spring", {"swing": 20, "wick": 0.45}),
        ("wyckoff_upthrust", {"swing": 20, "wick": 0.45}),
        ("inside_break", {}),
        ("fakey", {}),
        ("silver_bullet", {"hours": (14, 15, 16)}),
        ("liq_sweep", {"swing": 20, "wick": 0.45}),
        ("liq_sweep_fvg", {"swing": 20, "wick": 0.45}),
        ("rsi_macd", {"rsi": 14, "os": 40, "ob": 60}),
        ("rsi_level", {"rsi": 14, "os": 35, "ob": 65, "band": 5}),
        ("rsi_cross", {"rsi": 14, "os": 30, "ob": 70}),
        ("donch_fade", {"period": 20, "rsi": 14, "os": 35, "ob": 65}),
        ("orb_fade", {"orb": "LONDON"}),
        ("orb_fade", {"orb": "NY"}),
        ("orb_break", {"orb": "LONDON"}),
        ("of_sweep_absorb", {}),
        ("cisd", {}),
        ("ob_retest", {}),
        ("mss_entry", {"swing": 20}),
        ("eqh_eql_sweep", {"swing": 20, "wick": 0.45}),
        ("pin_rsi", {"rsi": 14, "os": 30, "ob": 70, "wick": 0.55, "body": 0.35}),
        ("engulf", {"ema": 21}),
    ):
        out.append((kind, {**extra, "sessions": s}))
    for sess in (("london", "overlap", "ny"), ("overlap", "ny"), ("london",), ("ny",)):
        out.append(("fvg_rsi", {"rsi": 14, "os": 40, "ob": 80, "sessions": sess}))
        out.append(("turtle_soup", {"period": 20, "wick": 0.35, "sessions": sess}))
        out.append(("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": sess}))
    return out


def dump(tested, quals, best, near, showcase):
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals": quals,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best.get("name") if best else None,
                "best_tpd": best.get("tpd") if best else None,
                "best_net_after_charges": best.get("net_pnl") if best else None,
                "near": len(near),
                "gate": f"1% RR (no ATR) WR>={TARGET_WR:.0%} tpd>=1 lot={FIXED_LOT} after fees",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    SUMMARY.write_text(
        json.dumps(
            {"tested": tested, "quals": quals, "best": best, "near": near[:20], "showcase": showcase},
            indent=2,
            default=str,
        )
    )
    REPORT.write_text(json.dumps(showcase, indent=2, default=str))


def main():
    OUT.mkdir(exist_ok=True)
    log(
        f"=== 1% RISK/REWARD HUNT (NO ATR) lot={FIXED_LOT} cap=${START_CAP:.0f} "
        f"spread=${SPREAD_PRICE} comm=${COMMISSION_PER_LOT_RT}/lot RT pct={PCT_RR:.1%} ==="
    )
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("5m", "15m", "30m"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])

    showcase = {
        "assumptions": {
            "mode": "1%_price_SL_TP_1:1",
            "atr_rr": "REMOVED",
            "start_cap": START_CAP,
            "lot": FIXED_LOT,
            "spread_price_usd": SPREAD_PRICE,
            "commission_per_trade": COMMISSION_PER_LOT_RT * FIXED_LOT,
        },
        "runs": [],
    }
    df5 = frames["5m"]
    n = len(df5)
    split = int(n * 0.7)
    for sk, sp in (
        ("fvg_rsi", {"rsi": 14, "os": 40, "ob": 80, "sessions": None}),
        ("turtle_soup", {"period": 20, "wick": 0.35, "sessions": None}),
        ("ob_retest", {"sessions": None}),
        ("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": None}),
    ):
        L, S = gen_signals(df5, sk, sp)
        ex = fee_pct_exits(PCT_RR, holds=(48,), sides=False)[0]
        kw = _exit_sim_kwargs(ex)
        full = simulate(df5, start_i=250, end_i=n - 2, long_e=L, short_e=S, **kw)
        is_m = simulate(df5, start_i=250, end_i=split, long_e=L, short_e=S, **kw)
        oos_m = simulate(df5, start_i=split, end_i=n - 2, long_e=L, short_e=S, **kw)
        row = {
            "signal": sk,
            "label": "1.0%_of_price_1:1",
            "WR_IS": is_m["winrate"],
            "WR_OOS": oos_m["winrate"],
            "tpd_IS": is_m["trades_per_day"],
            "tpd_OOS": oos_m["trades_per_day"],
            "n_IS": is_m["trades"],
            "n_OOS": oos_m["trades"],
            "net_after_charges_FULL": full["net_pnl"],
            "final_equity_FULL": full["final_equity"],
            "profit_on_500_FULL": full["final_equity"] - START_CAP,
            "total_charges_FULL": full.get("total_charges"),
            "max_dd_FULL": full["max_dd"],
        }
        showcase["runs"].append(row)
        log(
            f"SHOW {sk} 1% WR={is_m['winrate']:.1%}/{oos_m['winrate']:.1%} "
            f"tpd={is_m['trades_per_day']:.2f} "
            f"profit@500=${full['final_equity']-START_CAP:.1f} eq=${full['final_equity']:.1f} "
            f"charges=${full.get('total_charges', 0):.1f}"
        )
    REPORT.write_text(json.dumps(showcase, indent=2, default=str))

    sp = specs()
    exs = exits()
    log(f"specs={len(sp)} exits={len(exs)} tfs={list(frames)} (ATR RR disabled)")

    tested = quals = 0
    best = None
    near: list[dict] = []

    while True:
        RNG.shuffle(sp)
        for kind, base in sp:
            for invert in (False, True):
                pp = {**base}
                if invert:
                    pp["invert"] = True
                for tf, df in frames.items():
                    try:
                        L, S = gen_signals(df, kind, pp)
                    except Exception as e:
                        if tested < 5:
                            log(f"skip {kind}: {e}")
                        continue
                    if int(L.sum() + S.sum()) < 40:
                        continue
                    for ex in exs:
                        tested += 1
                        tag = f"{'inv|' if invert else 'pct1|'}{kind}|1%|{tf}"
                        res = evaluate(tag, tf, df, L, S, dict(ex), {"kind": kind, **pp})
                        if not res or res.trades < 40 or res.oos_trades < 20:
                            continue
                        min_wr = min(res.winrate, res.oos_winrate)
                        yok = year_stable(df, L, S, dict(ex))
                        row = {
                            "name": res.name,
                            "wr": res.winrate,
                            "oos_wr": res.oos_winrate,
                            "min_wr": min_wr,
                            "tpd": res.trades_per_day,
                            "oos_tpd": res.oos_trades_per_day,
                            "n": res.trades,
                            "oos_n": res.oos_trades,
                            "pf": res.profit_factor,
                            "net_pnl": res.net_pnl,
                            "oos_net_pnl": res.oos_net_pnl,
                            "params": res.params,
                            "year_ok": yok,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        sk = (
                            1 if (row["tpd"] >= 1.0 and min_wr >= 0.60 and res.net_pnl > 0) else 0,
                            min_wr,
                            row["tpd"],
                            res.net_pnl,
                        )
                        if best is None or sk > best.get("_sk", (0, 0, 0, 0)):
                            best = {**row, "_sk": sk}
                        if min_wr >= 0.65 and res.trades_per_day >= 0.7 and res.net_pnl > 0:
                            near.append(row)
                            near.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                            near[:] = near[:40]
                            log(
                                f"NEAR pct1 {min_wr:.1%} tpd={res.trades_per_day:.2f} "
                                f"net=${res.net_pnl:.1f} year={yok} | {res.name}"
                            )
                        is_m = {
                            "trades": res.trades,
                            "winrate": res.winrate,
                            "profit_factor": res.profit_factor,
                            "net_pnl": res.net_pnl,
                            "trades_per_day": res.trades_per_day,
                            "max_dd": res.max_dd,
                        }
                        oos_m = {
                            "trades": res.oos_trades,
                            "winrate": res.oos_winrate,
                            "profit_factor": res.oos_profit_factor,
                            "net_pnl": res.oos_net_pnl,
                            "trades_per_day": res.oos_trades_per_day,
                        }
                        if qualifies(is_m, oos_m) and yok:
                            quals += 1
                            if save_champion(res, force=True):
                                write_bot(res)
                            log(
                                f"QUAL PCT1 {res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f} net=${res.net_pnl:.1f} | {res.name}"
                            )
                        if tested % 100 == 0:
                            dump(tested, quals, best, near, showcase)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            log(f"pct1 tested={tested} quals={quals} best={bw} near={len(near)}")

        dump(tested, quals, best, near, showcase)
        log(f"pct1 pass done tested={tested} quals={quals}; reshuffle")


if __name__ == "__main__":
    main()
