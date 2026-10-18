#!/usr/bin/env python3
"""Course-mastery hunt: Silver Bullet, Turtle Soup, Wyckoff, inside-bar, fakey.

Educated from free ICT glossary / Silver Bullet playbooks, BabyPips School of
Pipsology, ORB rule cards, Wyckoff Analytics spring/upthrust notes, and Al Brooks
Always-In / price-action samples. Encodes causal proxies and hunts dual gates:
  Classic RR1: WR≥75% IS/OOS, tpd≥1, PF, year_stable
  Pos-RR:      geoRR≥1.5, WR≥60% IS/OOS, tpd≥1, year_stable
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

from rr1_search import (
    OUT,
    TARGET_WR,
    add_indicators,
    evaluate,
    gen_signals,
    load_frames,
    log,
    qualifies,
    qualifies_posrr,
    save_champion,
    write_bot,
    year_stable,
)

RNG = random.Random(1810)
LEADER = OUT / "COURSE_LEADERBOARD.jsonl"
PROGRESS = OUT / "course_progress.json"
SUMMARY = OUT / "COURSE_SUMMARY.json"
MASTERY = OUT / "STRATEGY_MASTERY.json"


def exits_rr1():
    return [
        {"rr": rr, "max_hold": mh, "atr": atr, **side}
        for rr in (1.0, 1.2)
        for mh in (4, 6, 8, 12, 18, 24)
        for atr in ("ATR_14", "ATR_7")
        for side in (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        )
    ]


def exits_posrr():
    # Favorable RR: tp > sl in ATR units (geoRR = tp/sl)
    out = []
    for sl, tp, mh, atr in itertools.product(
        (1.0, 1.2, 1.5),
        (1.5, 2.0, 2.5, 3.0),
        (6, 8, 12, 18, 24),
        ("ATR_14", "ATR_7"),
    ):
        if tp / sl < 1.5:
            continue
        for side in (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        ):
            out.append(
                {
                    "rr": sl,
                    "sl_atr": sl,
                    "tp_atr": tp,
                    "max_hold": mh,
                    "atr": atr,
                    **side,
                }
            )
    return out


def specs():
    """Course-derived signal grid (sessions + killzone hours)."""
    out = []
    sess_opts = [
        None,
        ("london",),
        ("overlap",),
        ("ny",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
    ]
    # ICT Silver Bullet killzones in UTC (data is tz-naive UTC-style)
    sb_hours = [
        (7, 8, 9),          # London SB ≈ 03–04 NY
        (14, 15),           # NY AM SB EDT
        (15, 16),           # NY AM SB EST
        (14, 15, 16),       # NY AM broader
        (18, 19),           # NY PM SB EDT-ish
        (7, 8, 9, 14, 15, 16),
    ]
    for sess, hours in itertools.product(sess_opts, sb_hours):
        out.append(("silver_bullet", {"sessions": sess, "hours": hours}))
    for period, wick, sess in itertools.product(
        (10, 20, 40), (0.25, 0.35, 0.45, 0.55), sess_opts
    ):
        out.append(("turtle_soup", {"period": period, "wick": wick, "sessions": sess}))
    for sw, wick, sess in itertools.product((10, 20, 40), (0.35, 0.45, 0.55), sess_opts):
        out.append(("wyckoff_spring", {"swing": sw, "wick": wick, "sessions": sess}))
        out.append(("wyckoff_upthrust", {"swing": sw, "wick": wick, "sessions": sess}))
    for sess in sess_opts:
        out.append(("inside_break", {"sessions": sess}))
        out.append(("fakey", {"sessions": sess}))
    # Stack course entries with RSI extremes (champ family style)
    for kind, base in (
        ("turtle_soup", {"period": 20, "wick": 0.35}),
        ("silver_bullet", {"hours": (14, 15, 16)}),
        ("inside_break", {}),
        ("fakey", {}),
    ):
        for sess in (None, ("london", "overlap", "ny"), ("overlap", "ny")):
            out.append((kind, {**base, "sessions": sess}))
    return out


def write_mastery():
    MASTERY.write_text(
        json.dumps(
            {
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sources_studied": [
                    {
                        "name": "ICT Concepts Glossary (Freedom Trading School / public ICT notes)",
                        "concepts": [
                            "FVG", "IFVG", "CE", "OB", "BOS", "MSS", "OTE", "PDH/PDL",
                            "EQH/EQL", "AMD", "BPR", "CISD", "NDOG/NWOG", "Silver Bullet",
                            "liquidity sweep", "premium/discount",
                        ],
                    },
                    {
                        "name": "ICT Silver Bullet AM playbook (free public summaries)",
                        "concepts": [
                            "NY AM 10-11 killzone", "London SB", "NY PM SB",
                            "prior-hour liquidity", "FVG entry after sweep",
                        ],
                    },
                    {
                        "name": "Opening Range Breakout rule cards (Benny-style / public ORB)",
                        "concepts": [
                            "OR window", "break", "fade", "retest", "first-only",
                            "walk-forward / stress framing",
                        ],
                    },
                    {
                        "name": "BabyPips School of Pipsology (free)",
                        "concepts": [
                            "support/resistance", "candlestick patterns", "engulfing",
                            "pin bar", "inside bar", "trend vs range", "risk:reward",
                            "position sizing basics",
                        ],
                    },
                    {
                        "name": "Wyckoff Analytics free PDF notes (springs / upthrusts)",
                        "concepts": [
                            "accumulation spring", "distribution upthrust",
                            "false break of range", "effort vs result",
                        ],
                    },
                    {
                        "name": "Al Brooks Always-In / price-action free samples",
                        "concepts": [
                            "always-in long/short", "inside bar breakout",
                            "failed breakout", "signal bar", "follow-through",
                        ],
                    },
                    {
                        "name": "Linda Raschke Turtle Soup (public write-ups)",
                        "concepts": [
                            "failed 20-bar Donchian breakout", "fade the false break",
                        ],
                    },
                    {
                        "name": "Price-action fakey (Nison / Brooks public notes)",
                        "concepts": [
                            "inside-bar false break", "reversal through mother range",
                        ],
                    },
                ],
                "mastered_and_encoded": {
                    "already_in_engine_before_this_pass": [
                        "rsi_cross/level", "bb_fade/bb_rsi_fade", "stoch_cross",
                        "ema_wick/stack/cross/pull", "pin_rsi", "engulf", "macd_cross",
                        "donch_fade", "turtle_break", "fvg_reclaim/fvg_rsi/fvg_ce/ifvg",
                        "liq_sweep/liq_sweep_fvg", "ob_retest/ob_fvg", "bos_pullback",
                        "mss_entry", "fib_bounce/discount/OTE", "smc_confluence",
                        "pdh_pdl/pwh_pwl", "ndog/nwog", "eqh_eql_sweep", "breaker",
                        "cisd", "bpr", "poi_stack", "amd_sweep_mss",
                        "orb_break/fade/retest/of", "ichimoku", "keltner", "zscore",
                        "cci", "adx_di", "supertrend", "heikin",
                        "orderflow proxies (CLV/CVD/imb/absorb/sweep)",
                    ],
                    "newly_encoded_this_pass": [
                        "silver_bullet",
                        "turtle_soup",
                        "wyckoff_spring",
                        "wyckoff_upthrust",
                        "inside_break",
                        "fakey",
                    ],
                },
                "goal": {
                    "classic_rr1": "WR≥75% IS+OOS, SL=TP, tpd≥1, PF gates, year_stable, causal next-open, MIN_ATR≥1.5",
                    "pos_rr": "geoRR≥1.5, WR≥60% IS+OOS, tpd≥1, PF gates, year_stable",
                },
            },
            indent=2,
        )
    )


def dump(tested, quals, quals_pos, best, near):
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals_rr1": quals,
                "quals_posrr": quals_pos,
                "best_min_wr": best["min_wr"] if best else None,
                "best_name": best["name"] if best else None,
                "best_tpd": best.get("tpd") if best else None,
                "near": len(near),
                "gate": f"RR1 WR>={TARGET_WR:.0%} tpd>=1 | PosRR geoRR>=1.5 WR>=60% tpd>=1",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "tested": tested,
                "quals_rr1": quals,
                "quals_posrr": quals_pos,
                "best": best,
                "near_hits": near[:25],
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )


def main():
    OUT.mkdir(exist_ok=True)
    write_mastery()
    log("=== COURSE MASTERY HUNT (Silver Bullet / Turtle Soup / Wyckoff / IB / Fakey) ===")
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("5m", "15m", "30m"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])

    sp = specs()
    ex_rr1 = exits_rr1()
    ex_pos = exits_posrr()
    log(f"specs={len(sp)} exits_rr1={len(ex_rr1)} exits_pos={len(ex_pos)} tfs={list(frames)}")

    tested = quals = quals_pos = 0
    best = None
    near: list[dict] = []

    while True:
        RNG.shuffle(sp)
        for kind, base in sp:
            for invert in (False, True):
                pp = {**base}
                if invert:
                    pp["invert"] = True
                # drop None hours
                if pp.get("hours") is None:
                    pp.pop("hours", None)
                for tf, df in frames.items():
                    try:
                        L, S = gen_signals(df, kind, pp)
                    except Exception as e:
                        if tested < 8:
                            log(f"skip {kind}: {e}")
                        continue
                    nsig = int(L.sum() + S.sum())
                    if nsig < 50:
                        continue
                    # RR1 exits
                    for ex in ex_rr1:
                        tested += 1
                        tag = f"{'inv|' if invert else 'course|'}{kind}|{tf}"
                        res = evaluate(tag, tf, df, L, S, dict(ex), {"kind": kind, **pp})
                        if not res or res.trades < 40 or res.oos_trades < 20:
                            continue
                        min_wr = min(res.winrate, res.oos_winrate)
                        yok = year_stable(df, L, S, dict(ex))
                        row = {
                            "name": res.name,
                            "path": "rr1",
                            "wr": res.winrate,
                            "oos_wr": res.oos_winrate,
                            "min_wr": min_wr,
                            "tpd": res.trades_per_day,
                            "oos_tpd": res.oos_trades_per_day,
                            "n": res.trades,
                            "oos_n": res.oos_trades,
                            "pf": res.profit_factor,
                            "oos_pf": res.oos_profit_factor,
                            "params": res.params,
                            "year_ok": yok,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        sk = (
                            1 if (row["tpd"] >= 1.0 and min_wr >= 0.60) else 0,
                            min_wr,
                            row["tpd"],
                        )
                        if best is None or sk > best.get("_sk", (0, 0, 0)):
                            best = {**row, "_sk": sk}
                        if min_wr >= 0.65 and res.trades_per_day >= 0.7:
                            near.append(row)
                            near.sort(key=lambda r: (r["min_wr"], r["tpd"]), reverse=True)
                            near[:] = near[:50]
                            log(
                                f"NEAR course {min_wr:.1%} tpd={res.trades_per_day:.2f} "
                                f"year={yok} | {res.name}"
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
                                f"QUAL RR1 {res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f} | {res.name}"
                            )
                        if tested % 120 == 0:
                            dump(tested, quals, quals_pos, best, near)
                            bw = f"{best['min_wr']:.1%}" if best else "n/a"
                            log(
                                f"course tested={tested} quals={quals}/{quals_pos} "
                                f"best={bw} near={len(near)}"
                            )
                    # Pos-RR exits (sample to keep RAM/CPU sane)
                    for ex in RNG.sample(ex_pos, min(18, len(ex_pos))):
                        tested += 1
                        tag = f"{'inv|' if invert else 'course|'}{kind}|pos|{tf}"
                        res = evaluate(tag, tf, df, L, S, dict(ex), {"kind": kind, **pp})
                        if not res or res.trades < 40 or res.oos_trades < 20:
                            continue
                        min_wr = min(res.winrate, res.oos_winrate)
                        yok = year_stable(df, L, S, dict(ex))
                        row = {
                            "name": res.name,
                            "path": "posrr",
                            "wr": res.winrate,
                            "oos_wr": res.oos_winrate,
                            "min_wr": min_wr,
                            "tpd": res.trades_per_day,
                            "oos_tpd": res.oos_trades_per_day,
                            "n": res.trades,
                            "oos_n": res.oos_trades,
                            "pf": res.profit_factor,
                            "params": res.params,
                            "year_ok": yok,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
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
                        if qualifies_posrr(is_m, oos_m, dict(ex)) and yok:
                            quals_pos += 1
                            if save_champion(res, force=True):
                                write_bot(res)
                            log(
                                f"QUAL POSRR {res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"tpd={res.trades_per_day:.2f} | {res.name}"
                            )

        dump(tested, quals, quals_pos, best, near)
        log(f"course pass done tested={tested} quals={quals}/{quals_pos}; reshuffle")


if __name__ == "__main__":
    main()
