#!/usr/bin/env python3
"""Phase A: hunt the LOWEST win-rate RR1 bases (anti-edge).
Phase B: invert those losers (buy when they sell) and promote high-WR fades.

Scoring for bases: minimize max(WR_IS, WR_OOS) (both periods weak), then
min(WR_IS, WR_OOS). This rejects IS~50%/OOS-collapse noise that does not fade.
At 1:1, a real dual-weak ~40% base → ~60%+ inverted WR if the anti-edge is genuine.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

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

RNG = random.Random(7)
LOW_BOARD = OUT / "LOWWR_BASES.jsonl"
INV_BOARD = OUT / "LOWWR_INVERT_LEADERBOARD.jsonl"
PROGRESS = OUT / "lowwr_invert_progress.json"
SUMMARY = OUT / "LOWWR_INVERT_SUMMARY.json"

MIN_BASE_N = 60
MIN_BASE_OOS_N = 20
# Prefer dual-weak sample: both IS+OOS low with enough OOS trades
MIN_GENUINE_N = 80
MIN_GENUINE_OOS_N = 30
# keep pushing toward the floor — no artificial "must be ≤45%" until invert stage
KEEP_WORST = 80


def base_rank_key(row: dict):
    """Lower is worse (better anti-edge). Prefer both periods weak + more trades."""
    return (
        row.get("max_wr", max(row["wr"], row["oos_wr"])),
        row["min_wr"],
        -(row["n"] + row["oos_n"]),
    )


def exits_base():
    """Exits used while hunting losers (favor more trades / cleaner WR estimate)."""
    return [
        {"rr": rr, "max_hold": mh, "atr": atr, "allow_long": True, "allow_short": True}
        for rr in (1.0, 1.2)
        for mh in (6, 8, 12, 18)
        for atr in ("ATR_14", "ATR_7")
    ]


def exits_inv():
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


def specs():
    """Bias toward trend/breakout kinds that often lose at RR1 (good fade candidates).

    sessions always None — do not block any trading session.
    """
    out = []
    sessions = [None]
    for sess in sessions:
        for fast, slow in itertools.product([3, 5, 8, 9, 12, 13, 21, 34], [21, 34, 50, 55, 100, 200]):
            if fast >= slow:
                continue
            out.append(("ema_cross", {"fast": fast, "slow": slow, "sessions": sess}))
        for n in (8, 10, 12, 15, 20, 25, 30, 40, 55, 80):
            out.append(("turtle_break", {"n": n, "sessions": sess}))
        for m in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
            out.append(("supertrend_flip", {"mult": m, "sessions": sess}))
        for per in (10, 15, 20, 30, 40, 55):
            out.append(("donch_fade", {"period": per, "sessions": sess}))
        out.append(("macd_cross", {"sessions": sess}))
        for orb in ("LONDON", "NY", "ASIAN"):
            try:
                out.append(("orb_break", {"orb": orb, "sessions": sess}))
            except Exception:
                pass
        # momentum RSI (cross with tight bands often wrong at 1:1)
        for rsi, os_, ob in itertools.product([3, 5, 7, 14], [40, 45, 50], [50, 55, 60]):
            if os_ >= ob:
                continue
            out.append(("rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}))
            if rsi in (7, 14):
                out.append(("rsi_macd", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}))
        for swing in (10, 20, 40):
            out.append(("mss_entry", {"swing": swing, "sessions": sess}))
            out.append(("bos_pullback", {"swing": swing, "ema": 21, "sessions": sess}))
        # Order-flow proxies (all-session)
        for lo, hi in ((0.2, 0.8), (0.25, 0.75), (0.3, 0.7)):
            out.append(("of_clv_fade", {"lo": lo, "hi": hi, "sessions": sess}))
        for n in (2, 3, 4, 5):
            out.append(("of_stacked_imb", {"n": n, "sessions": sess}))
            out.append(("of_stacked_fade", {"n": n, "sessions": sess}))
            out.append(("of_delta_flip", {"n": n, "sessions": sess}))
        out.append(("of_absorption", {"sessions": sess}))
        out.append(("of_sweep_absorb", {"sessions": sess}))
        for span in (8, 20):
            out.append(("of_cvd_cross", {"span": span, "sessions": sess}))
            out.append(("of_cvd_div", {"look": span, "sessions": sess}))
    seen, uniq = set(), []
    for kind, p in out:
        key = (kind, json.dumps(p, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((kind, p))
    return uniq


def append_jsonl(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def dump_summary(worst_bases, best_invs, tested, inverted, quals):
    SUMMARY.write_text(
        json.dumps(
            {
                "tested_base": tested,
                "inverted": inverted,
                "quals": quals,
                "worst_bases": worst_bases[:25],
                "best_inverts": best_invs[:25],
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )
    PROGRESS.write_text(
        json.dumps(
            {
                "tested_base": tested,
                "inverted": inverted,
                "quals": quals,
                "worst_min_wr": worst_bases[0]["min_wr"] if worst_bases else None,
                "worst_max_wr": worst_bases[0].get("max_wr") if worst_bases else None,
                "best_inv_min_wr": best_invs[0]["min_wr"] if best_invs else None,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )


def invert_base(df, tf, kind, base_p, base_row, exs_inv, best_invs, quals, champion_holder):
    """Fade one low-WR base across invert exits; return updated quals + counts."""
    inv_p = {**base_p, "invert": True}
    try:
        Li, Si = gen_signals(df, kind, inv_p)
    except Exception:
        return 0, quals
    n_done = 0
    for ex in exs_inv:
        n_done += 1
        exi = dict(ex)
        res = evaluate(
            f"inv|{kind}|{tf}",
            tf,
            df,
            Li,
            Si,
            exi,
            {"kind": kind, **inv_p},
        )
        if not res or res.trades < 25 or res.oos_trades < 12:
            continue
        row = {
            "name": res.name,
            "wr": res.winrate,
            "oos_wr": res.oos_winrate,
            "min_wr": min(res.winrate, res.oos_winrate),
            "tpd": res.trades_per_day,
            "oos_tpd": res.oos_trades_per_day,
            "n": res.trades,
            "oos_n": res.oos_trades,
            "pf": res.profit_factor,
            "oos_pf": res.oos_profit_factor,
            "pnl": res.net_pnl,
            "oos_pnl": res.oos_net_pnl,
            "base_min_wr": base_row["min_wr"],
            "base_wr": base_row["wr"],
            "base_name": base_row["name"],
            "params": res.params,
            "year_ok": year_stable(df, Li, Si, exi),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_jsonl(INV_BOARD, row)
        best_invs.append(row)
        # promote
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
        q = qualifies(is_m, oos_m)
        if q:
            q = year_stable(df, Li, Si, exi)
        near = res.winrate >= 0.65 and res.oos_winrate >= 0.65 and res.trades >= 30
        if q or near:
            cur = champion_holder[0]
            score = (min(res.winrate, res.oos_winrate), min(res.trades_per_day, res.oos_trades_per_day))
            bscore = (
                (min(cur.winrate, cur.oos_winrate), min(cur.trades_per_day, cur.oos_trades_per_day))
                if cur
                else (-1, -1)
            )
            if cur is None or score > bscore:
                champion_holder[0] = res
                save_champion(res, force=bool(q))
                write_bot(res)
                log(
                    f"{'QUAL ' if q else 'NEAR '}LOWWR→INV WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                    f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                    f"from base minWR={base_row['min_wr']:.1%} | {res.name}"
                )
            if q:
                quals += 1
    return n_done, quals


def main():
    OUT.mkdir(exist_ok=True)
    log("=== LOW-WR HUNT → INVERT (phase A losers, phase B fades) ===")
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in ("5m", "15m", "1h", "30m"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])

    sp = specs()
    exs_b = exits_base()
    exs_i = exits_inv()
    log(f"specs={len(sp)} base_exits={len(exs_b)} inv_exits={len(exs_i)} tfs={list(frames)}")

    tested = 0
    inverted = 0
    quals = 0
    worst_bases: list[dict] = []
    best_invs: list[dict] = []
    champion_holder = [None]
    faded_keys: set[str] = set()

    while True:
        RNG.shuffle(sp)
        for kind, base_p in sp:
            for tf, df in frames.items():
                try:
                    L, S = gen_signals(df, kind, base_p)
                except Exception:
                    continue
                if int(L.sum() + S.sum()) < 120:
                    continue
                # pick the WORST (lowest WR) exit for this signal — that's the anti-edge we want
                base_best = None
                for ex in exs_b:
                    tested += 1
                    res = evaluate(f"low|{kind}|{tf}", tf, df, L, S, ex, {"kind": kind, **base_p})
                    if not res:
                        continue
                    if res.trades < MIN_BASE_N or res.oos_trades < MIN_BASE_OOS_N:
                        continue
                    # prefer lower max(IS,OOS) (both weak); then lower min; then more trades
                    key = (
                        max(res.winrate, res.oos_winrate),
                        min(res.winrate, res.oos_winrate),
                        -(res.trades + res.oos_trades),
                    )
                    bkey = (
                        (
                            max(base_best.winrate, base_best.oos_winrate),
                            min(base_best.winrate, base_best.oos_winrate),
                            -(base_best.trades + base_best.oos_trades),
                        )
                        if base_best
                        else (9.0, 9.0, 0)
                    )
                    if key < bkey:
                        base_best = res
                        base_ex = ex
                if base_best is None:
                    continue

                brow = {
                    "name": base_best.name,
                    "wr": base_best.winrate,
                    "oos_wr": base_best.oos_winrate,
                    "min_wr": min(base_best.winrate, base_best.oos_winrate),
                    "max_wr": max(base_best.winrate, base_best.oos_winrate),
                    "avg_wr": (base_best.winrate + base_best.oos_winrate) / 2,
                    "tpd": base_best.trades_per_day,
                    "oos_tpd": base_best.oos_trades_per_day,
                    "n": base_best.trades,
                    "oos_n": base_best.oos_trades,
                    "pf": base_best.profit_factor,
                    "oos_pf": base_best.oos_profit_factor,
                    "params": base_best.params,
                    "tf": tf,
                    "kind": kind,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                append_jsonl(LOW_BOARD, brow)
                # Only promote into "worst" board when both periods are weak with sample
                genuine = (
                    brow["n"] >= MIN_GENUINE_N
                    and brow["oos_n"] >= MIN_GENUINE_OOS_N
                    and brow["wr"] <= 0.48
                    and brow["oos_wr"] <= 0.48
                )
                if genuine:
                    worst_bases.append(brow)
                    worst_bases.sort(key=base_rank_key)
                    worst_bases[:] = worst_bases[:KEEP_WORST]

                # Phase B: only fade GENUINE losers (both IS and OOS weak) — avoids
                # IS~50%/OOS~25% noise that does not invert into a real high-WR edge.
                fade_key = json.dumps({"kind": kind, "tf": tf, "p": base_p}, sort_keys=True, default=str)
                both_weak = brow["wr"] <= 0.48 and brow["oos_wr"] <= 0.48
                should_fade = both_weak and genuine and (
                    brow["max_wr"] <= 0.46
                    or (
                        worst_bases
                        and brow["max_wr"]
                        <= worst_bases[min(14, len(worst_bases) - 1)].get("max_wr", 1.0) + 0.02
                    )
                )
                if should_fade and fade_key not in faded_keys:
                    faded_keys.add(fade_key)
                    n_inv, quals = invert_base(
                        df, tf, kind, base_p, brow, exs_i, best_invs, quals, champion_holder
                    )
                    inverted += n_inv
                    best_invs.sort(key=lambda r: (r["min_wr"], r["n"] + r["oos_n"]), reverse=True)
                    best_invs[:] = best_invs[:KEEP_WORST]
                    if brow["max_wr"] <= 0.44 or (best_invs and best_invs[0]["min_wr"] >= 0.65):
                        log(
                            f"LOW base maxWR={brow['max_wr']:.1%} {brow['wr']:.1%}/{brow['oos_wr']:.1%} "
                            f"n={brow['n']}/{brow['oos_n']} | {brow['name']} "
                            f"→ best_inv minWR={(best_invs[0]['min_wr'] if best_invs else 0):.1%}"
                        )

                if tested % 150 == 0:
                    dump_summary(worst_bases, best_invs, tested, inverted, quals)
                    wb = (
                        f"max={worst_bases[0]['max_wr']:.1%}/min={worst_bases[0]['min_wr']:.1%}"
                        if worst_bases
                        else "n/a"
                    )
                    bi = f"{best_invs[0]['min_wr']:.1%}" if best_invs else "n/a"
                    log(
                        f"lowwr tested={tested} inverted={inverted} quals={quals} "
                        f"worst_base={wb} best_inv={bi}"
                    )

        dump_summary(worst_bases, best_invs, tested, inverted, quals)
        log(f"lowwr pass done tested={tested} inverted={inverted} quals={quals}; reshuffle")
        # re-invert absolute worst 20 each pass (in case exits grid missed)
        for brow in worst_bases[:20]:
            kind = brow["kind"]
            tf = brow["tf"]
            df = frames[tf]
            base_p = {k: v for k, v in brow["params"].items() if k not in ("kind", "exit")}
            if isinstance(base_p.get("sessions"), list):
                base_p["sessions"] = tuple(base_p["sessions"])
            fade_key = "re|" + json.dumps({"kind": kind, "tf": tf, "p": base_p}, sort_keys=True, default=str)
            if fade_key in faded_keys:
                continue
            faded_keys.add(fade_key)
            n_inv, quals = invert_base(df, tf, kind, base_p, brow, exs_i, best_invs, quals, champion_holder)
            inverted += n_inv
        best_invs.sort(key=lambda r: (r["min_wr"], r["n"] + r["oos_n"]), reverse=True)
        best_invs[:] = best_invs[:KEEP_WORST]
        dump_summary(worst_bases, best_invs, tested, inverted, quals)


if __name__ == "__main__":
    main()
