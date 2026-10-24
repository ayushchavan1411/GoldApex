#!/usr/bin/env python3
"""2h hunt: FVG + Fib + Bollinger + EMA only · 5m/15m · multi · no sessions.

Goal (unchanged):
  IS WR ≥ 80% AND OOS WR ≥ 80% AND |gap| ≤ 5pp
  tpd ≥ 1.0 on IS and full
  TP nets ≥ $2 after spread/commission/slip; SL ≤ 2× TP
  unlimited concurrent positions; no timezone/session blocks
  causal next-open fills
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rr1_hf80_10d_hunt import MIN_NET_TP, net_on_tp, simulate_multi
from rr1_search import add_indicators, gen_signals, load_frames, log, sess_mask

OUT = Path("rr1_search_results")
PROGRESS = OUT / "HF80_FFBE_PROGRESS.json"
LEADER = OUT / "HF80_FFBE_LEADERBOARD.jsonl"
BEST = OUT / "HF80_FFBE_BEST.json"
CHAMP = OUT / "HF80_FFBE_CHAMPION.json"
SUMMARY = OUT / "HF80_FFBE_SUMMARY.json"

TARGET_WR = 0.80
MAX_GAP = 0.05
# Override with FFBE_DEADLINE_S (seconds) when restarting mid-window
DEADLINE_S = float(__import__("os").environ.get("FFBE_DEADLINE_S", 2 * 60 * 60))
UPDATE_EVERY = 25


def as_bool(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(bool)
    return np.asarray(x, dtype=bool)


def local_combo(df: pd.DataFrame, kind: str, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Composite signals using ONLY FVG / Fib / BB / EMA (no RSI, no sessions)."""
    c = df["Close"]
    n = len(df)
    true = np.ones(n, dtype=bool)

    def fvg_touch():
        bt, tp = df["FVG_BULL_BOT"], df["FVG_BULL_TOP"]
        brt, brb = df["FVG_BEAR_TOP"], df["FVG_BEAR_BOT"]
        L = as_bool((df["Low"] <= tp) & (df["Low"] >= bt) & (c > tp))
        S = as_bool((df["High"] >= brb) & (df["High"] <= brt) & (c < brb))
        return L, S

    def fvg_ce():
        ce_b, ce_s = df["FVG_BULL_CE"], df["FVG_BEAR_CE"]
        L = as_bool((df["Low"] <= ce_b) & (c > ce_b) & (c >= df["FVG_BULL_BOT"]))
        S = as_bool((df["High"] >= ce_s) & (c < ce_s) & (c <= df["FVG_BEAR_TOP"]))
        return L, S

    def fib_touch():
        sw = p["swing"]
        fib = str(p["fib"])
        long_lvl = df[f"FIB_UP{fib}_{sw}"]
        short_lvl = df[f"FIB{fib}_{sw}"]
        wick = p.get("wick", 0.0)
        L = (df["Low"] <= long_lvl) & (c > long_lvl)
        S = (df["High"] >= short_lvl) & (c < short_lvl)
        if wick > 0:
            L = L & (df["lower_wick"] > wick)
            S = S & (df["upper_wick"] > wick)
        return as_bool(L), as_bool(S)

    def fib_discount():
        sw = p["swing"]
        fib = str(p["fib"])
        long_lvl = df[f"FIB_UP{fib}_{sw}"]
        short_lvl = df[f"FIB{fib}_{sw}"]
        L = as_bool((c < df[f"EQ_{sw}"]) & (df["Low"] <= long_lvl) & (c > long_lvl))
        S = as_bool((c > df[f"EQ_{sw}"]) & (df["High"] >= short_lvl) & (c < short_lvl))
        return L, S

    def bb_fade():
        lo = df[f"BB_LO_{p['period']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['period']}_{p['k']}"].shift(1)
        L = as_bool((df["Low"] <= lo) & (c > lo))
        S = as_bool((df["High"] >= up) & (c < up))
        return L, S

    def bb_mid():
        lo = df[f"BB_LO_{p['period']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['period']}_{p['k']}"].shift(1)
        mid = df[f"BB_MID_{p['period']}_{p['k']}"].shift(1)
        L = as_bool((df["Low"] <= lo) & (c > mid))
        S = as_bool((df["High"] >= up) & (c < mid))
        return L, S

    def ema_touch():
        ema = df[f"EMA_{p['ema']}"].shift(1)
        L = as_bool((df["Low"] <= ema) & (c > ema))
        S = as_bool((df["High"] >= ema) & (c < ema))
        return L, S

    def ema_stack():
        a = df[f"EMA_{p['a']}"].shift(1)
        b = df[f"EMA_{p['b']}"].shift(1)
        e = df[f"EMA_{p['c']}"].shift(1)
        L = as_bool((a > b) & (b > e) & (df["Low"] <= a) & (c > a))
        S = as_bool((a < b) & (b < e) & (df["High"] >= a) & (c < a))
        return L, S

    def ema_cross():
        fast = df[f"EMA_{p['fast']}"]
        slow = df[f"EMA_{p['slow']}"]
        L = as_bool((fast > slow) & (fast.shift(1) <= slow.shift(1)))
        S = as_bool((fast < slow) & (fast.shift(1) >= slow.shift(1)))
        return L, S

    def ema_trend():
        fast = df[f"EMA_{p.get('fast', 21)}"].shift(1)
        slow = df[f"EMA_{p.get('slow', 50)}"].shift(1)
        return as_bool(fast > slow), as_bool(fast < slow)

    def aand(La, Sa, Lb, Sb):
        return La & Lb, Sa & Sb

    if kind == "fvg_reclaim":
        return fvg_touch()
    if kind == "fvg_ce":
        return fvg_ce()
    if kind == "fvg1_reclaim":
        L, S = gen_signals(df, "fvg1_reclaim", {"sessions": None})
        return as_bool(L), as_bool(S)
    if kind == "fib_bounce":
        return fib_touch()
    if kind == "fib_discount":
        return fib_discount()
    if kind == "bb_fade":
        return bb_fade()
    if kind == "bb_mid_reject":
        return bb_mid()
    if kind == "ema_touch":
        return ema_touch()
    if kind == "ema_stack_pull":
        return ema_stack()
    if kind == "ema_cross":
        return ema_cross()
    if kind == "fvg_ema":
        return aand(*fvg_touch(), *ema_trend())
    if kind == "fvg_ce_ema":
        return aand(*fvg_ce(), *ema_trend())
    if kind == "fvg_bb":
        return aand(*fvg_touch(), *bb_fade())
    if kind == "fvg_fib":
        return aand(*fvg_touch(), *fib_touch())
    if kind == "fib_ema":
        return aand(*fib_touch(), *ema_trend())
    if kind == "fib_bb":
        return aand(*fib_touch(), *bb_fade())
    if kind == "bb_ema":
        return aand(*bb_fade(), *ema_trend())
    if kind == "bb_ema_touch":
        return aand(*bb_fade(), *ema_touch())
    if kind == "fvg_fib_ema":
        L, S = aand(*fvg_touch(), *fib_touch())
        return aand(L, S, *ema_trend())
    if kind == "fvg_bb_ema":
        L, S = aand(*fvg_touch(), *bb_fade())
        return aand(L, S, *ema_trend())
    if kind == "fib_bb_ema":
        L, S = aand(*fib_touch(), *bb_fade())
        return aand(L, S, *ema_trend())
    if kind == "fvg_fib_bb":
        L, S = aand(*fvg_touch(), *fib_touch())
        return aand(L, S, *bb_fade())
    if kind == "fvg_fib_bb_ema":
        L, S = aand(*fvg_touch(), *fib_touch())
        L, S = aand(L, S, *bb_fade())
        return aand(L, S, *ema_trend())
    if kind == "ema_stack_bb":
        return aand(*ema_stack(), *bb_fade())
    if kind == "ema_stack_fvg":
        return aand(*ema_stack(), *fvg_touch())
    if kind == "ema_stack_fib":
        return aand(*ema_stack(), *fib_touch())
    raise KeyError(kind)


def build_specs() -> list[tuple[str, dict]]:
    specs: list[tuple[str, dict]] = []
    # --- singles (no sessions) ---
    specs.append(("fvg_reclaim", {}))
    specs.append(("fvg_ce", {}))
    specs.append(("fvg1_reclaim", {}))
    for sw, fib, wick in itertools.product((10, 20, 40, 60), ("382", "500", "618", "786"), (0.0, 0.35, 0.45, 0.55)):
        specs.append(("fib_bounce", {"swing": sw, "fib": fib, "wick": wick}))
        specs.append(("fib_discount", {"swing": sw, "fib": fib}))
    for per, k in itertools.product((10, 20, 40, 50), (1.5, 2.0, 2.5, 3.0)):
        specs.append(("bb_fade", {"period": per, "k": k}))
        specs.append(("bb_mid_reject", {"period": per, "k": k}))
    for ema in (8, 13, 21, 34, 50, 100, 200):
        specs.append(("ema_touch", {"ema": ema}))
    for a, b, c in itertools.product((5, 8, 12), (21, 34), (50, 100, 200)):
        if a < b < c:
            specs.append(("ema_stack_pull", {"a": a, "b": b, "c": c}))
    for fast, slow in ((8, 21), (9, 21), (12, 26), (21, 50), (50, 200)):
        specs.append(("ema_cross", {"fast": fast, "slow": slow}))

    # --- two-way combos ---
    for fast, slow in ((8, 21), (21, 50), (50, 200)):
        specs.append(("fvg_ema", {"fast": fast, "slow": slow}))
        specs.append(("fvg_ce_ema", {"fast": fast, "slow": slow}))
    for per, k in ((10, 2.0), (20, 2.0), (20, 2.5), (40, 2.0)):
        specs.append(("fvg_bb", {"period": per, "k": k}))
        specs.append(("bb_ema", {"period": per, "k": k, "fast": 21, "slow": 50}))
        specs.append(("bb_ema", {"period": per, "k": k, "fast": 8, "slow": 21}))
        specs.append(("bb_ema_touch", {"period": per, "k": k, "ema": 21}))
        specs.append(("bb_ema_touch", {"period": per, "k": k, "ema": 50}))
        specs.append(("ema_stack_bb", {"a": 8, "b": 21, "c": 50, "period": per, "k": k}))
    for sw, fib in itertools.product((10, 20, 40, 60, 80), ("382", "500", "618", "786")):
        specs.append(("fvg_fib", {"swing": sw, "fib": fib, "wick": 0.0}))
        specs.append(("fib_bb", {"swing": sw, "fib": fib, "wick": 0.0, "period": 20, "k": 2.0}))
        specs.append(("ema_stack_fib", {"a": 8, "b": 21, "c": 50, "swing": sw, "fib": fib, "wick": 0.0}))
        specs.append(("ema_stack_fib", {"a": 5, "b": 21, "c": 50, "swing": sw, "fib": fib, "wick": 0.35}))
        specs.append(("ema_stack_fvg", {"a": 8, "b": 21, "c": 50}))
    # Dense fib_ema — current ceiling family (~75%)
    for sw, fib, wick, fast, slow in itertools.product(
        (10, 20, 40, 60, 80),
        ("382", "500", "618", "786"),
        (0.0, 0.25, 0.35, 0.45, 0.55),
        (8, 13, 21, 34),
        (21, 34, 50, 100, 200),
    ):
        if fast >= slow:
            continue
        specs.append(("fib_ema", {"swing": sw, "fib": fib, "wick": wick, "fast": fast, "slow": slow}))
    for sw, fib, wick, fast, slow in itertools.product(
        (20, 40, 60), ("618", "786"), (0.0, 0.35), (8, 21), (50, 200)
    ):
        specs.append(("fvg_fib_ema", {"swing": sw, "fib": fib, "wick": wick, "fast": fast, "slow": slow}))

    # --- three / four-way ---
    for sw, fib, per, k in itertools.product((20, 40, 60), ("618", "786"), (20, 40), (2.0, 2.5)):
        specs.append(("fvg_bb_ema", {"period": per, "k": k, "fast": 21, "slow": 50}))
        specs.append(("fib_bb_ema", {"swing": sw, "fib": fib, "wick": 0.0, "period": per, "k": k, "fast": 21, "slow": 50}))
        specs.append(("fib_bb_ema", {"swing": sw, "fib": fib, "wick": 0.35, "period": per, "k": k, "fast": 8, "slow": 21}))
        specs.append(("fvg_fib_bb", {"swing": sw, "fib": fib, "wick": 0.0, "period": per, "k": k}))
        specs.append(
            (
                "fvg_fib_bb_ema",
                {
                    "swing": sw,
                    "fib": fib,
                    "wick": 0.0,
                    "period": per,
                    "k": k,
                    "fast": 21,
                    "slow": 50,
                },
            )
        )
    # dedupe
    seen = set()
    uniq = []
    for kind, p in specs:
        key = (kind, tuple(sorted((k, str(v)) for k, v in p.items())))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((kind, p))
    return uniq


def build_exits() -> list[dict]:
    out = []
    for lot, tp, sm, mh, side in itertools.product(
        (0.02, 0.05, 0.1),
        (1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0),
        (1.5, 1.75, 2.0),
        (8, 12, 20, 30, 48, 72, 96, 120),
        (
            dict(allow_long=True, allow_short=True),
            dict(allow_long=True, allow_short=False),
            dict(allow_long=False, allow_short=True),
        ),
    ):
        if net_on_tp(tp, lot) < MIN_NET_TP:
            continue
        out.append(
            dict(
                tp_dist=tp,
                sl_dist=tp * sm,
                max_hold=mh,
                lot=lot,
                sl_mult=sm,
                net_tp=net_on_tp(tp, lot),
                **side,
            )
        )
    return out


def write_progress(payload: dict) -> None:
    OUT.mkdir(exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    LEADER.write_text("")
    t_start = time.time()
    deadline = t_start + DEADLINE_S
    log("FFBE hunt load (FVG/Fib/BB/EMA · 5m/15m · multi · no sessions)…")
    # Prefer longest local 5m CSV (Dukascopy 2y over short Yahoo seed)
    data_dir = Path("data")
    if data_dir.exists():
        fives = sorted(data_dir.glob("XAUUSD_5min_*.csv"), key=lambda p: p.stat().st_size)
        if fives:
            best5 = fives[-1]
            for other in fives[:-1]:
                # keep yahoo as backup only if dukascopy missing; else hide shorter seeds
                if "yahoo" in other.name and "dukascopy" in best5.name:
                    other.rename(other.with_suffix(other.suffix + ".bak"))
                    log(f"parked shorter seed {other.name}")
    raw = load_frames(include_resampled=True)
    # Force 15m from 5m even when span < 200d (load_frames gate)
    if "5m" in raw and "15m" not in raw:
        from rr1_search import resample_ohlc

        raw["15m"] = resample_ohlc(raw["5m"], "15min")
        log(f"Forced resample 15m: {len(raw['15m'])} from 5m")
    frames = {tf: add_indicators(raw[tf]) for tf in ("5m", "15m") if tf in raw}
    if not frames:
        raise SystemExit("need 5m and/or 15m data")

    windows = {}
    for tf, df in frames.items():
        n = len(df)
        warm = 400 if tf == "5m" else 250
        split = int(n * 0.7)
        windows[tf] = (warm, split, n - 2)
        days = max((df.index[-1] - df.index[0]).days, 1)
        log(f"{tf} bars={n} days≈{days} split={df.index[split]}")

    specs = build_specs()
    exits = build_exits()
    log(f"specs={len(specs)} exits={len(exits)} deadline={DEADLINE_S/3600:.1f}h")

    # probe exit biased to WR
    probe_tp = 2.0
    while net_on_tp(probe_tp, 0.05) < MIN_NET_TP:
        probe_tp += 0.3

    tested = 0
    signal_combos = 0
    hits80 = 0
    best = None  # by min(is_wr, oos_wr) among tpd≥1
    best_any = None  # highest min WR regardless of tpd (for reporting)
    champ = None
    board: list[dict] = []

    def consider_best(row: dict, *, require_tpd: bool) -> None:
        nonlocal best, best_any
        score = min(row["is_wr"], row["oos_wr"])
        if best_any is None or score > best_any["score"]:
            best_any = {**row, "score": score}
        if require_tpd and row["is_tpd"] >= 1.0 and row["full_tpd"] >= 1.0:
            if best is None or score > best["score"]:
                best = {**row, "score": score}

    tf_order = [t for t in ("15m", "5m") if t in frames]
    stop = False
    for tf in tf_order:
        if stop:
            break
        df = frames[tf]
        warm, split, end = windows[tf]
        log(f"=== TF {tf} ===")
        for kind, base in specs:
            if time.time() >= deadline:
                stop = True
                break
            for invert in (False, True):
                if time.time() >= deadline:
                    stop = True
                    break
                try:
                    L, S = local_combo(df, kind, base)
                except Exception:
                    continue
                if invert:
                    L, S = S, L
                nsig = int(L[warm:end].sum() + S[warm:end].sum())
                if nsig < 400 or nsig > 25000:
                    continue
                signal_combos += 1

                is_pr = simulate_multi(
                    df,
                    L,
                    S,
                    tp_dist=probe_tp,
                    sl_dist=2 * probe_tp,
                    max_hold=20,
                    lot=0.05,
                    allow_long=True,
                    allow_short=True,
                    start_i=warm,
                    end_i=split,
                )
                # track probe even if weak
                probe_row = {
                    "kind": kind,
                    "tf": tf,
                    "invert": invert,
                    "params": base,
                    "phase": "probe",
                    "is_wr": is_pr["winrate"],
                    "oos_wr": is_pr["winrate"],
                    "is_tpd": is_pr["tpd"],
                    "full_tpd": is_pr["tpd"],
                    "is_n": is_pr["trades"],
                    "oos_n": 0,
                    "is_net": is_pr["net"],
                    "oos_net": 0.0,
                    "full_net": is_pr["net"],
                    "exit": f"probe_tp{probe_tp}",
                }
                consider_best(probe_row, require_tpd=True)

                if is_pr["tpd"] < 0.85 or is_pr["trades"] < 150:
                    continue
                # expand if probe WR looks promising
                if is_pr["winrate"] < 0.62:
                    continue

                ex_iter = exits
                if is_pr["winrate"] < 0.72:
                    ex_iter = [e for e in exits if e["sl_mult"] >= 1.75 and e["lot"] in (0.05, 0.1)]

                for ex in ex_iter:
                    if time.time() >= deadline:
                        stop = True
                        break
                    tested += 1
                    is_m = simulate_multi(
                        df,
                        L,
                        S,
                        tp_dist=ex["tp_dist"],
                        sl_dist=ex["sl_dist"],
                        max_hold=ex["max_hold"],
                        lot=ex["lot"],
                        allow_long=ex["allow_long"],
                        allow_short=ex["allow_short"],
                        start_i=warm,
                        end_i=split,
                    )
                    if is_m["tpd"] < 1.0 or is_m["trades"] < 200:
                        continue
                    if is_m["winrate"] < 0.65:
                        continue
                    oos = simulate_multi(
                        df,
                        L,
                        S,
                        tp_dist=ex["tp_dist"],
                        sl_dist=ex["sl_dist"],
                        max_hold=ex["max_hold"],
                        lot=ex["lot"],
                        allow_long=ex["allow_long"],
                        allow_short=ex["allow_short"],
                        start_i=split,
                        end_i=end,
                    )
                    if oos["tpd"] < 0.85 or oos["trades"] < 60:
                        continue
                    gap = abs(is_m["winrate"] - oos["winrate"])
                    sim_full = simulate_multi(
                        df,
                        L,
                        S,
                        tp_dist=ex["tp_dist"],
                        sl_dist=ex["sl_dist"],
                        max_hold=ex["max_hold"],
                        lot=ex["lot"],
                        allow_long=ex["allow_long"],
                        allow_short=ex["allow_short"],
                        start_i=warm,
                        end_i=end,
                    )
                    if sim_full["tpd"] < 1.0:
                        continue
                    side = (
                        "LS"
                        if ex["allow_long"] and ex["allow_short"]
                        else ("L" if ex["allow_long"] else "S")
                    )
                    name = (
                        f"{'inv|' if invert else ''}{kind}|tp{ex['tp_dist']:g}/sl{ex['sl_dist']:g}|"
                        f"h{ex['max_hold']}|{tf}|{side}|lot{ex['lot']:g}|multi"
                    )
                    row = {
                        "name": name,
                        "kind": kind,
                        "tf": tf,
                        "invert": invert,
                        "params": base,
                        "exit": {
                            "tp": ex["tp_dist"],
                            "sl": ex["sl_dist"],
                            "hold": ex["max_hold"],
                            "lot": ex["lot"],
                            "side": side,
                            "net_tp": ex["net_tp"],
                        },
                        "is_wr": float(is_m["winrate"]),
                        "oos_wr": float(oos["winrate"]),
                        "wr_gap": float(gap),
                        "is_tpd": float(is_m["tpd"]),
                        "oos_tpd": float(oos["tpd"]),
                        "full_tpd": float(sim_full["tpd"]),
                        "is_n": int(is_m["trades"]),
                        "oos_n": int(oos["trades"]),
                        "full_n": int(sim_full["trades"]),
                        "is_net": float(is_m["net"]),
                        "oos_net": float(oos["net"]),
                        "full_net": float(sim_full["net"]),
                        "full_dd": float(sim_full.get("dd", 0.0)),
                        "score": float(min(is_m["winrate"], oos["winrate"])),
                    }
                    consider_best(row, require_tpd=True)
                    if (
                        is_m["winrate"] >= TARGET_WR
                        and oos["winrate"] >= TARGET_WR
                        and gap <= MAX_GAP
                        and is_m["net"] > 0
                        and oos["net"] > 0
                        and sim_full["tpd"] >= 1.0
                    ):
                        hits80 += 1
                        champ = row
                        CHAMP.write_text(json.dumps({"status": "FOUND", "champion": row}, indent=2, default=str))
                        log(
                            f"FOUND {name} IS={is_m['winrate']:.1%} OOS={oos['winrate']:.1%} "
                            f"tpd={sim_full['tpd']:.2f}"
                        )
                        stop = True
                        break

                    if row["score"] >= 0.70 or (row["score"] >= 0.65 and is_m["net"] > 0):
                        board.append(row)
                        with LEADER.open("a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        log(
                            f"HIT {row['score']:.1%} IS={is_m['winrate']:.1%} OOS={oos['winrate']:.1%} "
                            f"tpd={sim_full['tpd']:.2f} net={sim_full['net']:.0f} | {name}"
                        )

                    if tested % UPDATE_EVERY == 0:
                        elapsed = time.time() - t_start
                        payload = {
                            "elapsed_s": elapsed,
                            "remaining_s": max(0, deadline - time.time()),
                            "signal_combos": signal_combos,
                            "tested_exits": tested,
                            "hits80": hits80,
                            "best_tpd1": best,
                            "best_any": best_any,
                            "status": "RUNNING",
                            "goal": "IS&OOS≥80% + tpd≥1 + TP net≥$2 + SL≤2×TP + FVG/Fib/BB/EMA only",
                        }
                        write_progress(payload)
                        BEST.write_text(json.dumps({"best_tpd1": best, "best_any": best_any}, indent=2, default=str))
                if stop:
                    break
            if stop:
                break

    elapsed = time.time() - t_start
    summary = {
        "elapsed_s": elapsed,
        "signal_combos": signal_combos,
        "tested_exits": tested,
        "hits80": hits80,
        "best_tpd1": best,
        "best_any": best_any,
        "champion": champ,
        "status": "FOUND" if champ else "NOT_FOUND",
        "verdict": (
            f"FOUND {champ['name']}"
            if champ
            else (
                f"No ≥80% at tpd≥1 after {signal_combos} signal combos / {tested} exits. "
                f"Best tpd≥1 minWR="
                + (
                    f"{best['score']:.1%} ({best.get('kind')}/{best.get('tf')})"
                    if best
                    else "none"
                )
            )
        ),
        "constraints": {
            "indicators": ["FVG", "Fib", "Bollinger", "EMA"],
            "tfs": list(frames.keys()),
            "sessions": None,
            "mode": "multi unlimited",
        },
    }
    write_progress({**summary, "status": summary["status"], "remaining_s": 0})
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    BEST.write_text(json.dumps({"best_tpd1": best, "best_any": best_any}, indent=2, default=str))
    log(f"DONE {summary['verdict']}")


if __name__ == "__main__":
    main()
