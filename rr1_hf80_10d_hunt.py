#!/usr/bin/env python3
"""10-day high-frequency scout: ≥80% WR, unlimited concurrent positions.

Rules (user 2026-10-19):
  - Free TP as long as a clean TP hit nets ≥ $2 after spread + commission (+ slip).
  - SL distance ≤ 2× TP distance.
  - No limit on open positions (stack every signal).
  - First pass: last 10 trading days only (expect high tpd).
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rr1_search import (
    COMMISSION_PER_LOT_RT,
    CONTRACT_SIZE,
    OUT,
    SPREAD_PRICE,
    START_CAP,
    add_indicators,
    gen_signals,
    load_frames,
    log,
    trading_days,
)

RNG = random.Random(801)
LEADER = OUT / "HF80_10D_LEADERBOARD.jsonl"
PROGRESS = OUT / "hf80_10d_progress.json"
SUMMARY = OUT / "HF80_10D_SUMMARY.json"
CHAMP = OUT / "HF80_10D_CHAMPION.json"

TARGET_WR = 0.80
MIN_NET_TP = 2.0  # $ after charges on a clean TP
SLIP_PRICE = 0.05  # extra adverse $ per side beyond quoted spread
DAYS = 10


def net_on_tp(tp_dist: float, lot: float, spread: float = SPREAD_PRICE, slip: float = SLIP_PRICE) -> float:
    """Expected $ net if price hits TP (same fill model as simulate_multi).

    entry = mid + spread/2 + slip; tp = entry + tp_dist; exit = tp - spread/2 - slip
    → gross = (tp_dist - spread/2 - slip) * oz; net = gross - commission.
    """
    oz = lot * CONTRACT_SIZE
    gross = (tp_dist - spread / 2.0 - slip) * oz
    return float(gross - COMMISSION_PER_LOT_RT * lot)


def min_tp_dist(lot: float) -> float:
    """Smallest TP $ distance that still clears MIN_NET_TP after costs."""
    # binary search
    lo, hi = 0.05, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if net_on_tp(mid, lot) >= MIN_NET_TP:
            hi = mid
        else:
            lo = mid
    return hi


def simulate_multi(
    df: pd.DataFrame,
    long_e,
    short_e,
    *,
    tp_dist: float,
    sl_dist: float,
    max_hold: int,
    lot: float,
    allow_long: bool = True,
    allow_short: bool = True,
    spread: float = SPREAD_PRICE,
    slip: float = SLIP_PRICE,
    commission_rt: float = COMMISSION_PER_LOT_RT,
    start_cap: float = START_CAP,
    start_i: int = 0,
    end_i: int | None = None,
) -> dict:
    """Unlimited concurrent positions; signal on bar i → fill open[i+1]."""
    n = len(df)
    if end_i is None:
        end_i = n - 2
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    times = df.index
    oz = lot * CONTRACT_SIZE
    comm = commission_rt * lot
    capital = float(start_cap)
    peak = capital
    max_dd = 0.0
    pnls: list[float] = []
    open_tr: list[dict] = []
    max_concurrent = 0
    signals_taken = 0
    wins_tp = losses_sl = time_exits = 0

    for i in range(start_i, end_i):
        still = []
        for t in open_tr:
            t["bars"] += 1
            hi, lo = highs[i], lows[i]
            hit = False
            exit_px = None
            reason = None
            if t["dir"] == "LONG":
                if lo <= t["sl"]:
                    exit_px, hit, reason = t["sl"], True, "sl"
                elif hi >= t["tp"]:
                    exit_px, hit, reason = t["tp"], True, "tp"
            else:
                if hi >= t["sl"]:
                    exit_px, hit, reason = t["sl"], True, "sl"
                elif lo <= t["tp"]:
                    exit_px, hit, reason = t["tp"], True, "tp"
            if not hit and t["bars"] >= max_hold:
                exit_px, hit, reason = closes[i], True, "time"
            if hit:
                if t["dir"] == "LONG":
                    exit_px = exit_px - spread / 2.0 - slip
                    gross = (exit_px - t["entry"]) * oz
                else:
                    exit_px = exit_px + spread / 2.0 + slip
                    gross = (t["entry"] - exit_px) * oz
                pnl = gross - comm
                capital += pnl
                peak = max(peak, capital)
                max_dd = max(max_dd, (peak - capital) / peak if peak else 0.0)
                pnls.append(pnl)
                if reason == "tp":
                    wins_tp += 1
                elif reason == "sl":
                    losses_sl += 1
                else:
                    time_exits += 1
            else:
                still.append(t)
        open_tr = still
        max_concurrent = max(max_concurrent, len(open_tr))

        if i + 1 >= n:
            continue
        gl = allow_long and bool(long_e[i])
        gs = allow_short and bool(short_e[i])
        if gl == gs:
            continue
        raw = opens[i + 1]
        if not np.isfinite(raw):
            continue
        direction = "LONG" if gl else "SHORT"
        if direction == "LONG":
            entry = raw + spread / 2.0 + slip
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            entry = raw - spread / 2.0 - slip
            sl = entry + sl_dist
            tp = entry - tp_dist
        open_tr.append({"dir": direction, "entry": entry, "sl": sl, "tp": tp, "bars": 0})
        signals_taken += 1
        max_concurrent = max(max_concurrent, len(open_tr))

    for t in open_tr:
        j = min(end_i, n - 1)
        exit_px = closes[j]
        if t["dir"] == "LONG":
            exit_px = exit_px - spread / 2.0 - slip
            gross = (exit_px - t["entry"]) * oz
        else:
            exit_px = exit_px + spread / 2.0 + slip
            gross = (t["entry"] - exit_px) * oz
        pnl = gross - comm
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0.0)
        pnls.append(pnl)
        time_exits += 1

    if not pnls:
        return {
            "trades": 0,
            "winrate": 0.0,
            "tpd": 0.0,
            "eq": capital,
            "net": 0.0,
            "dd": max_dd,
            "pf": 0.0,
            "max_concurrent": 0,
            "signals": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "wins_tp": 0,
            "losses_sl": 0,
            "time_exits": 0,
            "expectancy": 0.0,
        }
    arr = np.asarray(pnls, float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    tdays = max(trading_days(times[start_i:end_i]), 1e-9)
    return {
        "trades": int(len(arr)),
        "winrate": float((arr > 0).mean()),
        "tpd": float(len(arr) / tdays),
        "eq": float(capital),
        "net": float(arr.sum()),
        "dd": float(max_dd),
        "pf": float((gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)),
        "max_concurrent": int(max_concurrent),
        "signals": int(signals_taken),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "wins_tp": int(wins_tp),
        "losses_sl": int(losses_sl),
        "time_exits": int(time_exits),
        "expectancy": float(arr.mean()),
    }


def local_signals(df: pd.DataFrame, kind: str, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extra high-fire scalp signals not in rr1_search.gen_signals."""
    c = df["Close"]
    if kind == "rsi_stay":
        rsi = df[f"RSI_{p['rsi']}"]
        # fire every bar while extreme (dense)
        long_e = (rsi < p["os"]).to_numpy(bool)
        short_e = (rsi > p["ob"]).to_numpy(bool)
        return long_e, short_e
    if kind == "rsi_recover":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (rsi > p["os"]) & (rsi.shift(1) <= p["os"])
        short_e = (rsi < p["ob"]) & (rsi.shift(1) >= p["ob"])
        return long_e.to_numpy(bool), short_e.to_numpy(bool)
    if kind == "ema_dist_fade":
        ema = df[f"EMA_{p['ema']}"]
        dist = (c - ema) / df["ATR_14"].replace(0, np.nan)
        long_e = (dist < -p["th"]) & (c > c.shift(1))
        short_e = (dist > p["th"]) & (c < c.shift(1))
        return long_e.fillna(False).to_numpy(bool), short_e.fillna(False).to_numpy(bool)
    if kind == "wick_reject":
        long_e = (df["lower_wick"] > p["wick"]) & (df["body_pct"] < p["body"]) & (c > df["Open"])
        short_e = (df["upper_wick"] > p["wick"]) & (df["body_pct"] < p["body"]) & (c < df["Open"])
        return long_e.fillna(False).to_numpy(bool), short_e.fillna(False).to_numpy(bool)
    if kind == "micro_bounce":
        # prior bar made lower low, this bar closes back above prior low
        long_e = (df["Low"] < df["Low"].shift(1)) & (c > df["Low"].shift(1)) & (c > df["Open"])
        short_e = (df["High"] > df["High"].shift(1)) & (c < df["High"].shift(1)) & (c < df["Open"])
        return long_e.fillna(False).to_numpy(bool), short_e.fillna(False).to_numpy(bool)
    raise KeyError(kind)


LOCAL_KINDS = {"rsi_stay", "rsi_recover", "ema_dist_fade", "wick_reject", "micro_bounce"}


def specs() -> list[tuple[str, dict]]:
    """High-fire mean-reversion / micro setups for 1m–5m."""
    out: list[tuple[str, dict]] = []
    for rsi, os, ob in itertools.product((2, 3, 5, 7), (10, 15, 20, 25, 30), (70, 75, 80, 85, 90)):
        out.append(("rsi_stay", {"rsi": rsi, "os": os, "ob": ob}))
        out.append(("rsi_recover", {"rsi": rsi, "os": os, "ob": ob}))
    for ema, th in itertools.product((8, 21), (0.5, 1.0, 1.5, 2.0)):
        out.append(("ema_dist_fade", {"ema": ema, "th": th}))
    for wick, body in itertools.product((0.5, 0.6, 0.7), (0.25, 0.35)):
        out.append(("wick_reject", {"wick": wick, "body": body}))
    out.append(("micro_bounce", {}))
    for rsi, os, ob in itertools.product((2, 3, 5, 7, 14), (10, 15, 20, 25, 30), (70, 75, 80, 85, 90)):
        out.append(("rsi_level", {"rsi": rsi, "os": os, "ob": ob}))
        out.append(("rsi_cross", {"rsi": rsi, "os": os, "ob": ob}))
    for per, k in itertools.product((10, 20), (1.5, 2.0, 2.5)):
        out.append(("bb_fade", {"period": per, "k": k}))
        out.append(("bb_rsi_fade", {"period": per, "k": k, "rsi": 5, "os": 25, "ob": 75}))
        out.append(("bb_rsi_fade", {"period": per, "k": k, "rsi": 7, "os": 20, "ob": 80}))
        out.append(("bb_mid_reject", {"period": per, "k": k}))
    for per, os, ob in itertools.product((5, 9, 14), (15, 20, 25), (75, 80, 85)):
        out.append(("stoch_cross", {"period": per, "os": os, "ob": ob}))
    for z, th in itertools.product((20, 40), (1.0, 1.5, 2.0, 2.5)):
        out.append(("zscore_fade", {"period": z, "th": th}))
    for per, th in itertools.product((14, 20), (100, 150, 200)):
        out.append(("cci_fade", {"period": per, "th": th}))
    for per, mult in ((10, 1.5), (20, 1.5), (20, 2.0)):
        out.append(("keltner_fade", {"period": per, "mult": mult}))
    for rsi, os, ob in itertools.product((3, 5, 7), (20, 25, 30), (70, 75, 80)):
        out.append(("micro_reclaim", {"rsi": rsi, "os": os, "ob": ob}))
        out.append(
            ("pin_rsi", {"rsi": rsi, "os": os, "ob": ob, "wick": 0.45, "body": 0.35})
        )
        out.append(("range_fade", {"look": 5, "rsi": rsi, "os": os, "ob": ob}))
        out.append(("range_fade", {"look": 8, "rsi": rsi, "os": os, "ob": ob}))
        out.append(("range_fade", {"look": 12, "rsi": rsi, "os": os, "ob": ob}))
    for ema in (8, 21, 34):
        out.append(("engulf", {"ema": ema}))
    out.append(("inside_break", {}))
    out.append(("macd_cross", {}))
    out.append(("heikin_flip", {}))
    for rvol, thr in itertools.product((10, 20), (1.0, 1.2, 1.4)):
        out.append(("levy_pv_confirm", {"rvol": rvol, "rvol_thr": thr}))
        out.append(("levy_accum", {"rvol": rvol, "rvol_thr": thr, "clv": 0.55}))
    for look, rvol, thr in itertools.product((8, 16), (10, 20), (1.0, 1.2)):
        out.append(("levy_regime", {"look": look, "rvol": rvol, "rvol_thr": thr}))
    return out


def exits(lots: tuple[float, ...], focused: bool = False) -> list[dict]:
    """TP free (as long as ≥$2 net); SL ≤ 2×TP. Use $ distances."""
    out = []
    # Prefer wider SL (≤2×) first — that's the high-WR side of the constraint.
    if focused:
        tp_cands = (1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
        sl_mults = (1.5, 1.75, 2.0)
        holds = (5, 8, 12, 20, 30)
    else:
        tp_cands = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0)
        sl_mults = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        holds = (3, 5, 8, 12, 20, 30, 48, 72)
    sides = (
        dict(allow_long=True, allow_short=True),
        dict(allow_long=True, allow_short=False),
        dict(allow_long=False, allow_short=True),
    )
    for lot, tp, sm, mh, side in itertools.product(lots, tp_cands, sl_mults, holds, sides):
        if net_on_tp(tp, lot) < MIN_NET_TP:
            continue
        sl = tp * sm
        if sl > 2.0 * tp + 1e-9:
            continue
        out.append(
            {
                "tp_dist": tp,
                "sl_dist": sl,
                "max_hold": mh,
                "lot": lot,
                "net_tp": net_on_tp(tp, lot),
                "sl_mult": sm,
                **side,
            }
        )
    return out


def slice_last_days(df: pd.DataFrame, days: int, warm: int = 400) -> tuple[pd.DataFrame, int, int]:
    """Return full df with indicators, plus [start_i, end_i) covering last `days` trading days."""
    dates = pd.Series(df.index.date).unique()
    if len(dates) < days:
        raise RuntimeError(f"need ≥{days} days, have {len(dates)}")
    keep_dates = set(dates[-days:])
    mask = np.array([d in keep_dates for d in df.index.date], dtype=bool)
    idxs = np.where(mask)[0]
    start_i = int(idxs[0])
    end_i = int(idxs[-1])
    # ensure warmup bars before start
    if start_i < warm:
        # shift window — prefer keeping last days; indicators already computed on full
        pass
    return df, start_i, end_i


def dump(tested: int, hits: int, best: dict | None, near: list) -> None:
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits_80": hits,
                "best_wr": best["wr"] if best else None,
                "best_name": best.get("name") if best else None,
                "best_net": best.get("net") if best else None,
                "best_tpd": best.get("tpd") if best else None,
                "near": len(near),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    SUMMARY.write_text(
        json.dumps({"tested": tested, "hits": hits, "best": best, "near": near[:30]}, indent=2, default=str)
    )


def maybe_save(row: dict) -> None:
    prev = None
    if CHAMP.exists():
        try:
            prev = json.loads(CHAMP.read_text()).get("champion")
        except Exception:
            prev = None
    key = (row["wr"], row["net"], row["tpd"], -row["dd"])
    old = (
        (
            float(prev.get("wr", 0)),
            float(prev.get("net", -1e18)),
            float(prev.get("tpd", 0)),
            -float(prev.get("dd", 1)),
        )
        if prev
        else None
    )
    if old is not None and key <= old:
        return
    CHAMP.write_text(
        json.dumps(
            {
                "rule": "≥80% WR on last 10d, TP net≥$2 after costs, SL≤2×TP, unlimited positions",
                "champion": row,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )
    log(
        f"NEW HF80 CHAMP {row['wr']:.1%} net=${row['net']:.0f} "
        f"tpd={row['tpd']:.1f} dd={row['dd']:.1%} | {row['name']}"
    )


def main(max_tests: int | None = None, passes: int = 2, focused: bool = True) -> None:
    OUT.mkdir(exist_ok=True)
    if LEADER.exists():
        LEADER.write_text("")  # fresh board for this scout
    for lot in (0.01, 0.02, 0.05):
        log(f"lot={lot}: min TP$ for ${MIN_NET_TP} net = {min_tp_dist(lot):.3f}")

    raw = load_frames(include_resampled=False)
    frames: dict[str, pd.DataFrame] = {}
    for tf in ("1m", "5m"):
        if tf not in raw:
            continue
        log(f"indicators {tf}…")
        # only need last ~15 calendar days + warmup for speed on 1m
        raw_tf = raw[tf]
        dates = pd.Series(raw_tf.index.date).unique()
        keep = set(dates[-(DAYS + 5) :])
        slim = raw_tf[[d in keep for d in raw_tf.index.date]].copy()
        frames[tf] = add_indicators(slim)

    if not frames:
        raise SystemExit("no 1m/5m frames")

    windows = {}
    for tf, df in frames.items():
        _, s, e = slice_last_days(df, DAYS, warm=200)
        # bump start if too little warmup inside slim frame
        s = max(s, 200)
        windows[tf] = (s, e)
        log(
            f"{tf} window bars={e - s} start={df.index[s]} end={df.index[min(e, len(df)-1)]} "
            f"days≈{trading_days(df.index[s:e]):.1f}"
        )

    sp = specs()
    lots = (0.01, 0.02, 0.05)
    exs = exits(lots, focused=focused)
    log(f"specs={len(sp)} exits={len(exs)} focused={focused} tfs={list(frames)}")

    # Precompute signals once
    sig_cache: list[tuple[str, dict, str, np.ndarray, np.ndarray, bool]] = []
    for kind, base in sp:
        for invert in (False, True):
            pp = {**base}
            for tf, df in frames.items():
                start_i, end_i = windows[tf]
                try:
                    if kind in LOCAL_KINDS:
                        L0, S0 = local_signals(df, kind, pp)
                    else:
                        L0, S0 = gen_signals(df, kind, pp)
                except Exception as e:
                    if len(sig_cache) < 8:
                        log(f"skip {kind}: {e}")
                    continue
                L0 = np.asarray(L0, bool)
                S0 = np.asarray(S0, bool)
                if invert:
                    L0, S0 = S0, L0
                sig_n = int(L0[start_i:end_i].sum() + S0[start_i:end_i].sum())
                if sig_n < 20:
                    continue
                sig_cache.append((kind, {**pp, **({"invert": True} if invert else {})}, tf, L0, S0, invert))
    log(f"signal cache={len(sig_cache)}")

    tested = hits = 0
    best: dict | None = None
    near: list[dict] = []

    # Probe exit: tightest legal TP @ 0.02 lot, SL=2×TP (max WR bias), dual side
    probe_tp = max(min_tp_dist(0.02), 1.3)
    probe_ex = {
        "tp_dist": probe_tp,
        "sl_dist": 2.0 * probe_tp,
        "max_hold": 12,
        "lot": 0.02,
        "net_tp": net_on_tp(probe_tp, 0.02),
        "sl_mult": 2.0,
        "allow_long": True,
        "allow_short": True,
    }

    for pass_i in range(passes):
        RNG.shuffle(sig_cache)
        RNG.shuffle(exs)
        for kind, pp, tf, L0, S0, invert in sig_cache:
            df = frames[tf]
            start_i, end_i = windows[tf]
            # Cheap probe — skip signal families that can't clear ~70% WR
            probed = simulate_multi(
                df,
                L0,
                S0,
                tp_dist=probe_ex["tp_dist"],
                sl_dist=probe_ex["sl_dist"],
                max_hold=probe_ex["max_hold"],
                lot=probe_ex["lot"],
                allow_long=True,
                allow_short=True,
                start_i=start_i,
                end_i=end_i,
            )
            tested += 1
            if probed["trades"] < 30 or probed["winrate"] < 0.70:
                if tested % 400 == 0:
                    dump(tested, hits, best, near)
                    bw = f"{best['wr']:.1%}" if best else "n/a"
                    top = f"{near[0]['wr']:.1%}" if near else "n/a"
                    log(f"hf80 tested={tested} hits80={hits} best80={bw} top75={top}")
                continue
            log(
                f"PROBE {probed['winrate']:.1%} tpd={probed['tpd']:.1f} "
                f"n={probed['trades']} | {kind} {tf} inv={invert}"
            )
            # Expand exits for promising signals
            if focused and pass_i == 0:
                ex_iter = exs
            else:
                ex_iter = RNG.sample(exs, min(len(exs), 400))
            for ex in ex_iter:
                tested += 1
                m = simulate_multi(
                    df,
                    L0,
                    S0,
                    tp_dist=ex["tp_dist"],
                    sl_dist=ex["sl_dist"],
                    max_hold=ex["max_hold"],
                    lot=ex["lot"],
                    allow_long=ex["allow_long"],
                    allow_short=ex["allow_short"],
                    start_i=start_i,
                    end_i=end_i,
                )
                if m["trades"] < 30:
                    if tested % 1000 == 0:
                        dump(tested, hits, best, near)
                    continue
                # Track near-misses ≥75% for diagnostics
                if m["winrate"] >= 0.75 and m["net"] > 0 and m["avg_win"] >= MIN_NET_TP * 0.8:
                    side_tag = (
                        "LS"
                        if ex["allow_long"] and ex["allow_short"]
                        else ("L" if ex["allow_long"] else "S")
                    )
                    name = (
                        f"{'inv|' if invert else ''}{kind}|tp{ex['tp_dist']:g}/"
                        f"sl{ex['sl_dist']:g}|h{ex['max_hold']}|{tf}|{side_tag}|"
                        f"lot{ex['lot']:g}"
                    )
                    row = {
                        "name": name,
                        "wr": m["winrate"],
                        "net": m["net"],
                        "eq": m["eq"],
                        "tpd": m["tpd"],
                        "dd": m["dd"],
                        "pf": m["pf"],
                        "trades": m["trades"],
                        "max_open": m["max_concurrent"],
                        "avg_win": m["avg_win"],
                        "avg_loss": m["avg_loss"],
                        "expectancy": m["expectancy"],
                        "net_tp_design": ex["net_tp"],
                        "tp_dist": ex["tp_dist"],
                        "sl_dist": ex["sl_dist"],
                        "sl_mult": ex["sl_mult"],
                        "lot": ex["lot"],
                        "wins_tp": m["wins_tp"],
                        "losses_sl": m["losses_sl"],
                        "time_exits": m["time_exits"],
                        "params": {"kind": kind, **pp, "tf": tf, "exit": dict(ex)},
                        "window": f"{DAYS}d",
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    if m["winrate"] >= TARGET_WR:
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        hits += 1
                        sk = (row["wr"], row["net"], row["tpd"], -row["dd"])
                        if best is None or sk > best.get("_sk", (0, 0, 0, 0)):
                            best = {**row, "_sk": sk}
                            maybe_save(row)
                        log(
                            f"HF80 HIT {m['winrate']:.1%} net=${m['net']:.0f} "
                            f"tpd={m['tpd']:.1f} open={m['max_concurrent']} "
                            f"dd={m['dd']:.1%} | {name}"
                        )
                    near.append(row)
                    near.sort(key=lambda r: (r["wr"], r["net"], r["tpd"]), reverse=True)
                    near[:] = near[:60]
                if max_tests and tested >= max_tests:
                    dump(tested, hits, best, near)
                    log(f"stop at max_tests={max_tests}")
                    return
                if tested % 400 == 0:
                    dump(tested, hits, best, near)
                    bw = f"{best['wr']:.1%}" if best else "n/a"
                    top = f"{near[0]['wr']:.1%}" if near else "n/a"
                    log(f"hf80 tested={tested} hits80={hits} best80={bw} top75={top}")
        dump(tested, hits, best, near)
        log(f"pass {pass_i+1}/{passes} done tested={tested} hits={hits}")

    dump(tested, hits, best, near)
    log(f"DONE tested={tested} hits80={hits} best={best['name'] if best else None}")
    if near:
        log(f"top near: {near[0]['wr']:.1%} net=${near[0]['net']:.0f} | {near[0]['name']}")


if __name__ == "__main__":
    main(passes=1, focused=True)
