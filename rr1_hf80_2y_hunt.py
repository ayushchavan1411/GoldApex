#!/usr/bin/env python3
"""2-year hunt for ≥80% WR with IS≈OOS (causal).

Rules carried forward:
  - TP sized so clean TP nets ≥ $2 after spread + commission + slip
  - SL ≤ 2× TP
  - Unlimited concurrent positions (also probe 1-pos — can change which trades fire)
  - Gate: IS WR ≥ 80% AND OOS WR ≥ 80% AND |IS−OOS| ≤ 5pp AND both nets > 0
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rr1_hf80_10d_hunt import (
    LOCAL_KINDS,
    MIN_NET_TP,
    SLIP_PRICE,
    local_signals,
    min_tp_dist,
    net_on_tp,
    simulate_multi,
)
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
    resample_ohlc,
    simulate,
    trading_days,
)

RNG = random.Random(2210)
LEADER = OUT / "HF80_2Y_HUNT_LEADERBOARD.jsonl"
PROGRESS = OUT / "hf80_2y_hunt_progress.json"
SUMMARY = OUT / "HF80_2Y_HUNT_SUMMARY.json"
CHAMP = OUT / "HF80_2Y_HUNT_CHAMPION.json"

TARGET_WR = 0.80
MAX_GAP = 0.05
MIN_IS_N = 80
MIN_OOS_N = 35


def specs() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    sess_opts = [
        None,
        ("london",),
        ("ny",),
        ("overlap",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("london", "overlap", "ny"),
        ("asian",),
    ]
    # Extreme RSI / recover — high-WR bias
    for sess, rsi, os, ob in itertools.product(
        sess_opts, (2, 3, 5, 7, 9, 14), (5, 8, 10, 12, 15, 20, 25), (75, 80, 85, 88, 90, 92, 95)
    ):
        if os >= 40 or ob <= 60:
            continue
        base = {"rsi": rsi, "os": os, "ob": ob, "sessions": sess}
        out.append(("rsi_cross", dict(base)))
        out.append(("rsi_level", dict(base)))
        out.append(("rsi_recover", {k: v for k, v in base.items() if k != "sessions"}))
    bb_grid = [(10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5), (50, 2.0)]
    for sess, (per, k) in itertools.product(sess_opts, bb_grid):
        out.append(("bb_fade", {"period": per, "k": k, "sessions": sess}))
        out.append(("bb_mid_reject", {"period": per, "k": k, "sessions": sess}))
        out.append(
            ("bb_rsi_fade", {"period": per, "k": k, "rsi": 5, "os": 20, "ob": 80, "sessions": sess})
        )
        out.append(
            ("bb_rsi_fade", {"period": per, "k": k, "rsi": 7, "os": 15, "ob": 85, "sessions": sess})
        )
    for sess, per, os, ob in itertools.product(sess_opts[:5], (5, 9, 14), (10, 15, 20), (80, 85, 90)):
        out.append(("stoch_cross", {"period": per, "os": os, "ob": ob, "sessions": sess}))
    for sess, z, th in itertools.product(sess_opts[:4], (20, 40), (1.5, 2.0, 2.5, 3.0)):
        out.append(("zscore_fade", {"period": z, "th": th, "sessions": sess}))
    for sess, per, th in itertools.product(sess_opts[:4], (14, 20), (100, 150, 200, 250)):
        out.append(("cci_fade", {"period": per, "th": th, "sessions": sess}))
    for per, mult in ((10, 1.5), (20, 1.5), (20, 2.0)):
        out.append(("keltner_fade", {"period": per, "mult": mult}))
    for rsi, os, ob in itertools.product((3, 5, 7), (15, 20, 25), (75, 80, 85)):
        out.append(("micro_reclaim", {"rsi": rsi, "os": os, "ob": ob}))
        out.append(("pin_rsi", {"rsi": rsi, "os": os, "ob": ob, "wick": 0.5, "body": 0.3}))
        out.append(("range_fade", {"look": 8, "rsi": rsi, "os": os, "ob": ob}))
        out.append(("range_fade", {"look": 12, "rsi": rsi, "os": os, "ob": ob}))
    for wick, body in itertools.product((0.55, 0.65, 0.75), (0.2, 0.3)):
        out.append(("wick_reject", {"wick": wick, "body": body}))
    out.append(("micro_bounce", {}))
    for ema in (8, 21, 34, 50):
        out.append(("engulf", {"ema": ema}))
        out.append(("ema_wick", {"ema": ema, "rsi": 5, "os": 25, "ob": 75}))
    for look, rvol, thr in itertools.product((8, 16, 24), (10, 20), (1.0, 1.2, 1.4)):
        out.append(("levy_regime", {"look": look, "rvol": rvol, "rvol_thr": thr}))
        out.append(("levy_accum", {"rvol": rvol, "rvol_thr": thr, "clv": 0.55}))
    return out


def exits() -> list[dict]:
    out = []
    # Larger lots → smaller $ TP still clears $2 → more room for high WR scalps
    lots = (0.01, 0.02, 0.05, 0.1)
    tps = (0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0)
    sl_mults = (1.25, 1.5, 1.75, 2.0)  # bias toward wider SL for WR
    holds = (6, 8, 12, 16, 20, 30, 48, 72, 96)
    sides = (
        dict(allow_long=True, allow_short=True),
        dict(allow_long=True, allow_short=False),
        dict(allow_long=False, allow_short=True),
    )
    for lot, tp, sm, mh, side in itertools.product(lots, tps, sl_mults, holds, sides):
        if net_on_tp(tp, lot) < MIN_NET_TP:
            continue
        sl = tp * sm
        if sl > 2.0 * tp + 1e-12:
            continue
        out.append(
            {
                "tp_dist": tp,
                "sl_dist": sl,
                "max_hold": mh,
                "lot": lot,
                "sl_mult": sm,
                "net_tp": net_on_tp(tp, lot),
                **side,
            }
        )
    return out


def simulate_one(
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
    start_i: int = 0,
    end_i: int | None = None,
) -> dict:
    """Single-position causal sim with $ distances (reuse rr1_search.simulate via pct proxy).

    Convert $ distances to approx pct at mid price so we can use the battle-tested 1-pos engine.
    """
    n = len(df)
    if end_i is None:
        end_i = n - 2
    # Use absolute $ via a fake pct relative to a fixed ref — better: custom loop like multi but 1-pos.
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    times = df.index
    oz = lot * CONTRACT_SIZE
    comm = COMMISSION_PER_LOT_RT * lot
    spr = SPREAD_PRICE
    slip = SLIP_PRICE
    capital = float(START_CAP)
    peak = capital
    max_dd = 0.0
    pnls: list[float] = []
    i = start_i
    while i < end_i and i + 1 < n:
        if capital < START_CAP * 0.15:
            break
        gl = allow_long and bool(long_e[i])
        gs = allow_short and bool(short_e[i])
        if gl == gs:
            i += 1
            continue
        raw = opens[i + 1]
        if not np.isfinite(raw):
            i += 1
            continue
        direction = "LONG" if gl else "SHORT"
        if direction == "LONG":
            entry = raw + spr / 2.0 + slip
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            entry = raw - spr / 2.0 - slip
            sl = entry + sl_dist
            tp = entry - tp_dist
        j = i + 1
        bars = 0
        hit = False
        exit_px = entry
        while j < n and bars < max_hold:
            hi, lo = highs[j], lows[j]
            bars += 1
            if direction == "LONG":
                if lo <= sl:
                    exit_px, hit = sl, True
                    break
                if hi >= tp:
                    exit_px, hit = tp, True
                    break
            else:
                if hi >= sl:
                    exit_px, hit = sl, True
                    break
                if lo <= tp:
                    exit_px, hit = tp, True
                    break
            j += 1
        if not hit:
            exit_px = closes[min(j, n - 1)]
        if direction == "LONG":
            exit_px = exit_px - spr / 2.0 - slip
            gross = (exit_px - entry) * oz
        else:
            exit_px = exit_px + spr / 2.0 + slip
            gross = (entry - exit_px) * oz
        pnl = gross - comm
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0.0)
        pnls.append(pnl)
        i = j + 1 if j > i else i + 1

    if not pnls:
        return {
            "trades": 0,
            "winrate": 0.0,
            "tpd": 0.0,
            "eq": capital,
            "net": 0.0,
            "dd": max_dd,
            "pf": 0.0,
            "max_concurrent": 1,
            "avg_win": 0.0,
            "avg_loss": 0.0,
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
        "max_concurrent": 1,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(arr.mean()),
    }


def dump(tested: int, hits: int, best: dict | None, near: list, probes: int = 0) -> None:
    PROGRESS.write_text(
        json.dumps(
            {
                "tested": tested,
                "probes": probes,
                "hits_80": hits,
                "best_min_wr": (min(best["is_wr"], best["oos_wr"]) if best else None),
                "best_name": best.get("name") if best else None,
                "best_gap": best.get("wr_gap") if best else None,
                "near": len(near),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "tested": tested,
                "hits": hits,
                "best": best,
                "near": near[:25],
                "gate": f"IS&OOS>={TARGET_WR:.0%} gap<={MAX_GAP:.0%} nets>0 TP_net>=${MIN_NET_TP}",
            },
            indent=2,
            default=str,
        )
    )


def maybe_save(row: dict) -> None:
    prev = None
    if CHAMP.exists():
        try:
            prev = json.loads(CHAMP.read_text()).get("champion")
        except Exception:
            prev = None
    key = (
        min(row["is_wr"], row["oos_wr"]),
        -row["wr_gap"],
        row["oos_net"],
        row["full_net"],
        -row["full_dd"],
    )
    old = (
        (
            min(float(prev["is_wr"]), float(prev["oos_wr"])),
            -float(prev["wr_gap"]),
            float(prev["oos_net"]),
            float(prev["full_net"]),
            -float(prev.get("full_dd", 1)),
        )
        if prev
        else None
    )
    if old is not None and key <= old:
        return
    CHAMP.write_text(
        json.dumps(
            {
                "rule": "2y IS&OOS ≥80% WR, gap≤5pp, TP net≥$2, SL≤2×TP, causal next-open",
                "champion": row,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )
    log(
        f"NEW 2Y-80 CHAMP minWR={min(row['is_wr'], row['oos_wr']):.1%} "
        f"IS={row['is_wr']:.1%} OOS={row['oos_wr']:.1%} gap={row['wr_gap']:.1%} "
        f"netOOS=${row['oos_net']:.0f} | {row['name']}"
    )


def make_signals(df, kind, pp):
    base = {k: v for k, v in pp.items() if k != "invert"}
    if kind in LOCAL_KINDS:
        # local kinds ignore sessions
        return local_signals(df, kind, base)
    L, S = gen_signals(df, kind, base)
    return np.asarray(L, bool), np.asarray(S, bool)


def main(passes: int = 4) -> None:
    OUT.mkdir(exist_ok=True)
    LEADER.write_text("")
    for lot in (0.01, 0.02, 0.05, 0.1):
        log(f"lot={lot}: min TP$ for ${MIN_NET_TP} = {min_tp_dist(lot):.3f}")

    raw = load_frames(include_resampled=True)
    frames: dict[str, pd.DataFrame] = {}
    for tf in ("15m", "30m", "1h", "2h"):
        if tf in raw:
            log(f"indicators {tf}…")
            frames[tf] = add_indicators(raw[tf])
    # also 5m but subsample exits harder (slow)
    if "5m" in raw:
        log("indicators 5m…")
        frames["5m"] = add_indicators(raw["5m"])

    windows = {}
    for tf, df in frames.items():
        n = len(df)
        warm = 300 if tf != "5m" else 500
        split = int(n * 0.70)
        windows[tf] = (warm, split, n - 2)
        log(f"{tf}: bars={n} {df.index[0]}→{df.index[-1]} split={df.index[split]}")

    sp = specs()
    exs = exits()
    log(f"specs={len(sp)} exits={len(exs)} tfs={list(frames)}")

    # Build signal cache (skip 5m for first passes to move faster)
    cache = []
    for kind, base in sp:
        for invert in (False, True):
            pp = {**base}
            if invert:
                pp["invert"] = True
            for tf, df in frames.items():
                if tf == "5m":
                    continue  # add later
                warm, split, end = windows[tf]
                try:
                    L, S = make_signals(df, kind, pp)
                except Exception:
                    continue
                if invert:
                    L, S = S, L
                nsig = int(L[warm:end].sum() + S[warm:end].sum())
                if nsig < 60:
                    continue
                cache.append((kind, pp, tf, L, S, invert, nsig))
    log(f"signal cache (no 5m)={len(cache)}")

    tested = hits = probes = 0
    best: dict | None = None
    near: list[dict] = []
    # track best even below 80 for progress
    best_any: dict | None = None

    modes = ("multi", "one")

    for pass_i in range(passes):
        RNG.shuffle(cache)
        # Prefer extreme RSI recover / level first on early passes
        if pass_i == 0:
            cache.sort(key=lambda x: (0 if x[0] in ("rsi_recover", "rsi_level", "rsi_cross", "bb_fade") else 1, -x[6]))
        for kind, pp, tf, L, S, invert, nsig in cache:
            df = frames[tf]
            warm, split, end = windows[tf]
            # Probe: SL=2×TP, dual or the side that matches nsig bias
            p_tp = max(min_tp_dist(0.05), 1.0)
            probes += 1
            probe = simulate_multi(
                df, L, S,
                tp_dist=p_tp, sl_dist=2 * p_tp, max_hold=30, lot=0.05,
                allow_long=True, allow_short=True, start_i=warm, end_i=end,
            )
            # Also one-pos probe
            probe1 = simulate_one(
                df, L, S,
                tp_dist=p_tp, sl_dist=2 * p_tp, max_hold=30, lot=0.05,
                allow_long=True, allow_short=True, start_i=warm, end_i=end,
            )
            best_probe_wr = max(probe["winrate"], probe1["winrate"])
            if probe["trades"] < 50 and probe1["trades"] < 40:
                continue
            if best_probe_wr < 0.74:
                continue
            log(
                f"PROBE {best_probe_wr:.1%} multi={probe['winrate']:.1%}/{probe['trades']} "
                f"one={probe1['winrate']:.1%}/{probe1['trades']} | {kind} {tf} inv={invert} nsig={nsig}"
            )

            # Expand exits — denser around high-WR region
            prefer = [e for e in exs if e["sl_mult"] >= 1.75 and e["lot"] in (0.02, 0.05, 0.1)]
            pool = prefer if prefer else exs
            n_sample = 220 if pass_i == 0 else 120
            sample = RNG.sample(pool, min(len(pool), n_sample))

            for ex in sample:
                for mode in modes:
                    tested += 1
                    sim = simulate_multi if mode == "multi" else simulate_one
                    is_m = sim(
                        df, L, S,
                        tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                        lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                        start_i=warm, end_i=split,
                    )
                    if is_m["trades"] < MIN_IS_N or is_m["winrate"] < 0.78 or is_m["net"] <= 0:
                        continue
                    oos = sim(
                        df, L, S,
                        tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                        lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                        start_i=split, end_i=end,
                    )
                    if oos["trades"] < MIN_OOS_N or oos["winrate"] < 0.78 or oos["net"] <= 0:
                        continue
                    gap = abs(is_m["winrate"] - oos["winrate"])
                    if gap > MAX_GAP:
                        continue
                    if is_m["avg_win"] < MIN_NET_TP * 0.75 or oos["avg_win"] < MIN_NET_TP * 0.75:
                        continue

                    full = sim(
                        df, L, S,
                        tp_dist=ex["tp_dist"], sl_dist=ex["sl_dist"], max_hold=ex["max_hold"],
                        lot=ex["lot"], allow_long=ex["allow_long"], allow_short=ex["allow_short"],
                        start_i=warm, end_i=end,
                    )
                    side = (
                        "LS"
                        if ex["allow_long"] and ex["allow_short"]
                        else ("L" if ex["allow_long"] else "S")
                    )
                    name = (
                        f"{'inv|' if invert else ''}{kind}|tp{ex['tp_dist']:g}/sl{ex['sl_dist']:g}|"
                        f"h{ex['max_hold']}|{tf}|{side}|lot{ex['lot']:g}|{mode}"
                    )
                    row = {
                        "name": name,
                        "mode": mode,
                        "kind": kind,
                        "params": pp,
                        "exit": ex,
                        "invert": invert,
                        "tf": tf,
                        "is_wr": is_m["winrate"],
                        "oos_wr": oos["winrate"],
                        "full_wr": full["winrate"],
                        "wr_gap": gap,
                        "is_n": is_m["trades"],
                        "oos_n": oos["trades"],
                        "full_n": full["trades"],
                        "is_net": is_m["net"],
                        "oos_net": oos["net"],
                        "full_net": full["net"],
                        "is_dd": is_m["dd"],
                        "oos_dd": oos["dd"],
                        "full_dd": full["dd"],
                        "tpd": full["tpd"],
                        "pf": full["pf"],
                        "max_open": full.get("max_concurrent", 1),
                        "avg_win": full["avg_win"],
                        "net_tp_design": ex["net_tp"],
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Track best-any ≥78
                    sk_any = (min(row["is_wr"], row["oos_wr"]), -gap, row["oos_net"])
                    if best_any is None or sk_any > best_any.get("_sk", (0, 0, 0)):
                        best_any = {**row, "_sk": sk_any}

                    if min(row["is_wr"], row["oos_wr"]) >= TARGET_WR:
                        with open(LEADER, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        hits += 1
                        sk = (
                            min(row["is_wr"], row["oos_wr"]),
                            -gap,
                            row["oos_net"],
                            row["full_net"],
                            -row["full_dd"],
                        )
                        if best is None or sk > best.get("_sk", (0, 0, 0, 0, 0)):
                            best = {**row, "_sk": sk}
                            maybe_save(row)
                        near.append(row)
                        near.sort(
                            key=lambda r: (min(r["is_wr"], r["oos_wr"]), -r["wr_gap"], r["oos_net"]),
                            reverse=True,
                        )
                        near[:] = near[:40]
                        log(
                            f"HIT80 IS {is_m['winrate']:.1%} OOS {oos['winrate']:.1%} "
                            f"gap={gap:.1%} netOOS=${oos['net']:.0f} FULL=${full['net']:.0f} "
                            f"dd={full['dd']:.1%} | {name}"
                        )

                if tested % 400 == 0:
                    dump(tested, hits, best, near, probes)
                    ba = (
                        f"{min(best_any['is_wr'], best_any['oos_wr']):.1%}"
                        if best_any
                        else "n/a"
                    )
                    bh = f"{min(best['is_wr'], best['oos_wr']):.1%}" if best else "n/a"
                    log(f"hunt tested={tested} hits80={hits} best80={bh} bestAny78+={ba}")

        dump(tested, hits, best, near, probes)
        log(f"pass {pass_i+1}/{passes} done tested={tested} hits={hits}")

        # After pass 0, if no hits, inject 5m extreme RSI into cache
        if pass_i == 0 and hits == 0 and "5m" in frames:
            log("Injecting selective 5m extreme RSI into cache…")
            df = frames["5m"]
            warm, split, end = windows["5m"]
            extra = []
            for kind, base in sp:
                if kind not in ("rsi_cross", "rsi_level", "rsi_recover", "bb_fade", "stoch_cross"):
                    continue
                if base.get("sessions") not in (None, ("overlap",), ("london", "overlap"), ("ny",)):
                    continue
                for invert in (False, True):
                    pp = {**base}
                    if invert:
                        pp["invert"] = True
                    try:
                        L, S = make_signals(df, kind, pp)
                    except Exception:
                        continue
                    if invert:
                        L, S = S, L
                    nsig = int(L[warm:end].sum() + S[warm:end].sum())
                    if nsig < 100:
                        continue
                    extra.append((kind, pp, "5m", L, S, invert, nsig))
            RNG.shuffle(extra)
            cache.extend(extra[:800])
            log(f"cache now {len(cache)} (added {min(800, len(extra))} 5m)")

    dump(tested, hits, best, near, probes)
    # Persist best_any even if <80
    payload = {
        "tested": tested,
        "hits_80": hits,
        "best_80": best,
        "best_any_ge78": best_any,
        "near": near[:25],
        "gate": f"IS&OOS>={TARGET_WR:.0%} gap<={MAX_GAP:.0%}",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, default=str))
    log(f"DONE hits80={hits} best={best['name'] if best else None}")
    if best_any and not best:
        log(
            f"Closest: IS {best_any['is_wr']:.1%} OOS {best_any['oos_wr']:.1%} "
            f"gap={best_any['wr_gap']:.1%} | {best_any['name']}"
        )


if __name__ == "__main__":
    main(passes=3)
