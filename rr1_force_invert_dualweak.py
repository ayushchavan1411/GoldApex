#!/usr/bin/env python3
"""One-shot: take genuine dual-weak low-WR bases (both IS+OOS weak, n floors)
and invert them across the fade exit grid. Writes GENUINE_LOWWR_INVERT.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from rr1_search import OUT, add_indicators, evaluate, gen_signals, load_frames, log, year_stable

LOW_BOARD = OUT / "LOWWR_BASES.jsonl"
OUT_JSON = OUT / "GENUINE_LOWWR_INVERT.json"
INV_BOARD = OUT / "LOWWR_INVERT_LEADERBOARD.jsonl"

MIN_N, MIN_OOS = 80, 30
TOP_BASES = 25


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


def main():
    bases = []
    seen = set()
    with open(LOW_BOARD) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("n", 0) < MIN_N or r.get("oos_n", 0) < MIN_OOS:
                continue
            if r.get("wr", 1) > 0.48 or r.get("oos_wr", 1) > 0.48:
                continue
            r["max_wr"] = max(r["wr"], r["oos_wr"])
            r["min_wr"] = min(r["wr"], r["oos_wr"])
            key = (
                r.get("kind"),
                r.get("tf"),
                json.dumps({k: v for k, v in r.get("params", {}).items() if k not in ("kind", "exit")}, sort_keys=True, default=str),
            )
            if key in seen:
                continue
            seen.add(key)
            bases.append(r)

    bases.sort(key=lambda r: (r["max_wr"], r["min_wr"], -(r["n"] + r["oos_n"])))
    bases = bases[:TOP_BASES]
    log(f"force-invert dual-weak bases={len(bases)} worst maxWR={bases[0]['max_wr']:.1%}" if bases else "no bases")

    tfs_needed = sorted({b["tf"] for b in bases})
    raw = load_frames(include_resampled=True)
    frames = {}
    for tf in tfs_needed:
        if tf not in raw:
            continue
        log(f"indicators {tf}…")
        frames[tf] = add_indicators(raw[tf])

    exs = exits_inv()
    results = []
    for i, brow in enumerate(bases, 1):
        kind = brow["kind"]
        tf = brow["tf"]
        if tf not in frames:
            continue
        df = frames[tf]
        base_p = {k: v for k, v in brow["params"].items() if k not in ("kind", "exit")}
        if isinstance(base_p.get("sessions"), list):
            base_p["sessions"] = tuple(base_p["sessions"])
        inv_p = {**base_p, "invert": True}
        try:
            Li, Si = gen_signals(df, kind, inv_p)
        except Exception as e:
            log(f"skip {brow['name']}: {e}")
            continue
        best = None
        for ex in exs:
            res = evaluate(f"inv|{kind}|{tf}", tf, df, Li, Si, dict(ex), {"kind": kind, **inv_p})
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
                "base_max_wr": brow["max_wr"],
                "base_min_wr": brow["min_wr"],
                "base_wr": brow["wr"],
                "base_oos_wr": brow["oos_wr"],
                "base_name": brow["name"],
                "base_n": brow["n"],
                "base_oos_n": brow["oos_n"],
                "params": res.params,
                "year_ok": year_stable(df, Li, Si, dict(ex)),
            }
            with open(INV_BOARD, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            if best is None or row["min_wr"] > best["min_wr"] or (
                row["min_wr"] == best["min_wr"] and row["n"] + row["oos_n"] > best["n"] + best["oos_n"]
            ):
                best = row
        if best:
            results.append(best)
            log(
                f"[{i}/{len(bases)}] base maxWR={brow['max_wr']:.1%} {brow['wr']:.1%}/{brow['oos_wr']:.1%} "
                f"→ inv minWR={best['min_wr']:.1%} {best['wr']:.1%}/{best['oos_wr']:.1%} "
                f"n={best['n']}/{best['oos_n']} tpd={best['tpd']:.2f} year={best['year_ok']} | {best['name']}"
            )

    results.sort(key=lambda r: (r["min_wr"], r["n"] + r["oos_n"]), reverse=True)
    payload = {
        "n_bases": len(bases),
        "n_inverted": len(results),
        "worst_bases": [
            {
                "name": b["name"],
                "wr": b["wr"],
                "oos_wr": b["oos_wr"],
                "max_wr": b["max_wr"],
                "min_wr": b["min_wr"],
                "n": b["n"],
                "oos_n": b["oos_n"],
                "tpd": b.get("tpd"),
                "params": b.get("params"),
                "tf": b.get("tf"),
                "kind": b.get("kind"),
            }
            for b in bases
        ],
        "best_inverts": results[:25],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    log(f"wrote {OUT_JSON} best_inv={results[0]['min_wr']:.1%}" if results else f"wrote {OUT_JSON} empty")


if __name__ == "__main__":
    main()
