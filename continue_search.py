#!/usr/bin/env python3
"""
Continue searching for BETTER strategies than the current winner.
Never exits until killed. Tracks champion by composite score requiring:
  WR>=80% IS, WR>=78% OOS, PF>=1.25 IS, PF>=1.15 OOS, positive PnL both,
  and prefers higher expectancy / lower DD / more trades.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from strategy_search import (
    OOS_MIN_TRADES,
    OOS_PF,
    OOS_WR,
    TARGET_PF,
    TARGET_WR,
    MIN_TRADES,
    OUT,
    Result,
    add_indicators,
    append_candidate,
    evaluate,
    exit_grid,
    gen_signals,
    load_frames,
    log,
    meets_target,
    save_progress,
    strategy_param_space,
    try_combo_phase,
)

CHAMPION_PATH = OUT / "CHAMPION.json"
CONSOLE = OUT / "continue_console.log"


def champion_key(r: Result) -> tuple:
    return (
        1 if meets_target(
            {"trades": r.trades, "winrate": r.winrate, "profit_factor": r.profit_factor, "net_pnl": r.net_pnl},
            {
                "trades": r.oos_trades or 0,
                "winrate": r.oos_winrate or 0,
                "profit_factor": r.oos_profit_factor or 0,
                "net_pnl": r.oos_net_pnl or 0,
            },
        )
        else 0,
        r.oos_winrate or 0,
        r.winrate,
        r.oos_profit_factor or 0,
        r.profit_factor,
        r.oos_net_pnl or 0,
        r.net_pnl,
        -r.max_dd,
        r.trades + (r.oos_trades or 0),
    )


def main() -> None:
    log("=== CONTINUE SEARCH (never stop) ===")
    frames_raw = load_frames()
    dfs = {k: add_indicators(v) for k, v in frames_raw.items()}

    champion: Result | None = None
    if (OUT / "WINNER.json").exists():
        d = json.loads((OUT / "WINNER.json").read_text())
        champion = Result(**d)
        log(f"Seed champion: WR={champion.winrate:.1%} OOS={champion.oos_winrate:.1%} PF={champion.profit_factor:.2f}")

    best: list[Result] = []
    if (OUT / "top50.json").exists():
        for d in json.loads((OUT / "top50.json").read_text()):
            best.append(Result(**d))

    tested = 0
    if (OUT / "progress.json").exists():
        tested = int(json.loads((OUT / "progress.json").read_text()).get("tested", 0))

    phase = 0
    stats = {
        "mode": "continue",
        "tested": tested,
        "phase": phase,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_wr": TARGET_WR,
        "best_score": champion.score if champion else -1e9,
    }

    while True:
        log(f"----- CONTINUE PHASE {phase} -----")
        space = strategy_param_space(min(phase, 3))
        exits = exit_grid(min(phase, 3))
        max_exits = 12 + phase * 4

        for idx, (kind, params) in enumerate(space):
            for tf, df in dfs.items():
                if phase == 0 and tf == "5m":
                    continue
                try:
                    long_e, short_e = gen_signals(df, kind, params)
                except Exception:
                    continue
                if long_e.sum() + short_e.sum() < 15:
                    continue

                probe = exits[:: max(1, len(exits) // max_exits)][:max_exits]
                for exit_p in probe:
                    tested += 1
                    res = evaluate(
                        f"{kind}|{tf}|c{phase}",
                        tf,
                        df,
                        long_e,
                        short_e,
                        exit_p,
                        {"kind": kind, **params},
                    )
                    if not res:
                        continue
                    best.append(res)
                    if champion is None or champion_key(res) > champion_key(champion):
                        if res.winrate >= 0.75 and (res.oos_winrate or 0) >= 0.70:
                            champion = res
                            log(
                                f"NEW CHAMPION WR={res.winrate:.1%} OOS_WR={res.oos_winrate:.1%} "
                                f"PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f} "
                                f"tr={res.trades}/{res.oos_trades} pnl={res.net_pnl:.1f}/{res.oos_net_pnl:.1f} "
                                f"{res.name}"
                            )
                            append_candidate(res)
                            with open(CHAMPION_PATH, "w", encoding="utf-8") as f:
                                json.dump(asdict(res), f, indent=2)
                            # also refresh WINNER if meets hard target
                            if meets_target(
                                {
                                    "trades": res.trades,
                                    "winrate": res.winrate,
                                    "profit_factor": res.profit_factor,
                                    "net_pnl": res.net_pnl,
                                },
                                {
                                    "trades": res.oos_trades or 0,
                                    "winrate": res.oos_winrate or 0,
                                    "profit_factor": res.oos_profit_factor or 0,
                                    "net_pnl": res.oos_net_pnl or 0,
                                },
                            ):
                                with open(OUT / "WINNER.json", "w", encoding="utf-8") as f:
                                    json.dump(asdict(res), f, indent=2)
                                log("Updated WINNER.json")

                # deepen if promising
                if champion and kind == champion.params.get("kind"):
                    for exit_p in exits:
                        tested += 1
                        res = evaluate(
                            f"{kind}|{tf}|deepc",
                            tf,
                            df,
                            long_e,
                            short_e,
                            exit_p,
                            {"kind": kind, **params},
                        )
                        if not res:
                            continue
                        if champion_key(res) > champion_key(champion):
                            champion = res
                            append_candidate(res)
                            with open(CHAMPION_PATH, "w", encoding="utf-8") as f:
                                json.dump(asdict(res), f, indent=2)
                            log(
                                f"NEW CHAMPION (deep) WR={res.winrate:.1%} OOS_WR={res.oos_winrate:.1%} "
                                f"PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f}"
                            )

            if idx % 20 == 0:
                stats.update(
                    {
                        "tested": tested,
                        "phase": phase,
                        "champion": asdict(champion) if champion else None,
                        "best_score": champion.score if champion else -1e9,
                    }
                )
                save_progress(stats, best)
                log(
                    f"continue phase={phase} idx={idx}/{len(space)} tested={tested} "
                    f"champ_wr={champion.winrate if champion else 0:.1%} "
                    f"champ_oos={champion.oos_winrate if champion else 0:.1%}"
                )

        # combos
        top = sorted(best, key=lambda r: r.score, reverse=True)[:50]
        tested, _ = try_combo_phase(dfs, top, tested, best, stats)
        # if combo found meeting target better than champion, try_combo returns winner;
        # re-scan best for new champion
        for r in best[-500:]:
            if champion is None or champion_key(r) > champion_key(champion):
                if r.winrate >= 0.75:
                    champion = r
                    with open(CHAMPION_PATH, "w", encoding="utf-8") as f:
                        json.dump(asdict(r), f, indent=2)
                    log(f"Champion from combo/pool WR={r.winrate:.1%} OOS={r.oos_winrate:.1%}")

        phase += 1
        stats.update({"tested": tested, "phase": phase, "champion": asdict(champion) if champion else None})
        save_progress(stats, best)
        log(f"Phase complete. tested={tested}. Looping forever for better champs...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
