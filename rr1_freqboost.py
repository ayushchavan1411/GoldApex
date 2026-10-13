#!/usr/bin/env python3
"""OR-combine the high-WR low-frequency RSI+MACD family with other signals to push tpd toward 1.0 while keeping WR high."""
from __future__ import annotations
import itertools, json, random, time
from pathlib import Path
from rr1_search import OUT, add_indicators, evaluate, gen_signals, load_frames, log, qualifies, save_champion, write_bot, year_stable

RNG = random.Random(77)

def main():
    OUT.mkdir(exist_ok=True)
    log("=== FREQBOOST around high-WR RSI+MACD ===")
    df = add_indicators(load_frames()["5m"])
    # seed family variants near 71% champ
    seeds = []
    for rsi, os_, ob, sess, hours, side in itertools.product(
        [7, 14], [40, 45, 50], [55, 60, 65],
        [("asian",), ("asian", "london"), None, ("london", "overlap")],
        [None, tuple(range(7, 22)), tuple(range(0, 12)), tuple(range(8, 17))],
        [dict(allow_long=True, allow_short=False), dict(allow_long=True, allow_short=True), dict(allow_long=False, allow_short=True)],
    ):
        p = {"rsi": rsi, "os": os_, "ob": ob, "sessions": sess}
        if hours: p["hours"] = hours
        seeds.append(("rsi_macd", p, side))
    # partners to OR with
    partners = []
    for kind, p in [
        ("fvg_reclaim", {"sessions": ("asian", "london", "overlap")}),
        ("liq_sweep", {"swing": 20, "wick": 0.45, "sessions": ("asian", "london", "overlap", "ny")}),
        ("ema_pull_rsi", {"fast": 9, "slow": 21, "rsi": 7, "os": 35, "ob": 65, "sessions": None}),
        ("heikin_ema", {"ema": 50, "sessions": None}),
        ("ichimoku_tk", {"sessions": ("london", "overlap", "ny")}),
        ("keltner_fade", {"period": 20, "mult": 2.0, "sessions": None}),
        ("zscore_fade", {"period": 20, "th": 2.0, "sessions": None}),
        ("orb_fade", {"orb": "LONDON", "sessions": ("london", "overlap")}),
        ("rsi_level", {"rsi": 7, "os": 25, "ob": 75, "band": 5, "sessions": None}),
        ("smc_confluence", {"swing": 20, "sessions": ("overlap", "ny")}),
    ]:
        partners.append((kind, p))

    exits = [{"rr": rr, "max_hold": mh, "atr": atr, **side}
             for rr in (0.8, 1.0, 1.2) for mh in (4, 6, 8, 12)
             for atr in ("ATR_14", "ATR_7")
             for side in ({},)]  # side from seed

    tested = 0; quals = 0; best = None
    # cache partners
    pc = []
    for kind, p in partners:
        try:
            L, S = gen_signals(df, kind, p)
            if L.sum() + S.sum() >= 40:
                pc.append((kind, p, L, S))
        except Exception:
            pass
    log(f"partners={len(pc)} seeds={len(seeds)}")

    while True:
        for kind, p, side in seeds:
            try:
                L0, S0 = gen_signals(df, kind, p)
            except Exception:
                continue
            # solo first with side
            for ex0 in [{"rr": rr, "max_hold": mh, "atr": atr, **side}
                        for rr in (1.0, 1.2) for mh in (6, 8, 12) for atr in ("ATR_7", "ATR_14")]:
                tested += 1
                res = evaluate(f"fb|{kind}", "5m", df, L0, S0, ex0, {"kind": kind, **p})
                if not res: continue
                q = qualifies({"trades":res.trades,"winrate":res.winrate,"profit_factor":res.profit_factor,"net_pnl":res.net_pnl,"trades_per_day":res.trades_per_day,"max_dd":res.max_dd},
                              {"trades":res.oos_trades,"winrate":res.oos_winrate,"profit_factor":res.oos_profit_factor,"net_pnl":res.oos_net_pnl,"trades_per_day":res.oos_trades_per_day})
                if q: q = year_stable(df, L0, S0, ex0)
                score = (min(res.winrate, res.oos_winrate), min(res.trades_per_day, res.oos_trades_per_day))
                if q or (res.winrate >= 0.68 and res.oos_winrate >= 0.68):
                    if best is None or score > (min(best.winrate,best.oos_winrate), min(best.trades_per_day,best.oos_trades_per_day)):
                        best = res; save_champion(res); write_bot(res)
                        if q: quals += 1
                        log(f"{'QUAL ' if q else ''}NEAR WR={res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}")
            # OR with partners
            for pk, pp, Lp, Sp in pc:
                L = L0 | Lp; S = S0 | Sp
                both = L & S; L, S = L & ~both, S & ~both
                if L.sum() + S.sum() < 80: continue
                for ex0 in [{"rr": rr, "max_hold": mh, "atr": "ATR_14", **side} for rr in (1.0, 1.2) for mh in (6, 8)]:
                    tested += 1
                    res = evaluate(f"fbOR|{kind}+{pk}", "5m", df, L, S, ex0, {"kind": "combo_or", "a": kind, "b": pk})
                    if not res: continue
                    q = qualifies({"trades":res.trades,"winrate":res.winrate,"profit_factor":res.profit_factor,"net_pnl":res.net_pnl,"trades_per_day":res.trades_per_day,"max_dd":res.max_dd},
                                  {"trades":res.oos_trades,"winrate":res.oos_winrate,"profit_factor":res.oos_profit_factor,"net_pnl":res.oos_net_pnl,"trades_per_day":res.oos_trades_per_day})
                    if q: q = year_stable(df, L, S, ex0)
                    # prefer high WR with tpd approaching 1
                    useful = res.winrate >= 0.65 and res.oos_winrate >= 0.65 and res.trades_per_day >= 0.5
                    if q or useful:
                        score = (min(res.winrate, res.oos_winrate), min(res.trades_per_day, res.oos_trades_per_day))
                        bscore = (min(best.winrate,best.oos_winrate), min(best.trades_per_day,best.oos_trades_per_day)) if best else (-1,)
                        if best is None or score > bscore:
                            best = res; save_champion(res); write_bot(res)
                            if q: quals += 1
                            log(f"{'QUAL ' if q else ''}OR WR={res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} {kind}+{pk}")
            if tested % 500 == 0:
                log(f"freqboost tested={tested} quals={quals} best={(f'{best.winrate:.1%}/{best.oos_winrate:.1%} tpd{best.trades_per_day:.2f}' if best else 'n/a')}")
                Path(OUT/"freqboost_progress.json").write_text(json.dumps({"tested":tested,"quals":quals,"time":time.strftime("%Y-%m-%d %H:%M:%S")},indent=2))
        log(f"freqboost pass done tested={tested}; loop")

if __name__ == "__main__":
    main()
