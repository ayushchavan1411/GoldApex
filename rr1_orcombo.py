#!/usr/bin/env python3
"""OR-combine high-WR low-frequency 1:1 signals to push tpd>=1 while keeping WR>=80%."""
from __future__ import annotations
import itertools, json, random, time
from pathlib import Path
import numpy as np
from rr1_search import OUT, add_indicators, evaluate, gen_signals, load_frames, log, qualifies, save_champion, write_bot

RNG = random.Random(7)

def main():
    OUT.mkdir(exist_ok=True)
    log("=== RR1 OR-COMBO START ===")
    frames = {k: add_indicators(v) for k, v in load_frames().items()}
    # Build selective RSI/BB signal library on 5m
    tf = "5m"
    df = frames[tf]
    lib = []
    for kind, grid in [
        ("rsi_level", itertools.product([3,5,7,9], range(15,36,3), range(70,96,3), [3,5,7,9],
            [("overlap","ny"),("london","overlap","ny"),("ny",),("overlap",)])),
        ("rsi_cross", itertools.product([3,5,7,9], range(15,36,3), range(70,96,3), [0],
            [("overlap","ny"),("london","overlap","ny"),("ny",)])),
        ("bb_rsi_fade", itertools.product([(10,2.0),(20,2.0),(20,2.5)], [3,7], [20,25,30], [70,75,80],
            [("overlap","ny"),("london","overlap","ny")])),
    ]:
        for item in grid:
            if kind == "rsi_level":
                rsi,os_,ob,band,sess = item
                p = {"rsi":rsi,"os":os_,"ob":ob,"band":band,"sessions":sess}
            elif kind == "rsi_cross":
                rsi,os_,ob,_,sess = item
                p = {"rsi":rsi,"os":os_,"ob":ob,"sessions":sess}
            else:
                (period,k),rsi,os_,ob,sess = item
                p = {"period":period,"k":k,"rsi":rsi,"os":os_,"ob":ob,"sessions":sess}
            try:
                L,S = gen_signals(df, kind, p)
            except Exception:
                continue
            nsig = int(L.sum()+S.sum())
            if 25 <= nsig <= 400:
                lib.append((kind,p,L,S,nsig))
    log(f"library size={len(lib)}")
    # Pre-score singles at 1:1 exits
    exits = [{"rr":rr,"max_hold":mh,"atr":"ATR_14","allow_long":True,"allow_short":True}
             for rr in (1.0,1.2,1.5) for mh in (2,3,4)]
    singles = []
    for kind,p,L,S,nsig in lib:
        best=None
        for ex in exits:
            res=evaluate(f"s|{kind}", tf, df, L, S, ex, {"kind":kind, **p})
            if not res: continue
            if best is None or min(res.winrate,res.oos_winrate) > min(best.winrate,best.oos_winrate):
                best=res
        if best and min(best.winrate,best.oos_winrate) >= 0.72 and best.trades >= 25:
            singles.append((best, kind, p, L, S))
    singles = sorted(singles, key=lambda x: min(x[0].winrate,x[0].oos_winrate), reverse=True)[:40]
    log(f"strong singles={len(singles)} top={singles[0][0].winrate:.1%}/{singles[0][0].oos_winrate:.1%}" if singles else "no singles")

    tested=0; quals=0; best_key=(-1,)
    while True:
        # OR pairs
        for (a, ka, pa, La, Sa), (b, kb, pb, Lb, Sb) in itertools.combinations(singles[:25], 2):
            if ka == kb and pa == pb: continue
            L = La | Lb; S = Sa | Sb
            # avoid both sides same bar
            both = L & S
            L = L & ~both; S = S & ~both
            if L.sum()+S.sum() < 50: continue
            for ex in exits:
                tested += 1
                res = evaluate(f"or|{ka}+{kb}|{tf}", tf, df, L, S, ex, {"kind":"combo_or","a":ka,"b":kb,"pa":pa,"pb":pb})
                if not res: continue
                q = qualifies(
                    {"trades":res.trades,"winrate":res.winrate,"profit_factor":res.profit_factor,"net_pnl":res.net_pnl,"trades_per_day":res.trades_per_day,"max_dd":res.max_dd},
                    {"trades":res.oos_trades,"winrate":res.oos_winrate,"profit_factor":res.oos_profit_factor,"net_pnl":res.oos_net_pnl,"trades_per_day":res.oos_trades_per_day},
                )
                key=(min(res.winrate,res.oos_winrate), min(res.trades_per_day,res.oos_trades_per_day), res.trades+res.oos_trades)
                if q:
                    quals += 1
                    save_champion(res); write_bot(res)
                    log(f"QUALIFIED #{quals} WR={res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}")
                elif key > best_key and res.winrate >= 0.78 and res.oos_winrate >= 0.78:
                    best_key = key
                    save_champion(res); write_bot(res)
                    log(f"NEAR WR={res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} n={res.trades}/{res.oos_trades}")
                if tested % 500 == 0:
                    log(f"or tested={tested} quals={quals} best_minWR={best_key[0]:.1%}" if best_key[0]>=0 else f"or tested={tested} quals={quals}")
                    Path(OUT/"orcombo_progress.json").write_text(json.dumps({"tested":tested,"quals":quals,"time":time.strftime("%Y-%m-%d %H:%M:%S")},indent=2))
        # also OR three random from top
        for _ in range(200):
            picks = RNG.sample(singles[:20], 3)
            L = picks[0][3] | picks[1][3] | picks[2][3]
            S = picks[0][4] | picks[1][4] | picks[2][4]
            both = L & S; L = L & ~both; S = S & ~both
            if L.sum()+S.sum() < 55: continue
            for ex in exits:
                tested += 1
                res = evaluate("or3|5m", tf, df, L, S, ex, {"kind":"combo_or3"})
                if not res: continue
                q = qualifies(
                    {"trades":res.trades,"winrate":res.winrate,"profit_factor":res.profit_factor,"net_pnl":res.net_pnl,"trades_per_day":res.trades_per_day,"max_dd":res.max_dd},
                    {"trades":res.oos_trades,"winrate":res.oos_winrate,"profit_factor":res.oos_profit_factor,"net_pnl":res.oos_net_pnl,"trades_per_day":res.oos_trades_per_day},
                )
                if q:
                    quals += 1
                    save_champion(res); write_bot(res)
                    log(f"QUALIFIED or3 #{quals} WR={res.winrate:.1%}/{res.oos_winrate:.1%} tpd={res.trades_per_day:.2f}")
        log(f"or pass done tested={tested} quals={quals}; rebuild+loop")
        # refresh library lightly by reshuffling seed
        RNG.seed(int(time.time()) % 100000)

if __name__ == "__main__":
    main()
