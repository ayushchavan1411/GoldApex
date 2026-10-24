# Strategy status (after look-ahead fix)

**Read `LOOKAHEAD_CORRECTION.md` first.** BB/triple “champions” were contaminated by same-bar BB look-ahead and are invalidated.

## Still standing: Asian-Range Fade · 1h

Causal rules (prior range only + next-bar open fill):

| | Full (~2.4y) | IS 70% | OOS 30% |
|--|--|--|--|
| Winrate | **91.9%** | 90.7% | 94.9% |
| Profit factor | **1.88** | — | — |
| Trades | 136 | — | — |
| Net P&L | **+$95** on $500 | — | — |

### Rules (causal)
1. TF: 1h gold  
2. Session: NY Overlap 13:00–17:00 UTC (bar open hour)  
3. `range_hi/lo = rolling(40) high/low shifted by 1` (excludes signal bar)  
4. LONG: Low ≤ range_lo AND Close > range_lo AND RSI(7) < 25  
5. SHORT: High ≥ range_hi AND Close < range_hi AND RSI(7) > 70  
6. Fill: **next bar open**  
7. SL 2×ATR(14) · TP 0.3×ATR(14) · max hold 12  

```bash
python3 strategy_asian_fade_1h.py
```

### Frequency
~0.2 trades per gold trading day (~1 per week).

## Invalidated
- 15m BB+RSI and Triple-confirm published earlier — look-ahead on same-bar bands.

## Caveats
TP ≪ SL drives WR. Spread/slippage can erase edge. GC=F proxy. Search selection bias remains.
