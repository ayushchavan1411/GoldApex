# High-frequency high-WR bot (causal)

Never-stop search: tmux session `hf-wr-search` (`python3 -u hf_wr_search.py`).

All results below use **causal** rules: signal at bar close → fill **next open**. No same-bar BB-wick look-ahead.

## Recommended: RSI Cross 15m (daily frequency)

| | IS | OOS | Full (~60d) |
|--|--|--|--|
| Winrate | **95.3%** | **92.3%** | **94.4%** |
| Trades/day | **3.39** | **3.15** | **3.37** |
| Profit factor | — | — | **1.83** |
| Trades / P&L | — | — | **180 / +$83** on $500 |

**Rules**
- TF 15m · sessions London+Overlap+NY (08–22 UTC)
- LONG: RSI(5) crosses below 25 · SHORT: RSI(5) crosses above 80
- SL 2.0×ATR · TP 0.2×ATR · max hold 6 bars · next-open fill

```bash
python3 live_hf_wr_bot.py --backtest --strategy rsi
python3 live_hf_wr_bot.py --live --strategy rsi   # Windows + MT5
```

## Alternative: Stoch Cross 15m (higher WR, fewer trades)

| Full | |
|--|--|
| Winrate | **98.8%** |
| Trades/day | **~1.4** |
| PF | **~7.9** |
| Trades | 81 · +$75 |

```bash
python3 live_hf_wr_bot.py --backtest --strategy stoch
```

## Caveats
TP ≪ SL is why WR is high. Spread/slippage can hurt. Yahoo GC=F ≠ MT5 XAUUSD. Search selection bias remains. Past ≠ future.
