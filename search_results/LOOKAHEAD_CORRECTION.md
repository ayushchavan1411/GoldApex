# Look-ahead correction (2026-10-13)

## The bug (real)

Several mean-reversion signals compared **same-bar Low/High** to **close-based levels** computed with that same bar’s Close:

- Bollinger bands (`bb_mr`, `bb_mr_rsi`, `triple_confirm_mr`, `quad_confirm_mr`)
- EMA pullback wick vs same-bar EMA

That is look-ahead: the band/EMA at the moment the wick prints is not the band/EMA finalized at the close.

## Fixes applied

1. Use **prior-bar** BB / EMA / Stoch levels for any wick comparison (`.shift(1)`).
2. Default fills: signal on bar `i` close → enter **`open[i+1]`** (not close[i]).
3. Stopped the old search process that used the buggy rules.

## Re-score after fix

| Strategy | Before (buggy) | After (causal) |
|----------|----------------|----------------|
| **Asian-Fade 1h (primary)** | 136 tr · 91.2% · PF 1.86 · +$93 | **136 tr · 91.9% · PF 1.88 · +$95** |
| BB+RSI 15m | 105 · 81.9% · PF 1.24 · +$47 | 79 · 79.7% · PF **1.04** · +$6 |
| Triple-confirm 15m | 90 · 93.3% · PF 2.17 · +$70 | 18 · 88.9% · PF 1.19 · +$4 |

## What this means

- **You were right** about look-ahead on BB-style winners — those results were inflated / not trustworthy.
- **Asian-Fade 1h was not using that bug** (range already `.shift(1)` on past highs/lows). It still holds under next-open fills.
- High winrate is still mostly from **TP ≪ SL** (~0.15R), not from magic alpha. Treat as fragile to spread/slippage.
- Published BB/triple “champions” in earlier messages should be considered **invalidated**.

## Still true caveats

- Yahoo GC=F ≠ broker XAUUSD
- Parameter search = data snooping risk even with causal bars
- Past ≠ future
