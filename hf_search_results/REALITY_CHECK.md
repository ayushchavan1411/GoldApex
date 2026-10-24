# Reality check: 98.8% WR is not a 2-year edge

## Code audit
- **No BB wick look-ahead** in Stoch/RSI bots (those were fixed earlier).
- **Fills are causal**: signal at bar close → enter next open.
- Stoch(K) uses same-bar OHLC; that is normal at close (not the prior BB bug).
- **No bug found that fabricates wins.**

## Why 98.8% looks fake anyway
With **TP = 0.2×ATR** and **SL = 2.0×ATR**, barrier math alone says:

> P(hit TP first) ≈ SL / (SL + TP) ≈ **90.9%**

even if direction is a coin flip. So a short 15m sample printing **98.8%** is mostly **asymmetric exits + luck**, not proven alpha.

## Results

### Stoch bot
| Sample | Trades | WR | PF | P&L |
|--------|--------|----|----|-----|
| 15m ~70 days (Yahoo max) | 82 | **98.8%** | 8.03 | +$76 |
| **1h ~2.4 years** | 251 | **89.2%** | **0.84** | **−$39** |

Yearly 1h: 2024 −$37 · 2025 +$3 · 2026 −$5. Several months &lt;80% WR.

### RSI bot (the “3 trades/day” one)
| Sample | Trades | WR | PF | P&L |
|--------|--------|----|----|-----|
| 15m ~70 days | 180 | **94.4%** | 1.83 | +$83 |
| **1h ~2.4 years** | more trades w/o halt | ~**82–89%** range in runs | **&lt;1** | **losing** |

## Data limit
Yahoo Finance does **not** provide 2 years of 15m bars (only ~60–70d).  
So a literal 2y **15m** backtest needs MT5/broker history export.

## Bottom line
- Code path is basically fine.
- **98.8% does not survive longer/higher-TF stress and is not “2 years of proof.”**
- High WR here is largely the **0.2R TP / 2R SL** structure.
- Do **not** trust these as production edges until MT5 XAUUSD multi-year 15m validation.
