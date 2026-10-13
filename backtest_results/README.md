# V26.2 Backtest Results

**Verdict: strategy lost −15.1% and hit the hard drawdown halt.**

| Metric | Value |
|--------|-------|
| Symbol / TF | GC=F (gold futures proxy) · 15m |
| Period | 2026-08-04 → 2026-10-13 (~60 days, Yahoo max for 15m) |
| Starting capital | $500.00 |
| Final equity | $424.48 |
| Net P&L | **−$75.52 (−15.1%)** |
| Trades | 29 (14 wins / 15 losses) |
| Win rate | 48.3% |
| Profit factor | 0.46 |
| Expectancy | −$2.60 / trade |
| Max drawdown | 15.1% (halted at $425) |
| TP2 hits | **0** |

## Setup breakdown

| Setup | N | Win % | P&L |
|-------|---|-------|-----|
| 1D_Fib_1.0 | 14 | 50% | −$31.57 |
| 1D_Fib_1.618 | 9 | 22% | −$57.22 |
| 1D_Fib_0.382 | 3 | 67% | −$0.13 |
| 1D_Fib_1.618 (Correction) | 2 | 100% | +$8.90 |
| Coiled_EMA_Breakout | 1 | 100% | +$4.52 |

## Key observations

1. **All 29 trades occurred on Aug 6–7**, then equity fell below the $425 hard-DD halt and no further entries were taken for the rest of the sample.
2. **Stops are structurally too tight.** With ~$10 risk and ~2.3 oz size at ~$4300 gold, SL distance is ~$4. Median 15m bar range was **$7.30**, so the next bar often spans both SL and TP1.
3. **Payoff asymmetry:** avg win +$4.54 (TP1 then BE) vs avg loss −$9.27. Without TP2 runners, break-even needs >67% win rate; actual was 48%.
4. **LONG-heavy and losing:** 27 longs (−$84.40) vs 2 shorts (+$8.90).
5. Exit mix: 15 Hard SL, 14 BE Stop, **0 TP2**.

## How to reproduce

```bash
pip install pandas numpy yfinance
python3 backtest_v26.py
```

Outputs: `trades.csv`, `summary.json`, `equity_curve.csv` in this folder.

## Assumptions

- Yahoo Finance `GC=F` as XAUUSD proxy (spot gold from an MT5 broker may differ)
- Entry at signal-bar close (live bot uses live bid/ask)
- If SL and TP1 both touch the same bar before TP1 is taken, SL wins (conservative)
- One position at a time; no spreads/commissions
