# Findings — 2y XAU/USD 5m, 1:1 RR hunt

**Data:** Twelve Data free `XAU/USD` 5m — **185,000 bars**, **~771 days** (2024-09-03 → 2026-10-13)  
**Rules:** SL = TP (1:1), next-open fills, ATR ≥ $1.5, year-stability gate  
**New target:** **WR ≥ 75%** IS+OOS, **≥ 1 trade/day**, profitable

## What we already know

| Setup | Window | Result |
|--------|--------|--------|
| RSI-level “78%” champion (Yahoo ~71d) | 2y Twelve Data | **~47% WR**, $500→~$411 — **failed** |
| Random 1:1 baselines (RSI/BB/FVG/fib/SMC smoke) | 2y | typically **~45–52% WR** (honest for 1:1) |
| Fake “90%+” bb_mid_reject | 2y without ATR floor | **invalid** — harvested flat ATR&lt;$0.5 quote bars |

## Methods now searching (24h never-stop)

1. Classic: RSI, BB, Stoch, EMA, MACD, pins, engulf  
2. **ICT/SMC:** FVG reclaim, FVG+RSI, liquidity sweep, sweep+FVG, order-block retest, BOS pullback, SMC confluence  
3. **Fib:** 38.2 / 50 / 61.8 / 78.6 bounce + premium/discount filter  
4. Combos: AND/OR, session/hour filters, mutation/hillclimb  

Processes: `rr1-search`, `rr1-random`, `rr1-hill`, `rr1-smc`

## Goal status

No honest **≥75% WR + ≥1 tpd** qualifier on full 2y yet. Hunt continues until manually stopped / 24h+.
