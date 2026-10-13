# Findings — 2y XAU/USD 5m, 1:1 RR hunt

**Data:** Twelve Data free `XAU/USD` 5m — **185,000 bars**, **~771 days** (2024-09-03 → 2026-10-13)  
**Rules:** SL = TP (1:1), next-open fills, ATR ≥ $1.5, year-stability gate  
**Target:** **WR ≥ 75%** IS+OOS, **≥ 1 trade/day**, PF gates, year-stable

## Current frontier (honest)

| Setup | min WR (IS/OOS) | tpd | Notes |
|--------|------------------|-----|--------|
| RSI+MACD Asian (best peak) | **71.0% / 70.6%** | **0.07 / 0.11** | Best WR; far below 1 tpd |
| Best ≥1 tpd combo | **~54% / ~53%** | **~2.2** | High frequency, no edge vs coin-flip+cost |
| Full qualifier (≥75% + ≥1 tpd) | — | — | **Not found** after ~2.5M+ configs |

## What failed / invalidated

| Setup | Result |
|--------|--------|
| Yahoo ~71d “78%” RSI champion on 2y | **~47% WR** — look-ahead / short-window artifact |
| Fake “90%+” bb_mid_reject | Invalid without ATR≥1.5 (flat quote bars) |
| Random / single indicator 1:1 baselines | typically **~45–52% WR** |

## Hunters (watchdog-supervised)

`rr1_search`, `rr1_random_opt`, `rr1_smc_search`, `rr1_hillclimb`, `rr1_select_search`, `rr1_public_search`, `rr1_freqboost`, **`rr1_union_search`** (multi-OR greedy + session partition), `rr1_orcombo`

`save_champion` now only overwrites when min(WR) / tpd improves (stops lower-WR noise from clobbering the 71% peak).

## Methods

ICT/SMC (FVG, sweeps, OB, BOS, fib), public playbooks (Ichimoku, HA, Keltner, Supertrend, Turtle, ORB), RSI+MACD, AND/OR/union stacking, session/hour partitions.

Hunt continues until manually stopped.
