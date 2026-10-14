# Findings — 2y XAU/USD 5m, 1:1 RR hunt

**Data:** Twelve Data 5m — 185k bars (~771 days)  
**Target:** WR ≥75% IS+OOS, ≥1 tpd, PF gates, year-stable, ATR≥1.5

## Frontier (honest)

| Setup | min WR | tpd | n IS/OOS |
|--------|--------|-----|----------|
| Union RSI+MACD + BB-RSI fade | **74.1% / 73.7%** | 0.12 | 54/19 |
| Prior RSI+MACD / SMC | ~71% / ~78% | ~0.07 | ~31/18 |
| Best ≥1 tpd | ~54% | ~2+ | — |
| Full qualifier | **none** | — | — |

## Hunters
10 under watchdog: search, random, smc, hill, select, public, freq, union, orcombo, regime.

Hunt continues.


## Checkpoint 2026-10-14 08:43 UTC
- Hunters: 12/12 alive (orcombo crash fixed: empty singles sample).
- Stage A pipeline: ~190k singles tested, 0 full quals.
- Best still: union `rsi_macd+bb_rsi_fade` **74.1%/73.7% WR**, tpd **0.12/0.13** (n=54/19).
- No ≥75% WR + ≥1.0 tpd qualifier yet.
