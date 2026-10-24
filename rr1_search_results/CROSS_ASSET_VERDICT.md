# Cross-asset goal verdict

**Question:** Can any non-gold market hit ≥75% WR IS+OOS, RR≥1.0, ≥1 trade/day?

**Answer: No — not on the free-plan proxies we can actually download.**

## Data coverage (~2y 5m)

| Bucket | Symbols tested | Notes |
|---|---|---|
| FX majors | EUR, GBP, JPY, AUD, CAD, CHF, NZD vs USD | Full ~175k bars each |
| US indices | SPY, QQQ, DIA, IWM | Cash SPX/NDX plan-locked |
| Oil | **USO** (oil ETF) | USOIL / WTI cash plan-locked |
| Metals | **GLD, SLV** | XAG/XPT/XPD cash plan-locked |

## Screens run

1. **Seed screen** — gold champion + ICT/SMC/RSI/MACD seeds, adaptive `MIN_ATR`
2. **OR-boost** — OR-union of high-WR low-tpd seeds (JPY/GBP/USO/SLV/GLD done; EUR+)
3. **HF ceiling** — maximize WR among setups with **tpd≥1** (EUR in progress)

## Results (no full qualifier anywhere)

| Symbol | Best high-WR path | WR IS/OOS | tpd IS/OOS | Gap |
|---|---|---|---|---|
| **USD/JPY** | fvg_rsi + ATR/session filters / OR | **~75–82% / 75–82%** | **~0.07–0.14** | WR OK, **tpd ≪ 1** |
| GBP/USD | fvg_rsi family | ~75% / ~82% | ~0.05–0.16 | same wall |
| USO | fvg_rsi dual | ~72% / ~73% | ~0.09–0.16 | below both gates |
| SLV / GLD | fvg_rsi / smc | ~68–70% | ~0.06–0.09 | below both |
| EUR/USD | HF ceiling (64k exits) | at tpd≥1: **58.2% / 69.0% WR** | 1.58 / 2.38 | **WR collapses when tpd rises** |
| EUR/USD | OR-boost high-WR seeds | ~66.7% / 68.2% | ~0.11 / 0.15 | WR below 75, tpd ≪ 1 |
| Index ETFs | smc / dual | ~55–62% | ~0.1 | far |

## Pattern (same as gold)

- Setups that clear ~75% WR are rare (~0.05–0.15 trades/day).
- Setups that clear ≥1 tpd sit near coin-flip / mid-50s WR under strict 1:1 ATR stops.
- OR-unions lift sample size a little but do **not** break the frequency wall while keeping 75% WR.

## Conclusion

Under the same gates as gold (**≥75% WR IS+OOS, RR≥1.0, ≥1 tpd**), **none of FX / USO / GLD / SLV / US index ETFs qualify** in these screens. Closest market is **USD/JPY** (high WR, tiny tpd) — same failure mode as XAUUSD.
