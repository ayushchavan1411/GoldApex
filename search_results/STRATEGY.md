# Best High-Winrate Strategies Found

Search status: still running in background (`continue_search.py`) for even better variants.  
Hard target **≥80% winrate + profitable IS/OOS** was met. Current champions exceed **90%**.

---

## Primary pick (most robust): Asian-Range Fade · 1h

| Metric | In-sample (70%) | Out-of-sample (30%) | Full (~2.4y) |
|--------|-----------------|---------------------|--------------|
| Winrate | **89.7%** | **94.9%** | **91.2%** |
| Profit factor | 1.62 | 2.76 | 1.86 |
| Trades | 97 | 39 | 136 |
| Net P&L | +$52.26 | +$40.30 | **+$92.55** |
| Final equity | — | — | **$592.55** (from $500) |

### Year stability
| Year | Trades | Winrate | P&L |
|------|--------|---------|-----|
| 2024 | 33 | 87.9% | +$24.72 |
| 2025 | 62 | 90.3% | +$24.24 |
| 2026 | 41 | 95.1% | +$43.60 |

### Rules
1. Timeframe: **1-hour** gold (GC=F / XAUUSD)
2. Session: **NY Overlap only** (13:00–17:00 UTC)
3. Build rolling range high/low over last **40 bars**
4. **LONG** if low tags range low AND RSI(7) < 25  
   **SHORT** if high tags range high AND RSI(7) > 70
5. SL = **2.0 × ATR(14)** · TP = **0.3 × ATR(14)** · max hold **12 bars**
6. Risk 2% equity capped at $20 · one position · halt under $425

---

## Secondary pick (highest WR on 15m): Triple Confirm MR · NY

| Metric | IS | OOS | Full (~60d) |
|--------|----|-----|-------------|
| Winrate | **90.8%** | **100%** (25/25) | **93.3%** |
| Profit factor | 1.59 | ∞ | 2.29 |
| Trades | 65 | 25 | 90 |
| Net P&L | +$35.42 | +$41.69 | **+$77.05** |

### Rules
1. Timeframe: **15-minute**
2. Session: **NY only** (17:00–22:00 UTC)
3. **LONG** when all true: Low ≤ BB(10,1.5) lower · RSI(3)<20 · Stoch(14)<20  
   **SHORT** when: High ≥ BB upper · RSI(3)>70 · Stoch(14)>70
4. SL **2.0×ATR** · TP **0.3×ATR** · max hold 24 · optional BE partial

---

## Earlier 80% baseline: BB+RSI · London/Overlap 15m
- IS 82.7% / OOS 80.0% / PF 1.26/1.21 — superseded by champions above.

---

## How winrate gets this high
All champions use **TP ≪ SL** (about 0.15R). Most trades scratch small wins at TP; occasional full SL is large. Expectancy stays positive because wins dominate.

## Caveats (read these)
- Yahoo `GC=F` proxy, not your broker’s XAUUSD; no spread/commission
- Selected via large search → some overfitting risk remains (1h multi-year result is the least fragile)
- High winrate ≠ risk-free; a cluster of SL hits can erase many small TPs
- Past backtests ≠ future performance

## Reproduce / keep searching
```bash
python3 winning_strategy_backtest.py   # original 80% BB+RSI
python3 continue_search.py             # never-stop hunter (already running in tmux)
```

Artifacts: `CHAMPION.json`, `BEST_STRATEGIES.json`, `champion_trades.csv`, `asian_fade_1h_trades.csv`, `winner_report.json`
