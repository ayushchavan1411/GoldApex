# 1:1 RR search status

**Running:** tmux `rr1-search` → `python3 -u rr1_search.py`  
**Constraint:** SL distance = TP distance (true 1:1). Causal next-open fills.

## Why this is harder
Previous “94–98% WR” bots used **TP ≪ SL** (~0.15R), which forces high WR by math.  
With **1:1**, random barrier expectation is ~50% WR. Hitting **≥80%** needs real edge.

## Live progress (updates in `progress.json`)
- Search never stops until you kill it.
- `qualified_count` = strategies that pass WR≥80% IS+OOS, tpd≥1, PF gates.
- Best current champion is always in `CHAMPION.json` / `live_rr1_bot.py` when improved.

## Reproduce
```bash
python3 -u rr1_search.py
tail -f rr1_search_results/search.log
```
