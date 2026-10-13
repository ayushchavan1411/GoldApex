# Invalid candidates note

Entries in `candidates.jsonl` claiming ≥75% WR at ≥1 tpd (esp. `bb_mid_reject` with `rr` 0.4–0.6 × ATR_7)
were produced by stale hunter processes and/or sub-1.0×ATR stops on flat Asian bars.

Revalidation with current engine (MIN_ATR≥1.5, dist≥MIN_ATR, year_stable) collapses these to ~50–60% WR.
Do not treat those candidate lines as qualifiers.
