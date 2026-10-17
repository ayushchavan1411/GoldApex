#!/usr/bin/env python3
"""RR1 live bot placeholder — NO FULL QUALIFIER YET.

Gates:
  A) WR≥75% IS+OOS, RR1, tpd≥1, PF, year_stable
  B) WR≥60% IS+OOS, geoRR≥1.5 (TP>SL), tpd≥1, PF, year_stable

Current best durable (not packaged — tpd≪1):
  sl2|fvg_rsi+atr_expanded+skip_tiny_body|5m ~76.1/76.2% @ tpd~0.10
"""
from __future__ import annotations

def main():
    raise SystemExit(
        "No full qualifier yet (need WR≥75% RR1 or WR≥60% pos-RR, both with tpd≥1 + year_stable). "
        "See rr1_search_results/CHAMPION.json."
    )

if __name__ == "__main__":
    main()
