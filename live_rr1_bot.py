#!/usr/bin/env python3
"""RR1 live bot placeholder — NO FULL QUALIFIER YET.

Gates: WR≥75% IS+OOS, tpd≥1, PF gates, year_stable, causal next-open, MIN_ATR≥1.5.

Current best durable (not packaged — tpd≪1):
  sl2|fvg_rsi+atr_expanded+skip_tiny_body|5m
  IS/OOS WR ≈ 76.1%/76.2%  tpd ≈ 0.10/0.14  PF ≈ 3.17/2.79

Invalid OF overwrite (parked CHAMPION_OF_INVALID.json) must not ship here.
Benny reel ORB (DdCYACaP-ef) does not qualify under RR1 on XAU.
"""
from __future__ import annotations


def main():
    raise SystemExit(
        "No full RR1 qualifier yet (need WR≥75% IS+OOS + tpd≥1 + PF + year_stable). "
        "See rr1_search_results/CHAMPION.json for best durable near-miss."
    )


if __name__ == "__main__":
    main()
