#!/usr/bin/env python3
"""AUTO-GENERATED high-frequency WR bot from hf_wr_search champion.
Causal: prior-bar levels, next-open fills.
Champion score=6730.5 IS_WR=83.3% OOS_WR=95.2%
tpd=2.07/1.27
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

PARAMS = {
  "kind": "rsi_level",
  "rsi": 7,
  "os": 20,
  "ob": 60,
  "sessions": [
    "overlap"
  ]
}
EXIT = {
  "sl": 2.0,
  "tp": 0.4,
  "max_hold": 6,
  "allow_long": true,
  "allow_short": true
}
TF = "15m"
KIND = "rsi_level"
START_CAP, RISK_PCT, MAX_RISK, HALT = 500.0, 0.02, 20.0, 425.0
OUT = Path("hf_search_results")

# Reuse searcher helpers
from hf_wr_search import add_indicators, gen_signals, simulate, load_frames

def main():
    frames = load_frames()
    df = add_indicators(frames[TF])
    params = dict(PARAMS)
    if params.get("sessions") is not None:
        params["sessions"] = tuple(params["sessions"])
    if params.get("hours") is not None:
        params["hours"] = tuple(params["hours"])
    long_e, short_e = gen_signals(df, KIND, params)
    full = simulate(df, long_e, short_e, EXIT["sl"], EXIT["tp"], EXIT["max_hold"], 250, len(df)-2,
                    EXIT.get("allow_long", True), EXIT.get("allow_short", True))
    split = int(len(df)*0.7)
    is_m = simulate(df, long_e, short_e, EXIT["sl"], EXIT["tp"], EXIT["max_hold"], 250, split,
                    EXIT.get("allow_long", True), EXIT.get("allow_short", True))
    oos_m = simulate(df, long_e, short_e, EXIT["sl"], EXIT["tp"], EXIT["max_hold"], split, len(df)-2,
                    EXIT.get("allow_long", True), EXIT.get("allow_short", True))
    report = {"kind": KIND, "timeframe": TF, "params": PARAMS, "exit": EXIT,
              "full": full, "is": is_m, "oos": oos_m}
    OUT.mkdir(exist_ok=True)
    (OUT / "bot_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
