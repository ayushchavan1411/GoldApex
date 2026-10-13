#!/usr/bin/env python3
"""AUTO-GENERATED 1:1 RR bot from rr1_search.
IS WR=78.6% OOS WR=68.2% tpd=1.09/1.33
CAUSAL next-open · SL=TP=1.2×ATR
"""
from __future__ import annotations
import json
from pathlib import Path
from rr1_search import load_frames, add_indicators, gen_signals, simulate

KIND = "rsi_level"
TF = "5m"
PARAMS = {
  "rsi": 7,
  "os": 20,
  "ob": 90,
  "sessions": [
    "overlap",
    "ny"
  ],
  "band": 5
}
EXIT = {
  "rr": 1.2,
  "max_hold": 2,
  "atr": "ATR_14",
  "allow_long": true,
  "allow_short": true
}

def main():
    df = add_indicators(load_frames()[TF])
    p = dict(PARAMS)
    if p.get("sessions") is not None:
        p["sessions"] = tuple(p["sessions"])
    if p.get("hours") is not None:
        p["hours"] = tuple(p["hours"])
    if p.get("dows") is not None:
        p["dows"] = tuple(p["dows"])
    L, S = gen_signals(df, KIND, p)
    n = len(df); split = int(n * 0.7)
    common = dict(long_e=L, short_e=S, rr_atr=EXIT["rr"], max_hold=EXIT["max_hold"],
                  atr_col=EXIT.get("atr","ATR_14"),
                  allow_long=EXIT.get("allow_long", True), allow_short=EXIT.get("allow_short", True))
    full = simulate(df, start_i=250, end_i=n-2, **common)
    is_m = simulate(df, start_i=250, end_i=split, **common)
    oos_m = simulate(df, start_i=split, end_i=n-2, **common)
    report = {"kind": KIND, "tf": TF, "params": PARAMS, "exit": EXIT, "rr": "1:1",
              "full": full, "is": is_m, "oos": oos_m}
    Path("rr1_search_results").mkdir(exist_ok=True)
    Path("rr1_search_results/bot_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
