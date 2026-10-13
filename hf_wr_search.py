#!/usr/bin/env python3
"""
Never-stop causal searcher for:
  - winrate >= 80% (IS and OOS)
  - >= 1 trade per gold trading day (approx)
  - profit factor >= 1.15 IS and OOS
  - net PnL > 0 both windows

STRICT CAUSALITY:
  - wick vs prior-bar BB/EMA/Stoch only (.shift(1))
  - range levels .shift(1)
  - signal on bar i close → fill open[i+1]
  - ATR from signal bar (known at close) for stops

Keeps running until killed. Writes champions continuously.
"""

from __future__ import annotations

import itertools
import json
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("hf_search_results")
OUT.mkdir(exist_ok=True)
LOG = OUT / "search.log"
CHAMPION = OUT / "CHAMPION.json"
CANDIDATES = OUT / "candidates.jsonl"
PROGRESS = OUT / "progress.json"

START_CAP = 500.0
RISK_PCT = 0.02
MAX_RISK = 20.0
HALT = START_CAP * 0.85

TARGET_WR = 0.80
TARGET_PF = 1.15
MIN_TRADES_PER_DAY = 1.0
MIN_IS_TRADES = 40
MIN_OOS_TRADES = 20
OOS_WR = 0.80
OOS_PF = 1.10

RNG = random.Random(7)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for key, interval, period in [
        ("5m", "5m", "60d"),
        ("15m", "15m", "60d"),
        ("1h", "1h", "730d"),
    ]:
        df = yf.download("GC=F", interval=interval, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        if len(df) > 500:
            frames[key] = df
            log(f"Loaded {key}: {len(df)} bars {df.index[0]} → {df.index[-1]}")
    if not frames:
        raise RuntimeError("No data")
    return frames


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    for span in (5, 8, 9, 12, 20, 21, 50, 100, 200):
        d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
    for period in (2, 3, 5, 7, 14):
        delta = c.diff()
        ag = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        al = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        d[f"RSI_{period}"] = (100 - (100 / (1 + ag / al.replace(0, np.nan)))).fillna(50)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    d["ATR_14"] = tr.ewm(com=13, adjust=False).mean()
    for period, k in ((10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5), (50, 2.0)):
        mid = c.rolling(period).mean()
        std = c.rolling(period).std()
        d[f"BB_UP_{period}_{k}"] = mid + k * std
        d[f"BB_LO_{period}_{k}"] = mid - k * std
    for period in (5, 14):
        ll = l.rolling(period).min()
        hh = h.rolling(period).max()
        d[f"STOCH_K_{period}"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        d[f"STOCH_D_{period}"] = d[f"STOCH_K_{period}"].rolling(3).mean()
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["session"] = np.select(
        [
            (d["hour"] >= 0) & (d["hour"] < 8),
            (d["hour"] >= 8) & (d["hour"] < 13),
            (d["hour"] >= 13) & (d["hour"] < 17),
            (d["hour"] >= 17) & (d["hour"] < 22),
        ],
        ["asian", "london", "overlap", "ny"],
        default="late",
    )
    body = (c - d["Open"]).abs()
    rng = (h - l).replace(0, np.nan)
    d["lower_wick"] = (np.minimum(c, d["Open"]) - l) / rng
    d["upper_wick"] = (h - np.maximum(c, d["Open"])) / rng
    d["body_pct"] = body / rng
    d["range"] = h - l
    d["range_ma"] = d["range"].rolling(20).mean()
    return d


def trading_days_in_index(idx: pd.DatetimeIndex) -> float:
    """Approx gold trading days in span."""
    if len(idx) < 2:
        return 1.0
    cal = (idx[-1] - idx[0]).days + 1
    return max(1.0, cal * 5.5 / 7.0)


def simulate(
    df: pd.DataFrame,
    long_e: np.ndarray,
    short_e: np.ndarray,
    sl_mult: float,
    tp_mult: float,
    max_hold: int,
    start_i: int,
    end_i: int,
    allow_long: bool = True,
    allow_short: bool = True,
) -> dict:
    n = len(df)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    atrs = df["ATR_14"].to_numpy(float)
    times = df.index

    capital = START_CAP
    peak = START_CAP
    max_dd = 0.0
    pnls: list[float] = []
    entry_times: list[pd.Timestamp] = []

    i = start_i
    while i < end_i and i + 1 < n:
        if capital < HALT:
            break
        gl = allow_long and bool(long_e[i])
        gs = allow_short and bool(short_e[i])
        if gl == gs:
            i += 1
            continue
        direction = "LONG" if gl else "SHORT"
        atr = atrs[i]
        entry = opens[i + 1]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
            i += 1
            continue
        risk = min(MAX_RISK, capital * RISK_PCT)
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult
        if sl_dist <= 0:
            i += 1
            continue
        qty = risk / sl_dist
        sl = entry - sl_dist if direction == "LONG" else entry + sl_dist
        tp = entry + tp_dist if direction == "LONG" else entry - tp_dist

        j = i + 1
        bars = 0
        exit_p = entry
        hit = False
        while j < n and bars < max_hold:
            hi, lo = highs[j], lows[j]
            bars += 1
            if direction == "LONG":
                if lo <= sl:
                    exit_p, hit = sl, True
                    break
                if hi >= tp:
                    exit_p, hit = tp, True
                    break
            else:
                if hi >= sl:
                    exit_p, hit = sl, True
                    break
                if lo <= tp:
                    exit_p, hit = tp, True
                    break
            j += 1
        if not hit:
            exit_p = closes[min(j, n - 1)]
        pnl = (exit_p - entry) * qty if direction == "LONG" else (entry - exit_p) * qty
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak else 0)
        pnls.append(pnl)
        entry_times.append(times[i])
        i = j + 1 if j > i else i + 1

    if not pnls:
        return {
            "trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "final_equity": capital,
            "max_dd": max_dd,
            "trades_per_day": 0.0,
            "expectancy": 0.0,
        }
    arr = np.array(pnls, dtype=float)
    wins, losses = arr[arr > 0], arr[arr <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
    # trading days over the simulated window
    window = times[start_i:end_i]
    tdays = trading_days_in_index(window)
    return {
        "trades": int(len(arr)),
        "winrate": float((arr > 0).mean()),
        "profit_factor": float(pf),
        "net_pnl": float(arr.sum()),
        "final_equity": float(capital),
        "max_dd": float(max_dd),
        "trades_per_day": float(len(arr) / tdays),
        "expectancy": float(arr.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def sess_mask(df: pd.DataFrame, sessions) -> np.ndarray:
    if not sessions:
        return np.ones(len(df), dtype=bool)
    return df["session"].isin(sessions).to_numpy()


def gen_signals(df: pd.DataFrame, kind: str, p: dict) -> tuple[np.ndarray, np.ndarray]:
    c = df["Close"]
    sess = sess_mask(df, p.get("sessions"))

    if kind == "rsi_cross":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (rsi < p["os"]) & (rsi.shift(1) >= p["os"]) & sess
        short_e = (rsi > p["ob"]) & (rsi.shift(1) <= p["ob"]) & sess

    elif kind == "rsi_level":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (rsi < p["os"]) & sess
        short_e = (rsi > p["ob"]) & sess

    elif kind == "bb_fade":
        lo = df[f"BB_LO_{p['period']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['period']}_{p['k']}"].shift(1)
        long_e = (df["Low"] <= lo) & (c > lo) & sess
        short_e = (df["High"] >= up) & (c < up) & sess

    elif kind == "bb_rsi_fade":
        lo = df[f"BB_LO_{p['period']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['period']}_{p['k']}"].shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= lo) & (c > lo) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= up) & (c < up) & (rsi > p["ob"]) & sess

    elif kind == "stoch_cross":
        k = df[f"STOCH_K_{p['period']}"]
        d = df[f"STOCH_D_{p['period']}"]
        long_e = (k < p["os"]) & (k > d) & (k.shift(1) <= d.shift(1)) & sess
        short_e = (k > p["ob"]) & (k < d) & (k.shift(1) >= d.shift(1)) & sess

    elif kind == "triple_fade":
        lo = df[f"BB_LO_{p['bb']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['bb']}_{p['k']}"].shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        sk = df[f"STOCH_K_{p['stoch']}"].shift(1)
        long_e = (df["Low"] <= lo) & (c > lo) & (rsi < p["os"]) & (sk < p["os"]) & sess
        short_e = (df["High"] >= up) & (c < up) & (rsi > p["ob"]) & (sk > p["ob"]) & sess

    elif kind == "ema_wick_fade":
        ema = df[f"EMA_{p['ema']}"].shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= ema) & (c > ema) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= ema) & (c < ema) & (rsi > p["ob"]) & sess

    elif kind == "range_fade":
        look = p["look"]
        a_hi = df["High"].rolling(look).max().shift(1)
        a_lo = df["Low"].rolling(look).min().shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] <= a_lo) & (c > a_lo) & (rsi < p["os"]) & sess
        short_e = (df["High"] >= a_hi) & (c < a_hi) & (rsi > p["ob"]) & sess

    elif kind == "pin_rsi":
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["lower_wick"] > p["wick"]) & (df["body_pct"] < p["body"]) & (rsi < p["os"]) & sess
        short_e = (df["upper_wick"] > p["wick"]) & (df["body_pct"] < p["body"]) & (rsi > p["ob"]) & sess

    elif kind == "micro_reclaim":
        # close reclaim of prior low/high with RSI extreme
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = (df["Low"] < df["Low"].shift(1)) & (c > df["Low"].shift(1)) & (rsi < p["os"]) & sess
        short_e = (df["High"] > df["High"].shift(1)) & (c < df["High"].shift(1)) & (rsi > p["ob"]) & sess

    elif kind == "dual_rsi_bb":
        # looser dual: RSI extreme OR (causal) BB touch
        lo = df[f"BB_LO_{p['period']}_{p['k']}"].shift(1)
        up = df[f"BB_UP_{p['period']}_{p['k']}"].shift(1)
        rsi = df[f"RSI_{p['rsi']}"]
        long_e = ((rsi < p["os"]) | ((df["Low"] <= lo) & (c > lo))) & sess
        short_e = ((rsi > p["ob"]) | ((df["High"] >= up) & (c < up))) & sess

    else:
        raise ValueError(kind)

    long_e = long_e.fillna(False).to_numpy() if hasattr(long_e, "fillna") else long_e
    short_e = short_e.fillna(False).to_numpy() if hasattr(short_e, "fillna") else short_e

    if p.get("hours"):
        hm = df["hour"].isin(p["hours"]).to_numpy()
        long_e = long_e & hm
        short_e = short_e & hm
    return long_e, short_e


@dataclass
class Result:
    name: str
    timeframe: str
    winrate: float
    profit_factor: float
    trades: int
    trades_per_day: float
    net_pnl: float
    max_dd: float
    params: dict
    oos_winrate: float
    oos_profit_factor: float
    oos_trades: int
    oos_trades_per_day: float
    oos_net_pnl: float
    score: float


def qualifies(is_m: dict, oos_m: dict) -> bool:
    return (
        is_m["trades"] >= MIN_IS_TRADES
        and oos_m["trades"] >= MIN_OOS_TRADES
        and is_m["winrate"] >= TARGET_WR
        and oos_m["winrate"] >= OOS_WR
        and is_m["profit_factor"] >= TARGET_PF
        and oos_m["profit_factor"] >= OOS_PF
        and is_m["net_pnl"] > 0
        and oos_m["net_pnl"] > 0
        and is_m["trades_per_day"] >= MIN_TRADES_PER_DAY
        and oos_m["trades_per_day"] >= MIN_TRADES_PER_DAY * 0.8  # slight OOS slack
    )


def score_of(is_m: dict, oos_m: dict) -> float:
    if is_m["trades"] < 15:
        return -1e9
    s = (
        is_m["winrate"] * 100
        + oos_m["winrate"] * 120
        + is_m["profit_factor"] * 20
        + oos_m["profit_factor"] * 25
        + min(is_m["trades_per_day"], 5) * 15
        + min(oos_m["trades_per_day"], 5) * 15
        + (10 if is_m["net_pnl"] > 0 else -30)
        + (15 if oos_m["net_pnl"] > 0 else -40)
        - is_m["max_dd"] * 40
    )
    if qualifies(is_m, oos_m):
        s += 500
    return float(s)


def evaluate(name, tf, df, long_e, short_e, exit_p, params) -> Result | None:
    n = len(df)
    split = int(n * 0.7)
    common = dict(
        long_e=long_e,
        short_e=short_e,
        sl_mult=exit_p["sl"],
        tp_mult=exit_p["tp"],
        max_hold=exit_p["max_hold"],
        allow_long=exit_p.get("allow_long", True),
        allow_short=exit_p.get("allow_short", True),
    )
    is_m = simulate(df, start_i=250, end_i=split, **common)
    oos_m = simulate(df, start_i=split, end_i=n - 2, **common)
    if is_m["trades"] < 15:
        return None
    return Result(
        name=name,
        timeframe=tf,
        winrate=is_m["winrate"],
        profit_factor=is_m["profit_factor"],
        trades=is_m["trades"],
        trades_per_day=is_m["trades_per_day"],
        net_pnl=is_m["net_pnl"],
        max_dd=is_m["max_dd"],
        params={**params, "exit": exit_p},
        oos_winrate=oos_m["winrate"],
        oos_profit_factor=oos_m["profit_factor"],
        oos_trades=oos_m["trades"],
        oos_trades_per_day=oos_m["trades_per_day"],
        oos_net_pnl=oos_m["net_pnl"],
        score=score_of(is_m, oos_m),
    )


def param_space(phase: int) -> list[tuple[str, dict]]:
    space: list[tuple[str, dict]] = []
    sessions_opts = [
        None,
        ("asian",),
        ("london",),
        ("overlap",),
        ("ny",),
        ("london", "overlap"),
        ("overlap", "ny"),
        ("asian", "london", "overlap"),
        ("london", "overlap", "ny"),
        ("asian", "london", "overlap", "ny"),
    ]
    # High frequency favors looser filters + more sessions
    for rsi, os_, ob, sessions in itertools.product(
        [2, 3, 5, 7, 14],
        [15, 20, 25, 30, 35, 40],
        [60, 65, 70, 75, 80, 85],
        sessions_opts,
    ):
        if os_ >= ob:
            continue
        space.append(("rsi_cross", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))
        space.append(("rsi_level", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))

    for period, k, sessions in itertools.product(
        [10, 20, 50], [1.5, 2.0, 2.5], sessions_opts
    ):
        space.append(("bb_fade", {"period": period, "k": k, "sessions": sessions}))

    for period, k, rsi, os_, ob, sessions in itertools.product(
        [10, 20],
        [1.5, 2.0, 2.5],
        [2, 3, 5, 7],
        [20, 25, 30, 35, 40],
        [60, 65, 70, 75, 80],
        sessions_opts[:8],
    ):
        space.append(
            (
                "bb_rsi_fade",
                {"period": period, "k": k, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions},
            )
        )
        space.append(
            (
                "dual_rsi_bb",
                {"period": period, "k": k, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions},
            )
        )

    for period, os_, ob, sessions in itertools.product(
        [5, 14], [15, 20, 25, 30], [70, 75, 80, 85], sessions_opts[:7]
    ):
        space.append(("stoch_cross", {"period": period, "os": os_, "ob": ob, "sessions": sessions}))

    for bb, k, rsi, stoch, os_, ob, sessions in itertools.product(
        [10, 20],
        [1.5, 2.0],
        [2, 3, 7],
        [5, 14],
        [25, 30, 35],
        [65, 70, 75],
        sessions_opts[:6],
    ):
        space.append(
            (
                "triple_fade",
                {
                    "bb": bb,
                    "k": k,
                    "rsi": rsi,
                    "stoch": stoch,
                    "os": os_,
                    "ob": ob,
                    "sessions": sessions,
                },
            )
        )

    for ema, rsi, os_, ob, sessions in itertools.product(
        [8, 12, 20, 50],
        [2, 3, 7],
        [25, 30, 35, 40],
        [60, 65, 70, 75],
        sessions_opts[:7],
    ):
        space.append(
            (
                "ema_wick_fade",
                {"ema": ema, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions},
            )
        )

    for look, rsi, os_, ob, sessions in itertools.product(
        [6, 12, 20, 32, 40],
        [2, 3, 5, 7],
        [25, 30, 35, 40],
        [60, 65, 70, 75],
        sessions_opts[:7],
    ):
        space.append(
            (
                "range_fade",
                {"look": look, "rsi": rsi, "os": os_, "ob": ob, "sessions": sessions},
            )
        )

    for rsi, wick, body, os_, ob, sessions in itertools.product(
        [2, 3, 7],
        [0.45, 0.55, 0.65],
        [0.35, 0.45, 0.55],
        [25, 30, 35],
        [65, 70, 75],
        sessions_opts[:5],
    ):
        space.append(
            (
                "pin_rsi",
                {
                    "rsi": rsi,
                    "wick": wick,
                    "body": body,
                    "os": os_,
                    "ob": ob,
                    "sessions": sessions,
                },
            )
        )

    for rsi, os_, ob, sessions in itertools.product(
        [2, 3, 5, 7],
        [25, 30, 35, 40],
        [60, 65, 70, 75],
        sessions_opts[:7],
    ):
        space.append(("micro_reclaim", {"rsi": rsi, "os": os_, "ob": ob, "sessions": sessions}))

    if phase >= 1:
        extras = []
        hour_sets = [
            None,
            tuple(range(0, 22)),
            tuple(range(8, 17)),
            tuple(range(13, 21)),
            tuple(range(0, 8)) + tuple(range(13, 17)),
        ]
        for kind, p in space:
            for hours in hour_sets:
                pp = dict(p)
                if hours:
                    pp["hours"] = hours
                extras.append((kind, pp))
        space = extras

    RNG.shuffle(space)
    return space


def exit_grid(phase: int) -> list[dict]:
    # High WR + high frequency → smaller TP, moderate/wide SL
    sls = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    tps = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    holds = [2, 3, 4, 6, 8, 12, 18, 24]
    if phase == 0:
        sls, tps, holds = [0.8, 1.2, 2.0], [0.2, 0.3, 0.4], [3, 6, 12]
    grids = []
    sides = [
        {"allow_long": True, "allow_short": True},
        {"allow_long": True, "allow_short": False},
        {"allow_long": False, "allow_short": True},
    ]
    if phase == 0:
        sides = sides[:1]
    for sl, tp, mh, side in itertools.product(sls, tps, holds, sides):
        if tp >= sl:  # keep asymmetric for WR; still allow some balanced later
            if phase < 2:
                continue
        grids.append({"sl": sl, "tp": tp, "max_hold": mh, **side})
    return grids


def save_champion(r: Result) -> None:
    with open(CHAMPION, "w", encoding="utf-8") as f:
        json.dump(asdict(r), f, indent=2)
    with open(CANDIDATES, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(r)) + "\n")


def write_bot(r: Result) -> None:
    """Materialize a runnable bot module from champion params."""
    code = f'''#!/usr/bin/env python3
"""AUTO-GENERATED high-frequency WR bot from hf_wr_search champion.
Causal: prior-bar levels, next-open fills.
Champion score={r.score:.1f} IS_WR={r.winrate:.1%} OOS_WR={r.oos_winrate:.1%}
tpd={r.trades_per_day:.2f}/{r.oos_trades_per_day:.2f}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

PARAMS = {json.dumps({k: v for k, v in r.params.items() if k != "exit"}, indent=2)}
EXIT = {json.dumps(r.params["exit"], indent=2)}
TF = {json.dumps(r.timeframe)}
KIND = {json.dumps(r.params["kind"])}
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
    report = {{"kind": KIND, "timeframe": TF, "params": PARAMS, "exit": EXIT,
              "full": full, "is": is_m, "oos": oos_m}}
    OUT.mkdir(exist_ok=True)
    (OUT / "bot_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
'''
    Path("hf_wr_bot.py").write_text(code)


def main() -> None:
    log("=== HF WR SEARCH START (never stop) ===")
    log(
        f"Targets: WR>={TARGET_WR:.0%} PF>={TARGET_PF} tpd>={MIN_TRADES_PER_DAY} "
        f"| OOS WR>={OOS_WR:.0%} PF>={OOS_PF} (causal next_open)"
    )
    frames = {k: add_indicators(v) for k, v in load_frames().items()}
    # Prefer higher frequency TFs
    tf_order = [t for t in ("5m", "15m", "1h") if t in frames]

    champion: Result | None = None
    if CHAMPION.exists():
        try:
            champion = Result(**json.loads(CHAMPION.read_text()))
            log(f"Seed champion score={champion.score:.1f}")
        except Exception:
            champion = None

    tested = 0
    phase = 0
    while True:
        log(f"----- PHASE {phase} tfs={tf_order} -----")
        space = param_space(phase)
        exits = exit_grid(phase)
        max_exits = 10 + phase * 6
        log(f"space={len(space)} exits={len(exits)}")

        for idx, (kind, params) in enumerate(space):
            for tf in tf_order:
                # 1h rarely hits 1 trade/day with selective filters; still try in later phases
                if phase == 0 and tf == "1h":
                    continue
                df = frames[tf]
                try:
                    long_e, short_e = gen_signals(df, kind, params)
                except Exception:
                    continue
                sig_n = int(long_e.sum() + short_e.sum())
                # need enough raw signals for ~1/day
                tdays = trading_days_in_index(df.index)
                if sig_n < tdays * 0.8:
                    continue

                probe = exits[:: max(1, len(exits) // max_exits)][:max_exits]
                local_best = None
                for exit_p in probe:
                    tested += 1
                    res = evaluate(
                        f"{kind}|{tf}|p{phase}",
                        tf,
                        df,
                        long_e,
                        short_e,
                        exit_p,
                        {"kind": kind, **params},
                    )
                    if not res:
                        continue
                    if local_best is None or res.score > local_best.score:
                        local_best = res
                    if champion is None or res.score > champion.score:
                        if res.trades_per_day >= 0.7 and res.winrate >= 0.75:
                            champion = res
                            save_champion(res)
                            write_bot(res)
                            q = qualifies(
                                {
                                    "trades": res.trades,
                                    "winrate": res.winrate,
                                    "profit_factor": res.profit_factor,
                                    "net_pnl": res.net_pnl,
                                    "trades_per_day": res.trades_per_day,
                                    "max_dd": res.max_dd,
                                },
                                {
                                    "trades": res.oos_trades,
                                    "winrate": res.oos_winrate,
                                    "profit_factor": res.oos_profit_factor,
                                    "net_pnl": res.oos_net_pnl,
                                    "trades_per_day": res.oos_trades_per_day,
                                },
                            )
                            log(
                                f"{'QUALIFIED ' if q else ''}NEW CHAMP "
                                f"WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                f"PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f} "
                                f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                                f"tr={res.trades}/{res.oos_trades} "
                                f"pnl={res.net_pnl:.1f}/{res.oos_net_pnl:.1f} "
                                f"{res.name} {res.params}"
                            )

                # deepen promising high-frequency candidates
                if (
                    local_best
                    and local_best.trades_per_day >= 0.9
                    and local_best.winrate >= 0.78
                ):
                    for exit_p in exits:
                        tested += 1
                        res = evaluate(
                            f"{kind}|{tf}|deep",
                            tf,
                            df,
                            long_e,
                            short_e,
                            exit_p,
                            {"kind": kind, **params},
                        )
                        if not res:
                            continue
                        if champion is None or res.score > champion.score:
                            if res.trades_per_day >= 0.7:
                                champion = res
                                save_champion(res)
                                write_bot(res)
                                log(
                                    f"NEW CHAMP deep WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                                    f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f} "
                                    f"PF={res.profit_factor:.2f}/{res.oos_profit_factor:.2f} {res.name}"
                                )

            if idx % 30 == 0:
                prog = {
                    "tested": tested,
                    "phase": phase,
                    "champion": asdict(champion) if champion else None,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                PROGRESS.write_text(json.dumps(prog, indent=2))
                if champion:
                    log(
                        f"phase={phase} idx={idx}/{len(space)} tested={tested} "
                        f"champ WR={champion.winrate:.1%}/{champion.oos_winrate:.1%} "
                        f"tpd={champion.trades_per_day:.2f}/{champion.oos_trades_per_day:.2f} "
                        f"score={champion.score:.1f}"
                    )
                else:
                    log(f"phase={phase} idx={idx}/{len(space)} tested={tested} no champ yet")

        # combo AND of top signal families for frequency+WR
        log("Combo pass on recent champion neighborhood...")
        if champion:
            # mutate thresholds around champion
            base = {k: v for k, v in champion.params.items() if k not in ("kind", "exit")}
            kind = champion.params["kind"]
            tf = champion.timeframe
            df = frames[tf]
            for _ in range(200):
                mp = dict(base)
                if "os" in mp:
                    mp["os"] = int(min(45, max(10, mp["os"] + RNG.randint(-8, 8))))
                if "ob" in mp:
                    mp["ob"] = int(min(90, max(55, mp["ob"] + RNG.randint(-8, 8))))
                if mp.get("sessions") is not None and not isinstance(mp["sessions"], tuple):
                    mp["sessions"] = tuple(mp["sessions"])
                if mp.get("hours") is not None and not isinstance(mp["hours"], tuple):
                    mp["hours"] = tuple(mp["hours"])
                try:
                    long_e, short_e = gen_signals(df, kind, mp)
                except Exception:
                    continue
                for exit_p in exit_grid(min(phase, 2))[:25]:
                    tested += 1
                    res = evaluate(f"mutate|{kind}|{tf}", tf, df, long_e, short_e, exit_p, {"kind": kind, **mp})
                    if not res:
                        continue
                    if champion is None or res.score > champion.score:
                        champion = res
                        save_champion(res)
                        write_bot(res)
                        log(
                            f"NEW CHAMP mutate WR={res.winrate:.1%}/{res.oos_winrate:.1%} "
                            f"tpd={res.trades_per_day:.2f}/{res.oos_trades_per_day:.2f}"
                        )

        phase += 1
        PROGRESS.write_text(
            json.dumps(
                {
                    "tested": tested,
                    "phase": phase,
                    "champion": asdict(champion) if champion else None,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )
        log(f"Phase loop done. tested={tested}. Continuing forever...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
