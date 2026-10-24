#!/usr/bin/env python3
"""Download XAUUSD ticks from Dukascopy → 5m OHLC CSV (no API key)."""
from __future__ import annotations

import argparse
import datetime as dt
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT = Path("data")
SCALE = 1000.0  # XAUUSD dukascopy point scale


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    s.mount(
        "https://",
        HTTPAdapter(pool_connections=48, pool_maxsize=48, max_retries=Retry(total=3, backoff_factor=0.4)),
    )
    return s


def hour_url(day: dt.date, hour: int) -> str:
    # Dukascopy months are 0-indexed
    return (
        f"https://datafeed.dukascopy.com/datafeed/XAUUSD/"
        f"{day.year}/{day.month - 1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"
    )


def parse_ticks(raw: bytes, day: dt.date, hour: int) -> list[tuple[pd.Timestamp, float]]:
    if not raw or len(raw) < 20:
        return []
    try:
        data = lzma.decompress(raw)
    except Exception:
        return []
    n = len(data) // 20
    if n == 0:
        return []
    base = dt.datetime(day.year, day.month, day.day, hour, tzinfo=dt.timezone.utc)
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(n):
        ms, ask, bid, _askv, _bidv = struct.unpack(">IIIff", data[i * 20 : (i + 1) * 20])
        mid = (ask + bid) / (2.0 * SCALE)
        ts = pd.Timestamp(base + dt.timedelta(milliseconds=ms)).tz_localize(None)
        out.append((ts, mid))
    return out


def fetch_hour(sess: requests.Session, day: dt.date, hour: int) -> list[tuple[pd.Timestamp, float]]:
    url = hour_url(day, hour)
    try:
        r = sess.get(url, timeout=45)
        if r.status_code != 200 or len(r.content) < 50:
            return []
        return parse_ticks(r.content, day, hour)
    except Exception:
        return []


def iter_hours(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        # gold: skip Sat fully; Sun from 22:00 UTC onward is thin — still fetch all hours, empty OK
        if d.weekday() != 5:  # Saturday
            for h in range(24):
                yield d, h
        d += dt.timedelta(days=1)


def ticks_to_5m(ticks: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    s = pd.Series({t: p for t, p in ticks}).sort_index()
    # mid-only → OHLC from mid path
    ohlc = s.resample("5min").agg(["first", "max", "min", "last"]).dropna()
    ohlc.columns = ["Open", "High", "Low", "Close"]
    return ohlc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.05)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk-days", type=int, default=7)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    end = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=int(365.25 * args.years))
    print(f"Dukascopy XAUUSD {start} → {end} workers={args.workers}", flush=True)

    sess = session()
    parts: list[pd.DataFrame] = []
    hours = list(iter_hours(start, end))
    total = len(hours)
    done = ok_ticks = 0
    t0 = time.time()
    # process in weekly chunks to bound memory
    i = 0
    while i < total:
        chunk = hours[i : i + args.chunk_days * 24]
        i += len(chunk)
        ticks: list[tuple[pd.Timestamp, float]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_hour, sess, d, h): (d, h) for d, h in chunk}
            for f in as_completed(futs):
                done += 1
                rows = f.result()
                if rows:
                    ok_ticks += 1
                    ticks.extend(rows)
                if done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1)
                    eta = (total - done) / max(rate, 1e-6)
                    print(
                        f"  hours {done}/{total} ({100*done/total:.1f}%) "
                        f"hit={ok_ticks} rate={rate:.1f}/s eta={eta/60:.1f}m",
                        flush=True,
                    )
        df5 = ticks_to_5m(ticks)
        if len(df5):
            parts.append(df5)
            # incremental save
            merged = pd.concat(parts).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            stem = f"XAUUSD_5min_dukascopy_{merged.index[0].date()}_{merged.index[-1].date()}.csv"
            path = OUT / stem
            # drop older partials
            for old in OUT.glob("XAUUSD_5min_dukascopy_*.csv"):
                if old != path:
                    old.unlink(missing_ok=True)
            out = merged.reset_index().rename(columns={"index": "datetime"})
            out.to_csv(path, index=False)
            print(
                f"  saved {path.name} bars={len(merged)} "
                f"{merged.index[0]} → {merged.index[-1]}",
                flush=True,
            )

    if not parts:
        raise SystemExit("no data downloaded")
    merged = pd.concat(parts).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    path = OUT / f"XAUUSD_5min_dukascopy_{merged.index[0].date()}_{merged.index[-1].date()}.csv"
    merged.reset_index().rename(columns={"index": "datetime"}).to_csv(path, index=False)
    print(f"DONE {path} bars={len(merged)} days={(merged.index[-1]-merged.index[0]).days}", flush=True)


if __name__ == "__main__":
    main()
