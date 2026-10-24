#!/usr/bin/env python3
"""Download multi-year XAU/USD bars from Twelve Data (free-tier friendly)."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

API = "https://api.twelvedata.com/time_series"
OUT = Path("data")
OUT.mkdir(exist_ok=True)


def api_key() -> str:
    k = os.environ.get("TWELVE_DATA_API_KEY")
    if not k and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("TWELVE_DATA_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("'\"")
    if not k:
        raise SystemExit("TWELVE_DATA_API_KEY missing")
    return k


def fetch_chunk(symbol: str, interval: str, end: datetime, key: str, outputsize: int = 5000) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": key,
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "order": "ASC",
        "timezone": "UTC",
    }
    for attempt in range(6):
        r = requests.get(API, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(15 + attempt * 5)
            continue
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            # rate limit message sometimes 200
            msg = str(data.get("message", data))
            if "run out of" in msg.lower() or "api credits" in msg.lower() or "limit" in msg.lower():
                time.sleep(20 + attempt * 10)
                continue
            raise RuntimeError(msg)
        df = pd.DataFrame(data["values"])
        if df.empty:
            return df
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        df = df.set_index("datetime").sort_index()
        return df[["Open", "High", "Low", "Close"]].dropna()
    raise RuntimeError("failed after retries")


def _existing_csvs(symbol: str, interval: str) -> list[Path]:
    stem = f"{symbol.replace('/', '')}_{interval}_"
    return sorted(OUT.glob(f"{stem}*.csv"))


def download(symbol: str, interval: str, years: float = 2.1, sleep_s: float = 8.5) -> Path:
    """Download history; merges any existing local CSVs for the same symbol/interval."""
    key = api_key()
    end = datetime.utcnow()
    start_target = end - timedelta(days=int(365.25 * years))
    parts: list[pd.DataFrame] = []
    # Seed with existing local data so we can resume deeper history without re-fetching
    for old in _existing_csvs(symbol, interval):
        try:
            prev = pd.read_csv(old, parse_dates=["datetime"]).set_index("datetime").sort_index()
            prev = prev[["Open", "High", "Low", "Close"]].dropna()
            if len(prev):
                parts.append(prev)
                print(f"seed existing {old.name}: {len(prev)} {prev.index[0]} → {prev.index[-1]}", flush=True)
        except Exception as e:
            print(f"skip seed {old}: {e}", flush=True)
    cursor = end
    if parts:
        # continue backward from earliest we already have
        earliest_have = min(p.index.min() for p in parts)
        cursor = earliest_have.to_pydatetime() - timedelta(minutes=1)
        print(f"resuming backward from {cursor} toward {start_target.date()}", flush=True)
    calls = 0
    print(f"Downloading {symbol} {interval} back to ~{start_target.date()} ...", flush=True)
    while cursor > start_target:
        df = fetch_chunk(symbol, interval, cursor, key)
        calls += 1
        if df.empty:
            print(f"empty at {cursor}; stop")
            break
        parts.append(df)
        earliest = df.index.min()
        print(
            f"call={calls} rows={len(df)} {df.index.min()} → {df.index.max()} "
            f"earliest_so_far={earliest}",
            flush=True,
        )
        # step before earliest to avoid overlap trap; keep tiny overlap for safety
        cursor = earliest.to_pydatetime() - timedelta(minutes=5)
        if len(df) < 50:
            break
        # checkpoint merge every 10 calls so a kill still keeps progress
        if calls % 10 == 0:
            tmp = pd.concat(parts).sort_index()
            tmp = tmp[~tmp.index.duplicated(keep="last")]
            ck = OUT / f"{symbol.replace('/', '')}_{interval}_PARTIAL.csv"
            tmp.to_csv(ck)
            print(f"checkpoint {ck.name} bars={len(tmp)} start={tmp.index[0]}", flush=True)
        time.sleep(sleep_s)  # free: 8 credits/min
    if not parts:
        raise RuntimeError("no data downloaded")
    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    stem = f"{symbol.replace('/', '')}_{interval}_{full.index[0].date()}_{full.index[-1].date()}"
    csv = OUT / f"{stem}.csv"
    full.to_csv(csv)
    # remove older/partial duplicates for this interval
    for old in _existing_csvs(symbol, interval):
        if old.resolve() != csv.resolve() and ("PARTIAL" in old.name or old.name.startswith(f"{symbol.replace('/', '')}_{interval}_")):
            try:
                if old != csv:
                    old.unlink(missing_ok=True)
            except Exception:
                pass
    out = csv
    try:
        pq = OUT / f"{stem}.parquet"
        full.to_parquet(pq)
        out = pq
    except Exception as e:
        print(f"parquet skipped: {e}", flush=True)
    meta = {
        "symbol": symbol,
        "interval": interval,
        "bars": int(len(full)),
        "start": str(full.index[0]),
        "end": str(full.index[-1]),
        "days": int((full.index[-1] - full.index[0]).days + 1),
        "calls": calls,
        "path": str(out),
        "csv": str(csv),
    }
    Path(OUT / "last_download.json").write_text(__import__("json").dumps(meta, indent=2))
    print("DONE", meta, flush=True)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--interval", default="5min")
    ap.add_argument("--years", type=float, default=2.1)
    ap.add_argument("--sleep", type=float, default=8.5)
    args = ap.parse_args()
    download(args.symbol, args.interval, years=args.years, sleep_s=args.sleep)
