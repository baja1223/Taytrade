#!/usr/bin/env python3
"""
Fetch historical klines from Binance and store monthly CSVs with headers:
timestamp,open,high,low,close,volume

Example:
  python history.py --symbol ETHUSDT --interval 1m --start 2023-01-01 --end 2025-08-10 --out-dir eth_1m_data
  # or: python history.py --days 1000  (backfill N days up to now)
"""

import os
import time
import math
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv


# -------------------- Helpers --------------------

def parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD (UTC midnight)."""
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def month_floor(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(dt: datetime) -> datetime:
    year = dt.year + (dt.month // 12)
    month = (dt.month % 12) + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def safe_write_csv(df: pd.DataFrame, path: str) -> None:
    """Atomic write to avoid partial files."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_month_path(out_dir: str, symbol: str, interval: str, month_dt: datetime) -> str:
    y_m = month_dt.strftime("%Y-%m")
    fname = f"eth_{interval.lower()}_{y_m}.csv"
    return os.path.join(out_dir, fname)


def normalize_klines(klines: List[List]) -> pd.DataFrame:
    """
    Binance klines come as lists:
    [
      open_time, open, high, low, close, volume,
      close_time, quote_asset_volume, num_trades,
      taker_buy_base_volume, taker_buy_quote_volume, ignore
    ]
    """
    if not klines:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)

    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df.rename(columns={"open_time": "timestamp"}, inplace=True)

    # Convert to UTC timestamp (ms) -> datetime ISO or keep ms; we keep ISO for readability
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Coerce numerics
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(subset=["timestamp","open","high","low","close","volume"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    # Final column order
    return df[["timestamp","open","high","low","close","volume"]]


def merge_month(existing_path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(existing_path):
        try:
            old = pd.read_csv(existing_path, parse_dates=["timestamp"], dtype={
                "open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"
            })
            combined = pd.concat([old, new_df], ignore_index=True)
        except Exception:
            # If the existing file is malformed, prefer the new data
            combined = new_df.copy()
    else:
        combined = new_df.copy()

    combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    combined.sort_values("timestamp", inplace=True)
    # Keep only the 6 canonical columns
    return combined[["timestamp","open","high","low","close","volume"]]


# -------------------- Downloader --------------------

def fetch_month(
    client: Client,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    tries: int = 5,
    sleep_base: float = 1.5,
) -> pd.DataFrame:
    """Fetch klines for [start, end) and return normalized OHLCV."""
    assert start.tzinfo is not None and end.tzinfo is not None, "Use timezone-aware datetimes (UTC)."
    for attempt in range(tries):
        try:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=to_ms(start),
                end_str=to_ms(end),
                limit=1000,  # Binance ignores limit for historical wrapper; keep it anyway
            )
            return normalize_klines(klines)
        except BinanceAPIException as e:
            # Throttle / network: exponential backoff
            wait = sleep_base * (2 ** attempt)
            print(f"[warn] BinanceAPIException on {symbol} {interval} {start:%Y-%m}: {e}. Retry in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            wait = sleep_base * (2 ** attempt)
            print(f"[warn] Error on {symbol} {interval} {start:%Y-%m}: {e}. Retry in {wait:.1f}s")
            time.sleep(wait)
    # Final failure returns empty
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])


def backfill_months(
    client: Client,
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    out_dir: str,
    skip_complete: bool = True,
) -> None:
    """Iterate month-by-month and save/merge monthly CSVs with atomic writes."""
    ensure_dir(out_dir)
    start_m = month_floor(start_dt)
    end_m = month_floor(end_dt)

    cur = start_m
    while cur <= end_m:
        nxt = next_month(cur)
        out_path = build_month_path(out_dir, symbol, interval, cur)

        # If file exists and appears complete (last ts >= last second of month), skip
        if skip_complete and os.path.exists(out_path):
            try:
                last_ts = pd.read_csv(out_path, usecols=["timestamp"]).timestamp
                last_ts = pd.to_datetime(last_ts, utc=True).max()
                if last_ts is not pd.NaT and last_ts >= (nxt - timedelta(seconds=1)):
                    print(f"[skip] {cur:%Y-%m} already complete -> {out_path}")
                    cur = nxt
                    continue
            except Exception:
                pass  # fall through to re-fetch/merge

        print(f"[fetch] {symbol} {interval} {cur:%Y-%m} ...")
        month_df = fetch_month(client, symbol, interval, cur, nxt)
        if month_df.empty:
            print(f"[warn] No data returned for {cur:%Y-%m}")
            cur = nxt
            continue

        merged = merge_month(out_path, month_df)
        safe_write_csv(merged, out_path)
        print(f"[save] {out_path}  rows={len(merged):,}")

        # polite rate limit
        time.sleep(0.5)
        cur = nxt


# -------------------- Main / CLI --------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download Binance klines into monthly CSVs with headers.")
    parser.add_argument("--symbol", default=os.getenv("SYMBOL", "ETHUSDT"))
    parser.add_argument("--interval", default=os.getenv("INTERVAL", Client.KLINE_INTERVAL_1MINUTE))
    parser.add_argument("--out-dir", default=os.getenv("DATA_DIR", "eth_1m_data"))
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--days", type=int, help="Backfill this many days up to now (UTC).")
    group.add_argument("--start", type=str, help="YYYY-MM-DD UTC (inclusive start).")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD UTC (exclusive end; default today).")
    parser.add_argument("--skip-complete", action="store_true", default=True, help="Skip months already complete.")
    parser.add_argument("--no-skip-complete", dest="skip_complete", action="store_false")
    args = parser.parse_args()

    api_key = os.getenv("BINANCE_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    if not api_key or not api_secret:
        raise ValueError("Missing BINANCE_KEY / BINANCE_SECRET in environment (or .env)")

    client = Client(api_key, api_secret, {"timeout": 30})

    # Compute time range
    utc_now = datetime.now(timezone.utc)
    if args.days:
        start_dt = utc_now - timedelta(days=args.days)
    elif args.start:
        start_dt = parse_date(args.start)
    else:
        # default: 2 years back
        start_dt = utc_now - timedelta(days=int(os.getenv("HISTORY_DAYS", "730")))

    end_dt = parse_date(args.end) if args.end else utc_now

    print(f"[config] SYM={args.symbol} INT={args.interval} RANGE={start_dt:%Y-%m-%d} -> {end_dt:%Y-%m-%d} OUT={args.out_dir}")
    backfill_months(
        client=client,
        symbol=args.symbol,
        interval=args.interval,
        start_dt=start_dt,
        end_dt=end_dt,
        out_dir=args.out_dir,
        skip_complete=args.skip_complete,
    )
    print("✅ Done.")

if __name__ == "__main__":
    main()