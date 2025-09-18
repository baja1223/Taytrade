#!/usr/bin/env python3
"""
Aggregate monthly tuning results into recommended global settings.

Reads model/tuning_summary.csv (written by tune_monthly.py) and produces
recommended threshold, tp_pct, sl_pct for:
 - all history
 - last 90 days (approx. last 3 months)
 - last 30 days (approx. last 1 month)

Selection rule per window:
 - For each parameter (threshold, tp, sl), choose the value that maximizes
   the sum of returns across months that selected that value (performance‑weighted mode).
 - If ties, fall back to median of unique values.

Outputs JSON at model/recommended_settings.json and prints a short summary.

Usage:
  python src/aggregate_tuning.py --summary model/tuning_summary.csv --out model/recommended_settings.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_month_from_filename(name: str) -> Tuple[int, int]:
    """Extract (year, month) from filenames like 'eth_1m_2024-01.csv'."""
    m = re.search(r"(20\d{2})[-_](0[1-9]|1[0-2])", name)
    if not m:
        return (1970, 1)
    return (int(m.group(1)), int(m.group(2)))


def month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def perf_weighted_mode(values: List[Tuple[float, float]]) -> float:
    """Return the value with the highest total weight (weight = return).
    values: list of (value, return)
    """
    bucket: Dict[float, float] = defaultdict(float)
    for v, ret in values:
        bucket[float(v)] += float(ret)
    if not bucket:
        return 0.0
    # pick argmax by weight; if tie, pick median of keys
    max_w = max(bucket.values())
    candidates = [k for k, w in bucket.items() if abs(w - max_w) < 1e-12]
    if len(candidates) == 1:
        return candidates[0]
    # fall back to median of unique candidates
    candidates.sort()
    mid = len(candidates) // 2
    if len(candidates) % 2 == 1:
        return candidates[mid]
    return 0.5 * (candidates[mid - 1] + candidates[mid])


def aggregate_window(df: pd.DataFrame) -> Dict[str, float]:
    vals_thr = list(zip(df["best_threshold"].astype(float), df["best_return"].astype(float)))
    vals_tp  = list(zip(df["best_tp_pct"].astype(float), df["best_return"].astype(float)))
    vals_sl  = list(zip(df["best_sl_pct"].astype(float), df["best_return"].astype(float)))

    thr = perf_weighted_mode(vals_thr)
    tp  = perf_weighted_mode(vals_tp)
    sl  = perf_weighted_mode(vals_sl)
    return {
        "threshold": round(float(thr), 4),
        "tp_pct": round(float(tp), 4),
        "sl_pct": round(float(sl), 4),
        "months": int(len(df)),
        "avg_return": round(float(df["best_return"].mean()) if len(df) else 0.0, 6),
        "median_return": round(float(df["best_return"].median()) if len(df) else 0.0, 6),
    }


def main():
    ap = argparse.ArgumentParser(description="Aggregate monthly tuning to global settings")
    ap.add_argument('--summary', type=str, default='model/tuning_summary.csv')
    ap.add_argument('--out', type=str, default='model/recommended_settings.json')
    args = ap.parse_args()

    p = Path(args.summary)
    if not p.exists():
        raise SystemExit(f"Summary CSV not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise SystemExit("Empty summary")

    # Extract (year, month) and month index
    ym = df['file'].apply(lambda x: parse_month_from_filename(str(x)))
    df["year"] = ym.apply(lambda t: t[0])
    df["month"] = ym.apply(lambda t: t[1])
    df["midx"] = df.apply(lambda r: month_index(int(r["year"]), int(r["month"])), axis=1)
    latest_midx = int(df["midx"].max())

    # Windows: all, last 90d (~3 months), last 30d (~1 month)
    def window_df(months_back: int | None):
        if months_back is None:
            return df.copy()
        cutoff = latest_midx - max(0, months_back - 1)
        return df[df["midx"] >= cutoff].copy()

    agg_all = aggregate_window(window_df(None))
    agg_90  = aggregate_window(window_df(3))
    agg_30  = aggregate_window(window_df(1))

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": str(p),
        "windows": {
            "all": agg_all,
            "last_90d": agg_90,
            "last_30d": agg_30,
        }
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("Recommended settings:")
    for k, v in out["windows"].items():
        print(f"  {k}: thr={v['threshold']}, tp={v['tp_pct']*100:.2f}%, sl={v['sl_pct']*100:.2f}% (months={v['months']}, avg_ret={v['avg_return']*100:.2f}%)")
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()

