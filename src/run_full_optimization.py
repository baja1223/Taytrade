#!/usr/bin/env python3
"""
Run end-to-end optimization for short-horizon (1–3m) model targeting 2–5% moves:
  1) Train model (window=60 default)
  2) Tune per-month threshold + TP/SL in target ranges (2–5% TP, 0.5–2% SL)
  3) Aggregate monthly results into recommended global settings (all/90d/30d)

Usage:
  python src/run_full_optimization.py \
    --data-dir eth_1m_data --model-dir model \
    --epochs 30 --window-size 60 --batch-size 512 \
    --threshold-range 0.4,0.8,0.02 --tp-range 0.02,0.05,0.005 --sl-range 0.005,0.02,0.0025 \
    --confirm-bars 2 --low-stop-pct 0.20

You can override any argument as needed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description="End-to-end short-horizon optimizer (train + tune + aggregate)")
    ap.add_argument('--data-dir', type=str, default='eth_1m_data')
    ap.add_argument('--model-dir', type=str, default='model')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--window-size', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--threshold-range', type=str, default='0.4,0.8,0.02')
    ap.add_argument('--tp-range', type=str, default='0.02,0.05,0.005')
    ap.add_argument('--sl-range', type=str, default='0.005,0.02,0.0025')
    ap.add_argument('--confirm-bars', type=int, default=2)
    ap.add_argument('--low-stop-pct', type=float, default=0.20)
    ap.add_argument('--fee-pct', type=float, default=0.0008)
    args = ap.parse_args()

    py = os.getenv('PYTHON', sys.executable or 'python')
    base = Path(__file__).resolve().parent

    # 1) Train
    run([py, str(base / 'train_model.py'),
         '--data-path', args.data_dir,
         '--output-dir', args.model_dir,
         '--window-size', str(args.window_size),
         '--epochs', str(args.epochs),
         '--batch-size', str(args.batch_size),
         '--workers', '0',
         '--amp', 'false',
    ])

    # 2) Tune monthly (2–5% TP focus)
    run([py, str(base / 'tune_monthly.py'),
         '--data-dir', args.data_dir,
         '--model-dir', args.model_dir,
         '--threshold-range', args.threshold_range,
         '--tp-range', args.tp_range,
         '--sl-range', args.sl_range,
         '--confirm-bars', str(args.confirm_bars),
         '--low-stop-pct', str(args.low_stop_pct),
         '--batch-size', str(args.batch_size),
    ])

    # 3) Aggregate to recommended settings (all/90d/30d)
    run([py, str(base / 'aggregate_tuning.py'),
         '--summary', str(Path(args.model_dir) / 'tuning_summary.csv'),
         '--out', str(Path(args.model_dir) / 'recommended_settings.json')])

    print("\nAll done. See:")
    print(f"  - {args.model_dir}/model.pt, scaler.joblib, model_meta.json")
    print(f"  - {args.model_dir}/tuning_summary.csv")
    print(f"  - {args.model_dir}/recommended_settings.json")


if __name__ == '__main__':
    main()

