#!/usr/bin/env python3
"""
Tune threshold and TP/SL per month (per-CSV) to target 2–5% moves.

Loads a trained model bundle (model_meta.json, model.pt, scaler.joblib),
computes the same features as training, predicts probabilities once per file,
and grid-searches threshold and TP/SL to maximize portfolio return on that file.

Usage examples:
  python tune_monthly.py --data-dir eth_1m_data --model-dir model \
    --threshold-range 0.4,0.8,0.02 --tp-range 0.02,0.05,0.005 --sl-range 0.005,0.02,0.0025 \
    --confirm-bars 2 --low-stop-pct 0.20

Writes a summary JSON and CSV under model/ (tuning_summary.json, tuning_summary.csv).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from backtest import (
    load_model_bundle,  # loads training LSTMModel architecture
    compute_features,
    build_windows,
    predict_probs,
    simulate_trades_with_tp_sl,
)


def parse_range(spec: str, default: List[float]) -> List[float]:
    try:
        parts = [float(x.strip()) for x in spec.split(',')]
        if len(parts) == 3:
            start, end, step = parts
            if step <= 0:
                step = (end - start) / 10.0 if end > start else 0.01
            vals = []
            v = start
            while v <= end + 1e-12:
                vals.append(round(v, 6))
                v += step
            return vals
        elif len(parts) == 2:
            start, end = parts
            step = (end - start) / 10.0 if end > start else 0.01
            vals = []
            v = start
            while v <= end + 1e-12:
                vals.append(round(v, 6))
                v += step
            return vals
        elif len(parts) == 1:
            return [parts[0]]
    except Exception:
        pass
    return default


def list_month_files(data_dir: str) -> List[Path]:
    p = Path(data_dir)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(data_dir)
    files = sorted(p.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f"No CSVs in {data_dir}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Monthly tuner for threshold and TP/SL")
    ap.add_argument('--data-dir', type=str, default='eth_1m_data')
    ap.add_argument('--model-dir', type=str, default='model')
    ap.add_argument('--capital', type=float, default=10_000.0)
    ap.add_argument('--fee-pct', type=float, default=0.0008)
    ap.add_argument('--confirm-bars', type=int, default=2)
    ap.add_argument('--low-stop-pct', type=float, default=0.20)
    ap.add_argument('--threshold-range', type=str, default='0.40,0.80,0.02')
    ap.add_argument('--tp-range', type=str, default='0.02,0.05,0.005')
    ap.add_argument('--sl-range', type=str, default='0.005,0.02,0.0025')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = ap.parse_args()

    model, scaler, meta = load_model_bundle(args.model_dir)
    feature_cols = list(meta.get('feature_cols', []))
    window_size = int(meta.get('window_size', 150))
    device = (
        torch.device('cpu') if args.device == 'cpu' else
        torch.device('cuda') if (args.device == 'cuda' and torch.cuda.is_available()) else
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model = model.to(device)

    thr_vals = parse_range(args.threshold_range, [0.5, 0.55, 0.6, 0.65, 0.7])
    tp_vals  = parse_range(args.tp_range, [0.02, 0.03, 0.04, 0.05])
    sl_vals  = parse_range(args.sl_range, [0.005, 0.01, 0.015, 0.02])

    results: List[Dict] = []

    for csv_path in list_month_files(args.data_dir):
        # Load one month
        df = pd.read_csv(csv_path, header=None)
        # Apply default headers if headerless
        if df.shape[1] == 6:
            df.columns = ['timestamp','open','high','low','close','volume']
        elif df.shape[1] == 7:
            df.columns = ['timestamp','open','high','low','close','volume','trades']
        else:
            df.columns = [f'col{i}' for i in range(df.shape[1])]

        # Compute features same as training/backtest
        df = compute_features(df)
        drop_cols = {'timestamp','time'}
        feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
        if not feat_cols or len(df) < window_size + 1:
            print(f"[SKIP] {csv_path.name}: not enough rows or missing features")
            continue

        X_flat = df[feat_cols].to_numpy(dtype=np.float32)
        X = build_windows(X_flat, window_size)
        opens = df['open'].to_numpy(dtype=float)[window_size - 1:]
        highs = df['high'].to_numpy(dtype=float)[window_size - 1:]
        lows  = df['low'].to_numpy(dtype=float)[window_size - 1:]
        closes= df['close'].to_numpy(dtype=float)[window_size - 1:]

        if scaler is not None:
            n, t, f = X.shape
            X = scaler.transform(X.reshape(n*t, f)).reshape(n, t, f)

        probs = predict_probs(model, X, int(args.batch_size), device)

        best = {"ret": -1e9, "thr": None, "tp": None, "sl": None}
        for thr in thr_vals:
            for tp in tp_vals:
                for sl in sl_vals:
                    rpt, _ = simulate_trades_with_tp_sl(
                        opens, highs, lows, closes, probs,
                        threshold=float(thr), start_capital=float(args.capital),
                        fee_pct=float(args.fee_pcnt) if hasattr(args, 'fee_pcnt') else float(args.fee_pct),
                        tp_pct=float(tp), sl_pct=float(sl),
                        low_stop_pct=float(args.low_stop_pct),
                        reenter_at_same_price=True,
                        confirm_bars=int(args.confirm_bars),
                    )
                    ret = rpt.get('portfolio',{}).get('return', -1e9)
                    if ret > best['ret']:
                        best.update({"ret": ret, "thr": float(thr), "tp": float(tp), "sl": float(sl)})

        results.append({
            "file": csv_path.name,
            "n": int(len(probs)),
            "best_return": float(best['ret']),
            "best_threshold": float(best['thr']) if best['thr'] is not None else None,
            "best_tp_pct": float(best['tp']) if best['tp'] is not None else None,
            "best_sl_pct": float(best['sl']) if best['sl'] is not None else None,
        })
        print(f"{csv_path.name}: best thr={best['thr']}, tp={best['tp']}, sl={best['sl']}, return={best['ret']*100:.2f}%")

    if not results:
        print("No results computed.")
        return

    outdir = Path(args.model_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv(outdir / 'tuning_summary.csv', index=False)
    (outdir / 'tuning_summary.json').write_text(json.dumps(results, indent=2))
    print(f"Saved tuning_summary.csv and tuning_summary.json in {outdir}")


if __name__ == '__main__':
    main()
