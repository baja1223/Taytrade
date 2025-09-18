#!/usr/bin/env python3
"""
Train a daily classification model to gate intraday trading.

Data: minute CSVs (6 columns: timestamp, open, high, low, close, volume)
Process:
  1) Concatenate all CSVs, resample/aggregate to daily OHLCV
  2) Compute daily features
  3) Label: next_day_up = 1 if next close > close else 0
  4) Train LogisticRegression with StandardScaler
  5) Tune probability threshold over [0.4..0.8] to maximize F1
  6) Save artifacts under <model-dir>/daily/

Usage:
  python src/train_daily.py --data-dir eth_1m_data --model-dir model --days 730
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib


def list_csvs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(path)
    files = sorted(p.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f"No CSVs under {path}")
    return files


def load_minute_concat(path: str) -> pd.DataFrame:
    frames = []
    for fp in list_csvs(path):
        df = pd.read_csv(fp, header=None)
        if df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = ['timestamp','open','high','low','close','volume']
            frames.append(df)
    if not frames:
        raise SystemExit("No valid CSVs found")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna()
    return df


def resample_daily(dfm: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(dfm['timestamp'], unit='ms')
    dfm = dfm.copy()
    dfm.index = ts
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    d = dfm.resample('1D').apply(agg).dropna()
    d = d[d['volume'] > 0]
    return d


def compute_daily_features(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out['ret_1'] = out['close'].pct_change().fillna(0.0)
    out['ema_5'] = out['close'].ewm(span=5, adjust=False).mean()
    out['ema_10'] = out['close'].ewm(span=10, adjust=False).mean()
    out['ema_20'] = out['close'].ewm(span=20, adjust=False).mean()
    out['sma_20'] = out['close'].rolling(20).mean()
    out['sma_ratio'] = (out['close'] / (out['sma_20'] + 1e-12)).fillna(1.0)
    # RSI(14)
    delta = out['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    out['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50.0)
    # ATR(14)
    tr = pd.concat([
        (out['high'] - out['low']),
        (out['high'] - out['close'].shift()).abs(),
        (out['low'] - out['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    out['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean().fillna(tr.mean())
    # Volatility proxy
    out['std_10'] = out['close'].rolling(10).std().bfill().fillna(0.0)
    # Volume change
    out['vol_change'] = out['volume'].pct_change().replace([np.inf,-np.inf],0.0).fillna(0.0)
    out = out.dropna()
    return out


def main():
    ap = argparse.ArgumentParser(description='Train daily gating model')
    ap.add_argument('--data-dir', type=str, default='eth_1m_data')
    ap.add_argument('--model-dir', type=str, default='model')
    ap.add_argument('--days', type=int, default=1000)
    args = ap.parse_args()

    dfm = load_minute_concat(args.data_dir)
    d = resample_daily(dfm)
    if len(d) < 60:
        raise SystemExit('Not enough daily rows to train')

    # recent window
    d = d.tail(args.days)
    feats = compute_daily_features(d)
    # Label: next-day up
    feats['label'] = (feats['close'].shift(-1) > feats['close']).astype(int)
    feats = feats.dropna()

    feature_cols = [
        'open','high','low','close','volume','ret_1','ema_5','ema_10','ema_20','sma_ratio','rsi_14','atr_14','std_10','vol_change'
    ]
    X = feats[feature_cols].to_numpy(dtype=np.float32)
    y = feats['label'].to_numpy(dtype=np.int64)

    # chronological split 80/20
    n = len(feats)
    k = max(1, int(n * 0.8))
    X_tr, X_va = X[:k], X[k:]
    y_tr, y_va = y[:k], y[k:]

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)

    clf = LogisticRegression(max_iter=500, n_jobs=None)
    clf.fit(X_trs, y_tr)

    # Threshold tuning
    prob_va = clf.predict_proba(X_vas)[:, 1]
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.4, 0.8, 41):
        pred = (prob_va >= t).astype(int)
        f1 = f1_score(y_va, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    outdir = Path(args.model_dir) / 'daily'
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, outdir / 'daily_model.joblib')
    joblib.dump(scaler, outdir / 'daily_scaler.joblib')
    meta = {
        'model_type': 'daily_logreg',
        'feature_cols': feature_cols,
        'threshold': best_t,
        'f1_val': best_f1,
        'samples': int(len(feats)),
        'split': {'train': int(len(X_tr)), 'val': int(len(X_va))},
        'paths': {'model': 'daily_model.joblib', 'scaler': 'daily_scaler.joblib'},
    }
    (outdir / 'daily_meta.json').write_text(json.dumps(meta, indent=2))
    print(f"Saved daily model to {outdir}")


if __name__ == '__main__':
    main()

