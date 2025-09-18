#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_trade.py — forward-only paper trading with TP/SL and threshold

Defaults:
  • Data directory: ./eth_1m_data
  • Model artifacts: ./ (model_meta.json, model.pt, scaler.joblib)
  • Buy when P(class=1) >= threshold; exit via intra-bar TP/SL
  • Prints a trade blotter and a $-formatted summary

Examples
--------
python paper_trade.py --capital 10000
python paper_trade.py --threshold 0.65 --tp-pct 0.005 --sl-pct 0.0025 --fee-pct 0.0008
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import (
    read_csv_concat_sorted, resolve_price_col, build_windows,
    load_model_bundle, fmt_money, DEFAULT_FEATURE_COLS
)

try:
    import joblib
except Exception:
    joblib = None

try:
    from models import LSTMClassifier
except Exception as e:
    raise SystemExit(
        "Could not import LSTMClassifier from models.py. Ensure models.py is on PYTHONPATH.\n"
        f"Underlying error: {e}"
    )

PRICE_CANDIDATES = ["close", "adj_close", "adj close", "close_price", "price", "last", "mid", "c"]

DEFAULT_COLS_6 = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_COLS_7 = ["timestamp", "open", "high", "low", "close", "volume", "trades"]

def _columns_look_headerless(cols: List[str]) -> bool:
    lowers = [str(c).strip().lower() for c in cols]
    if any(k in lowers for k in ["open","high","low","close","volume","timestamp","time","c","o","h","l","v"]):
        return False
    numeric_like = 0
    for c in cols:
        s = str(c).strip().replace(".", "", 1).replace("-", "", 1)
        if s.isdigit():
            numeric_like += 1
    return numeric_like >= max(3, len(cols)//2)

def _apply_default_headers(df: pd.DataFrame) -> pd.DataFrame:
    n = df.shape[1]
    if n == 6:
        df.columns = DEFAULT_COLS_6
    elif n == 7:
        df.columns = DEFAULT_COLS_7
    else:
        df.columns = [f"col{i}" for i in range(n)]
    return df

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    if _columns_look_headerless(list(df.columns)):
        df = _apply_default_headers(df)
    return df

def load_meta(model_dir: str) -> Dict:
    meta_path = Path(model_dir) / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta.json not found in {model_dir}")
    with open(meta_path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="Paper (simulated) trading with TP/SL and threshold.")
    ap.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or single CSV")
    ap.add_argument("--model-dir", type=str, default=".", help="Where model_meta.json & model.pt live")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--tp-pct", type=float, default=None)
    ap.add_argument("--sl-pct", type=float, default=None)
    ap.add_argument("--fee-pct", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--use-daily-gate", type=int, default=1)
    ap.add_argument("--daily-model-dir", type=str, default="model/daily")
    args = ap.parse_args()

    # Load model + meta
    model, scaler, meta = load_model_bundle(args.model_dir)
    feature_cols = list(meta.get("feature_cols", []))
    window_size = int(meta.get("window_size", 150))
    buy_threshold = float(meta.get("buy_threshold", 0.60)) if args.threshold is None else float(args.threshold)
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = 0.005 if args.tp_pct is None else float(args.tp_pct)
    sl_pct = 0.0025 if args.sl_pct is None else float(args.sl_pct)

    # Load data
    df = read_csv_concat_sorted(args.data_dir)
    price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
    if price_col is None:
        raise SystemExit(f"Could not find a price column. Available: {list(df.columns)}")

    # Ensure required features exist; if not, build a minimal set (mirror backtest)
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        ROLL_WINDOW=20
        ATR_ALPHA=1/14
        RSI_ALPHA=1/14
        for c in ["open","high","low","close"]:
            if c not in df.columns:
                raise SystemExit(f"Missing column {c} for feature computation")
        df = df.copy()
        df["body"] = df["close"] - df["open"]
        rng = (df["high"] - df["low"]) ; df["range"] = rng.replace(0,1e-12)
        df["upper_wick"] = (df["high"] - df[["close","open"]].max(axis=1))
        df["lower_wick"] = (df[["close","open"]].min(axis=1) - df["low"])
        df["return"] = df["close"].pct_change().fillna(0.0)
        df["ret_1"] = df["close"].pct_change(1).fillna(0.0)
        df["ret_2"] = df["close"].pct_change(2).fillna(0.0)
        df["ret_3"] = df["close"].pct_change(3).fillna(0.0)
        df["mom_3"] = (df["close"] - df["close"].shift(3)).fillna(0.0)
        df["mom_6"] = (df["close"] - df["close"].shift(6)).fillna(0.0)
        df["mom_12"] = (df["close"] - df["close"].shift(12)).fillna(0.0)
        sma = df["close"].rolling(ROLL_WINDOW).mean()
        df["sma_ratio"] = (df["close"] / (sma + 1e-12)).fillna(1.0)
        df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        delta = df["close"].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=RSI_ALPHA, adjust=False).mean(); roll_down = down.ewm(alpha=RSI_ALPHA, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)
        roll_up7 = up.ewm(alpha=2/(7+1), adjust=False).mean(); roll_down7 = down.ewm(alpha=2/(7+1), adjust=False).mean()
        rs7 = roll_up7 / (roll_down7 + 1e-12)
        df["rsi_7"] = (100 - (100 / (1 + rs7))).fillna(50.0)
        if "volume" in df.columns:
            df["vol_change"] = df["volume"].pct_change().replace([np.inf,-np.inf],0.0).fillna(0.0)
            df["vol_ema_10"] = df["volume"].ewm(span=10, adjust=False).mean().fillna(method="bfill").fillna(0.0)
        else:
            df["vol_change"] = 0.0; df["vol_ema_10"] = 0.0
        tr = pd.concat([(df["high"]-df["low"]),(df["high"]-df["close"].shift()).abs(),(df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=ATR_ALPHA, adjust=False).mean().fillna(tr.mean())
        ema_12 = df["close"].ewm(span=12, adjust=False).mean(); ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26; signal = macd.ewm(span=9, adjust=False).mean(); df["macd"] = (macd - signal).fillna(0.0)
        hourly = df["close"].ewm(span=60, adjust=False).mean(); df["price_vs_hourly_trend"] = (df["close"]/(hourly+1e-12)).fillna(1.0)
        std_5 = df["close"].rolling(5).std(); std_10 = df["close"].rolling(10).std(); std_20 = df["close"].rolling(20).std()
        upper = sma + 2*std_20; lower = sma - 2*std_20; df["bb_width"] = ((upper - lower)/(sma+1e-12)).fillna(0.0)
        df["std_5"] = std_5.fillna(method="bfill").fillna(0.0); df["std_10"] = std_10.fillna(method="bfill").fillna(0.0)
        df["hl_pct"] = ((df["high"]-df["low"]) / (df["close"] + 1e-12)).fillna(0.0)
        df["oc_pct"] = ((df["close"]-df["open"]) / (df["open"] + 1e-12)).fillna(0.0)
        mean5 = df["close"].rolling(5).mean(); mean10 = df["close"].rolling(10).mean()
        df["zscore_5"] = ((df["close"]-mean5)/(std_5+1e-12)).replace([np.inf,-np.inf],0.0).fillna(0.0)
        df["zscore_10"] = ((df["close"]-mean10)/(std_10+1e-12)).replace([np.inf,-np.inf],0.0).fillna(0.0)
        return df

    # Compute features if needed
    if any(c not in df.columns for c in feature_cols):
        df = compute_features(df)

    # Build feature matrix in the SAME order as meta
    drop_cols = {price_col, "timestamp", "time"}
    feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
    if not feat_cols:
        raise SystemExit("No valid feature columns found for inference.")
    X_flat = df[feat_cols].to_numpy(dtype=np.float32)

    # Windows
    X = build_windows(X_flat, window_size)
    if len(X) == 0:
        raise SystemExit("Not enough rows to build any sequences. Increase data or reduce window size.")

    # Align OHLC arrays to window ends
    opens = df["open"].to_numpy(dtype=float)[window_size - 1:]
    highs = df["high"].to_numpy(dtype=float)[window_size - 1:]
    lows  = df["low"].to_numpy(dtype=float)[window_size - 1:]
    closes= df[price_col].to_numpy(dtype=float)[window_size - 1:]
    times = df["timestamp"].iloc[window_size - 1:] if "timestamp" in df.columns else pd.Series(range(len(closes)))

    # Scale if scaler exists
    if scaler is not None:
        n, t, f = X.shape
        X = scaler.transform(X.reshape(n*t, f)).reshape(n, t, f)

    # Predict probabilities (class 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    probs = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        BS = int(args.batch_size)
        for i in range(0, len(X), BS):
            xb = torch.from_numpy(X[i:i+BS]).to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs[i:i+BS] = p

    # Optional: daily gating (requires model/daily artifacts)
    gate = None
    if int(getattr(args, 'use_daily_gate', 1)) and joblib is not None:
        try:
            daily_dir = Path(getattr(args, 'daily_model_dir', 'model/daily'))
            meta_path = daily_dir / 'daily_meta.json'
            if meta_path.exists():
                daily_meta = json.loads(meta_path.read_text())
                clf = joblib.load(daily_dir / daily_meta['paths']['model'])
                dscaler = joblib.load(daily_dir / daily_meta['paths']['scaler'])
                thr = float(daily_meta.get('threshold', 0.5))
                # Build daily features from raw minute data
                raw = read_csv_concat_sorted(args.data_dir)
                if 'timestamp' in raw.columns:
                    ts = pd.to_datetime(raw['timestamp'], unit='ms')
                else:
                    ts = pd.to_datetime(raw.index)
                raw = raw.copy(); raw.index = ts
                agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
                daily = raw.resample('1D').apply(agg).dropna()
                daily = daily[daily['volume'] > 0]
                if not daily.empty:
                    dfD = daily.copy()
                    dfD['ret_1'] = dfD['close'].pct_change().fillna(0.0)
                    dfD['ema_5'] = dfD['close'].ewm(span=5, adjust=False).mean()
                    dfD['ema_10'] = dfD['close'].ewm(span=10, adjust=False).mean()
                    dfD['ema_20'] = dfD['close'].ewm(span=20, adjust=False).mean()
                    dfD['sma_20'] = dfD['close'].rolling(20).mean()
                    dfD['sma_ratio'] = (dfD['close'] / (dfD['sma_20'] + 1e-12)).fillna(1.0)
                    delta = dfD['close'].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
                    roll_up = up.ewm(alpha=1/14, adjust=False).mean(); roll_down = down.ewm(alpha=1/14, adjust=False).mean()
                    rs = roll_up / (roll_down + 1e-12); dfD['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50.0)
                    tr = pd.concat([(dfD['high']-dfD['low']),(dfD['high']-dfD['close'].shift()).abs(),(dfD['low']-dfD['close'].shift()).abs()],axis=1).max(axis=1)
                    dfD['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean().fillna(tr.mean())
                    dfD['std_10'] = dfD['close'].rolling(10).std().bfill().fillna(0.0)
                    dfD['vol_change'] = dfD['volume'].pct_change().replace([np.inf,-np.inf],0.0).fillna(0.0)
                    feat_cols_daily = daily_meta.get('feature_cols', ['open','high','low','close','volume','ret_1','ema_5','ema_10','ema_20','sma_ratio','rsi_14','atr_14','std_10','vol_change'])
                    Xd = dfD[feat_cols_daily].to_numpy(dtype=np.float32)
                    Xds = dscaler.transform(Xd)
                    pday = clf.predict_proba(Xds)[:,1]
                    # map to minute sequence indices
                    if hasattr(times, 'iloc'):
                        tser = pd.to_datetime(times)
                    else:
                        tser = pd.to_datetime(pd.Series(times))
                    days_end = tser.dt.floor('D')
                    dmap = {idx: (prob >= thr) for idx, prob in zip(dfD.index, pday)}
                    gate = np.array([dmap.get(d, True) for d in days_end], dtype=bool)
                    print(f"[DAILY] Paper gating enabled. True rate={gate.mean():.2%}")
        except Exception as e:
            print(f"[WARN] Daily gating disabled due to error: {e}")

    # Paper trading loop (same logic as backtester)
    cash = float(args.capital)
    in_trade = False
    entry_price = None
    tp_price = None
    sl_price = None
    trades = []
    wins = losses = 0

    for i in range(1, len(closes)):
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        t = times.iloc[i] if hasattr(times, "iloc") else times[i]

        if not in_trade:
            allowed = True if gate is None else bool(gate[i])
            if (probs[i-1] >= buy_threshold) and allowed:
                entry_price = o
                tp_price = entry_price * (1.0 + tp_pct)
                sl_price = entry_price * (1.0 - sl_pct)
                cash *= (1.0 - fee_pct)  # entry fee
                in_trade = True
                trades.append({
                    "time": t, "side": "BUY", "price": entry_price, "prob": float(probs[i-1])
                })
        else:
            exit_px = None
            win = None
            # conservative order: SL before TP if both touched (path dependent)
            if l <= sl_price <= h:
                exit_px = sl_price
                win = False
            elif h >= tp_price:
                exit_px = tp_price
                win = True
            elif l <= sl_price:
                exit_px = sl_price
                win = False

            if exit_px is not None:
                gross_ret = (exit_px / entry_price) - 1.0
                cash *= (1.0 + gross_ret)
                cash *= (1.0 - fee_pct)  # exit fee
                in_trade = False
                trades.append({
                    "time": t, "side": "SELL", "price": exit_px, "result": "WIN" if win else "LOSS"
                })
                if win: wins += 1
                else: losses += 1

    # If still open at end, close at last close
    if in_trade and entry_price is not None:
        final_exit = float(closes[-1])
        gross_ret = (final_exit / entry_price) - 1.0
        cash *= (1.0 + gross_ret)
        cash *= (1.0 - fee_pct)
        trades.append({"time": times.iloc[-1] if hasattr(times,"iloc") else times[-1], "side": "SELL", "price": final_exit, "result": "CLOSE_END"})

    # Blotter + summary
    print("\n=== PAPER TRADE BLOTTER ===")
    for tr in trades:
        if tr["side"] == "BUY":
            print(f"{tr['time']}  BUY  @ {fmt_money(tr['price'])}  (prob={tr['prob']:.3f})")
        else:
            res = tr.get("result", "")
            print(f"{tr['time']}  SELL @ {fmt_money(tr['price'])}  {res}")

    start = float(args.capital)
    end = cash
    multiple = end / start if start else float("nan")
    ret = multiple - 1.0
    print("\n=== PAPER TRADE SUMMARY ===")
    print(f"Start capital  : {fmt_money(start)}")
    print(f"End equity     : {fmt_money(end)}")
    print(f"Return         : {ret*100:.2f}%  (×{multiple:.2f})")
    total_trades = sum(1 for tr in trades if tr["side"] == "SELL")
    if total_trades:
        print(f"Trades         : {total_trades}  (wins {wins}, losses {losses}, win rate {wins/max(1,total_trades):.2%})")
    print("")

if __name__ == "__main__":
    main()
