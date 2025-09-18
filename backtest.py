#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py — Backtester with TP/SL, reading model_meta.json and
recomputing the same features used at training.

Usage
-----
python backtest.py --data-dir eth_1m_data --model-dir model
"""

from __future__ import annotations

import argparse
import json
import os
import math as _math
from typing import Dict, Tuple, Optional
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils import (
    read_csv_concat_sorted, resolve_price_col, build_windows,
    fmt_money, fmt_pct, DEFAULT_FEATURE_COLS
)

from train_model import LSTMModel  # matches training architecture
import joblib

# =========================
# Feature engineering (same as train_model.py)
# =========================
ROLL_WINDOW = 20
ATR_ALPHA = 1 / 14
RSI_ALPHA = 1 / 14

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")
    df = df.copy()
    df["body"] = df["close"] - df["open"]
    rng = (df["high"] - df["low"])
    df["range"] = rng.replace(0, 1e-12)
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1))
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"])
    df["return"] = df["close"].pct_change().fillna(0.0)
    # Short-horizon features
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
    delta = df["close"].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=RSI_ALPHA, adjust=False).mean()
    roll_down = down.ewm(alpha=RSI_ALPHA, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)
    # RSI(7)
    roll_up7 = up.ewm(alpha=2/(7+1), adjust=False).mean()
    roll_down7 = down.ewm(alpha=2/(7+1), adjust=False).mean()
    rs7 = roll_up7 / (roll_down7 + 1e-12)
    df["rsi_7"] = (100 - (100 / (1 + rs7))).fillna(50.0)
    if "volume" in df.columns:
        df["vol_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["vol_ema_10"] = df["volume"].ewm(span=10, adjust=False).mean().fillna(method="bfill").fillna(0.0)
    else:
        df["vol_change"] = 0.0
        df["vol_ema_10"] = 0.0
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=ATR_ALPHA, adjust=False).mean().fillna(tr.mean())
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = (macd - signal).fillna(0.0)
    hourly = df["close"].ewm(span=60, adjust=False).mean()
    df["price_vs_hourly_trend"] = (df["close"] / (hourly + 1e-12)).fillna(1.0)
    std_5 = df["close"].rolling(5).std()
    std_10 = df["close"].rolling(10).std()
    std_20 = df["close"].rolling(ROLL_WINDOW).std()
    upper = sma + 2 * std_20
    lower = sma - 2 * std_20
    df["bb_width"] = ((upper - lower) / (sma + 1e-12)).fillna(0.0)
    df["std_5"] = std_5.fillna(method="bfill").fillna(0.0)
    df["std_10"] = std_10.fillna(method="bfill").fillna(0.0)

    # Microstructure-like ratios
    df["hl_pct"] = ((df["high"] - df["low"]) / (df["close"] + 1e-12)).fillna(0.0)
    df["oc_pct"] = ((df["close"] - df["open"]) / (df["open"] + 1e-12)).fillna(0.0)

    # Rolling z-scores
    mean5 = df["close"].rolling(5).mean()
    z5 = (df["close"] - mean5) / (std_5 + 1e-12)
    mean10 = df["close"].rolling(10).mean()
    z10 = (df["close"] - mean10) / (std_10 + 1e-12)
    df["zscore_5"] = z5.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["zscore_10"] = z10.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df

# =========================
# Model/scaler loader
# =========================
def load_model_bundle(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = os.path.join(model_dir, "model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    model = LSTMModel(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
        bidirectional=meta.get("bidirectional", False),
        num_classes=int(meta.get("num_classes", 2)),
        regression=False,
    ).to(device)
    ckpt_path = os.path.join(model_dir, meta.get("model_state_path", "model.pt"))
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    scaler = None
    scaler_path = os.path.join(model_dir, meta.get("scaler_path", "scaler.joblib"))
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    return model, scaler, meta

# =========================
# Pretty formatting
# =========================
def print_portfolio_report(report: Dict, currency: str = "$") -> None:
    m = (report or {}).get("metrics", {}) or {}
    p = (report or {}).get("portfolio", {}) or {}
    n = int(m.get("n", 0))
    start = p.get("start_capital", 0.0)
    end = p.get("end_equity", None)
    trades = int(p.get("trades", 0))
    wins = int(p.get("wins", 0))
    losses = int(p.get("losses", 0))
    mdd = p.get("max_drawdown", None)
    multiple = float(end) / float(start) if start not in (None, 0) and end is not None else None
    print("\n=== PORTFOLIO MODE — SUMMARY ===")
    print(f"Bars processed : {n:,}")
    print(f"Trades         : {trades:,}  (wins {wins}, losses {losses}, win rate {wins/max(1,trades):.2%})")
    print(f"Start capital  : {fmt_money(start, currency)}")
    print(f"End equity     : {fmt_money(end, currency)}")
    if multiple is not None and np.isfinite(multiple):
        print(f"Return         : {fmt_pct(multiple-1.0)}  (×{multiple:.2f})")
    else:
        print("Return         : —")
    if mdd is not None:
        print(f"Max drawdown   : {fmt_pct(mdd)}")
    print("")

# =========================
# Trade simulation
# =========================
def simulate_trades_with_tp_sl(opens, highs, lows, closes, probs, *, threshold, start_capital,
                               fee_pct=0.0008, tp_pct=0.005, sl_pct=0.0025,
                               low_stop_pct: float = 0.20,
                               reenter_at_same_price: bool = True,
                               confirm_bars: int = 1,
                               gate: Optional[np.ndarray] = None,
                               signal_exit: bool = False,
                               exit_threshold: Optional[float] = None) -> Tuple[Dict, pd.DataFrame]:
    n = len(closes); cash = float(start_capital); in_trade = False
    entry_price = tp_price = sl_price = None
    last_entry = None
    reentry_target = None
    confirm = 0
    equity_curve = np.empty(n, dtype=float); equity_curve[0] = cash
    trades = wins = losses = 0
    for i in range(1, n):
        o, hi, lo, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        if not in_trade:
            # Optional consecutive-bar confirmation to reduce noise
            confirm = confirm + 1 if probs[i-1] >= threshold else 0
            ready = confirm >= max(1, int(confirm_bars))
            allowed = True if gate is None else bool(gate[i])
            if reentry_target is not None:
                # Re-enter only once price has recovered to the original entry level
                if hi >= reentry_target:
                    cash *= (1.0 - fee_pct); entry_price = reentry_target
                    tp_price = entry_price * (1.0 + tp_pct); sl_price = entry_price * (1.0 - sl_pct)
                    in_trade = True; trades += 1
                    last_entry = entry_price; reentry_target = None
            elif ready and allowed:
                cash *= (1.0 - fee_pct); entry_price = o
                tp_price = entry_price * (1.0 + tp_pct); sl_price = entry_price * (1.0 - sl_pct)
                in_trade = True; trades += 1
                last_entry = entry_price
        else:
            exit_price = None; win = None
            # Low-stop: exit if drawdown from last entry exceeds low_stop_pct
            if last_entry is not None and c <= last_entry * (1.0 - low_stop_pct):
                exit_price = max(lo, last_entry * (1.0 - low_stop_pct)); win = False
            elif lo <= sl_price <= hi:   exit_price = sl_price; win = False
            elif hi >= tp_price:         exit_price = tp_price; win = True
            elif lo <= sl_price:         exit_price = sl_price; win = False
            # Optional signal-based exit to increase turnover
            if exit_price is None and signal_exit and exit_threshold is not None:
                if probs[i] < exit_threshold:
                    exit_price = c
                    win = (exit_price >= entry_price)
            if exit_price is not None:
                cash *= (1.0 + (exit_price / entry_price) - 1.0); cash *= (1.0 - fee_pct)
                in_trade = False; entry_price = tp_price = sl_price = None
                wins += int(win); losses += int(not win)
                # If low-stop, optionally set reentry target at the original entry price
                if (not win) and reenter_at_same_price and last_entry is not None and exit_price <= last_entry * (1.0 - low_stop_pct) + 1e-12:
                    reentry_target = float(last_entry)
                last_entry = None
            else:
                mtm = (c / entry_price) - 1.0
                equity_curve[i] = cash * (1.0 + mtm); continue
        equity_curve[i] = cash
    if in_trade and entry_price is not None:
        cash *= (1.0 + (closes[-1] / entry_price) - 1.0); cash *= (1.0 - fee_pct)
    peaks = np.maximum.accumulate(equity_curve); dd = (equity_curve - peaks) / peaks
    report = {
        "metrics": {"n": int(n)},
        "portfolio": {
            "start_capital": float(start_capital),
            "end_equity": float(equity_curve[-1]),
            "return": float(equity_curve[-1] / max(1e-12, start_capital) - 1.0),
            "max_drawdown": float(abs(np.min(dd)) if len(dd) else 0.0),
            "trades": int(trades), "wins": int(wins), "losses": int(losses),
        },
    }
    df_curve = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                             "equity": equity_curve, "prob": probs})
    return report, df_curve

# =========================
# Robust prediction (auto-shrinks batch on CUDA OOM, CPU fallback)
# =========================
def predict_probs(model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    probs = np.zeros(len(X), dtype=np.float32)
    i = 0
    bs = max(1, int(batch_size))
    while i < len(X):
        try:
            xb_np = X[i:i+bs]
            if len(xb_np) == 0:
                break
            xb = torch.from_numpy(xb_np).to(device, non_blocking=(device.type == "cuda"))
            with torch.no_grad():
                logits = model(xb)
                p = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs[i:i+len(p)] = p
            i += len(p)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda" and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[WARN] CUDA OOM — reducing batch size to {bs}")
                continue
            # If still failing on CUDA with bs=1, fall back to CPU
            if "out of memory" in msg and device.type == "cuda":
                print("[WARN] CUDA OOM at batch size 1 — falling back to CPU")
                device = torch.device("cpu")
                model = model.to(device)
                torch.cuda.empty_cache()
                continue
            raise
    return probs

# =========================
# CLI
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtester with TP/SL and probability threshold.")
    p.add_argument("--mode", choices=["simple", "portfolio"], default="portfolio")
    p.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or a single CSV")
    p.add_argument("--model-dir", type=str, default="model", help="Root where model_meta.json & model.pt live")
    p.add_argument("--threshold", type=float, default=None, help="Buy threshold (default from model_meta.json)")
    p.add_argument("--tp-pct", type=float, default=None, help="Take-profit as fraction (0.005 = 0.5%)")
    p.add_argument("--sl-pct", type=float, default=None, help="Stop-loss as fraction (0.0025 = 0.25%)")
    p.add_argument("--fee-pct", type=float, default=None, help="Per-side fee fraction (0.0008 = 0.08%)")
    p.add_argument("--capital", type=float, default=10_000.0, help="Starting capital for portfolio mode")
    p.add_argument("--batch-size", type=int, default=512, help="Prediction batch size (auto-shrinks if OOM)")  # smaller default
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force device (default auto)")
    p.add_argument("--low-stop-pct", type=float, default=0.20, help="Low stop drawdown from entry (e.g., 0.20 = 20%)")
    p.add_argument("--reenter-same", type=int, default=1, help="Re-enter at original entry price after low stop (1/0)")
    p.add_argument("--confirm-bars", type=int, default=1, help="Consecutive bars above threshold required to enter")
    p.add_argument("--tune-threshold", type=int, default=0, help="Grid-search threshold for best return (1/0)")
    p.add_argument("--tune-tpsl", type=int, default=0, help="Grid-search TP/SL over ranges for best return (1/0)")
    p.add_argument("--tp-range", type=str, default="0.02,0.05,0.005", help="TP range: start,end,step (fractions)")
    p.add_argument("--sl-range", type=str, default="0.005,0.02,0.0025", help="SL range: start,end,step (fractions)")
    p.add_argument("--use-daily-gate", type=int, default=1, help="Use daily gating model if available (1/0)")
    p.add_argument("--daily-model-dir", type=str, default="model/daily", help="Directory with daily_meta.json")
    # Trade frequency controls
    p.add_argument("--signal-exit", type=int, default=0, help="Enable signal-based exit to increase trade count (1/0)")
    p.add_argument("--exit-threshold", type=float, default=None, help="Exit threshold when --signal-exit=1 (defaults to threshold*0.9)")
    p.add_argument("--min-trades", type=int, default=0, help="Minimum trades constraint during tuning (prefer params with >= min trades)")
    p.add_argument("--optimize-for", choices=["return","trades","ret_per_trade"], default="return", help="Objective for tuning")
    return p

# =========================
# Main
# =========================
def main():
    args = build_argparser().parse_args()

    # Load model + meta
    model, scaler, meta = load_model_bundle(args.model_dir)
    feature_cols = list(meta.get("feature_cols", DEFAULT_FEATURE_COLS))
    window_size = int(meta.get("window_size", 150))
    buy_threshold = float(meta.get("buy_threshold", 0.60)) if args.threshold is None else float(args.threshold)
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = 0.005 if args.tp_pct is None else float(args.tp_pct)
    sl_pct = 0.0025 if args.sl_pct is None else float(args.sl_pct)

    # Load raw data and compute the SAME features as training
    df = read_csv_concat_sorted(args.data_dir)
    price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
    if price_col is None:
        raise SystemExit(f"Could not locate a price column. Available: {list(df.columns)}")
    df = compute_features(df)

    # Build feature matrix in the SAME order as meta (do NOT drop 'close')
    drop_cols = {"timestamp", "time"}  # only drop non-features
    feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
    missing = [c for c in feature_cols if c not in feat_cols]
    if missing:
        print(f"[WARN] Missing features in data (will drop): {missing}")
    if not feat_cols:
        raise SystemExit("No valid feature columns found in data for inference.")
    X_flat = df[feat_cols].to_numpy(dtype=np.float32)

    # Windows
    X = build_windows(X_flat, window_size)
    if len(X) == 0:
        raise SystemExit("Not enough rows to build any sequences. Increase data or reduce window size.")

    # Align OHLC arrays to window ends
    opens = df["open"].to_numpy(dtype=float)[window_size - 1:]
    highs = df["high"].to_numpy(dtype=float)[window_size - 1:]
    lows = df["low"].to_numpy(dtype=float)[window_size - 1:]
    closes = df[price_col].to_numpy(dtype=float)[window_size - 1:]

    # Scale using same scaler (fit on all meta features)
    if scaler is not None:
        n, t, f = X.shape
        X = scaler.transform(X.reshape(n * t, f)).reshape(n, t, f)

    # Choose device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Predict probabilities (auto-handles OOM)
    probs = predict_probs(model, X, int(args.batch_size), device)

    # Optional: Daily gating
    gate = None
    if int(args.use_daily_gate):
        daily_dir = Path(args.daily_model_dir)
        meta_path = daily_dir / 'daily_meta.json'
        if meta_path.exists():
            try:
                daily_meta = json.loads(meta_path.read_text())
                clf = joblib.load(daily_dir / daily_meta['paths']['model'])
                dscaler = joblib.load(daily_dir / daily_meta['paths']['scaler'])
                thr = float(daily_meta.get('threshold', 0.5))
                # Build daily features from raw df and map to minute windows
                raw = read_csv_concat_sorted(args.data_dir)
                if 'timestamp' in raw.columns:
                    ts = pd.to_datetime(raw['timestamp'], unit='ms')
                else:
                    ts = pd.to_datetime(raw.index)
                raw = raw.copy(); raw.index = ts
                agg = {
                    'open': 'first','high': 'max','low': 'min','close': 'last','volume': 'sum'
                }
                daily = raw.resample('1D').apply(agg).dropna()
                daily = daily[daily['volume'] > 0]
                # compute daily features mirroring train_daily
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
                tr = pd.concat([
                    (dfD['high'] - dfD['low']),
                    (dfD['high'] - dfD['close'].shift()).abs(),
                    (dfD['low'] - dfD['close'].shift()).abs(),
                ], axis=1).max(axis=1)
                dfD['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean().fillna(tr.mean())
                dfD['std_10'] = dfD['close'].rolling(10).std().bfill().fillna(0.0)
                dfD['vol_change'] = dfD['volume'].pct_change().replace([np.inf,-np.inf],0.0).fillna(0.0)
                dfD = dfD.dropna()
                feat_cols_daily = daily_meta.get('feature_cols', ['open','high','low','close','volume','ret_1','ema_5','ema_10','ema_20','sma_ratio','rsi_14','atr_14','std_10','vol_change'])
                Xd = dfD[feat_cols_daily].to_numpy(dtype=np.float32)
                Xds = dscaler.transform(Xd)
                pday = clf.predict_proba(Xds)[:,1]
                days = df.index.floor('D')
                days_end = days[window_size-1:]
                # map day -> gate bool
                dmap = {idx: (prob >= thr) for idx, prob in zip(dfD.index, pday)}
                gate = np.array([dmap.get(d, True) for d in days_end], dtype=bool)
                print(f"[DAILY] Gating enabled. True rate={gate.mean():.2%}")
            except Exception as e:
                print(f"[WARN] Daily gating disabled due to error: {e}")

    if args.mode == "simple":
        print(json.dumps({
            "metrics": {"n": int(len(probs))},
            "threshold": buy_threshold,
            "mean_prob": float(probs.mean()),
            "p90_prob": float(np.percentile(probs, 90)),
        }, indent=2))
        return

    # Optional threshold tuning for best return
    if int(args.tune_threshold):
        best_ret = -1e9
        best_t = buy_threshold
        best_trades = -1
        for t in np.linspace(max(0.05, buy_threshold - 0.2), min(0.99, buy_threshold + 0.2), 41):
            rpt, _ = simulate_trades_with_tp_sl(
                opens, highs, lows, closes, probs,
                threshold=float(t), start_capital=float(args.capital),
                fee_pct=fee_pct, tp_pct=tp_pct, sl_pct=sl_pct,
                low_stop_pct=float(args.low_stop_pct),
                reenter_at_same_price=bool(int(args.reenter_same)),
                confirm_bars=int(args.confirm_bars),
                signal_exit=bool(int(args.signal_exit)),
                exit_threshold=(float(args.exit_threshold) if args.exit_threshold is not None else float(t)*0.9),
            )
            port = rpt.get("portfolio", {})
            ret = port.get("return", -1e9)
            trades = int(port.get("trades", 0))
            # Enforce min trades if provided
            if trades < int(args.min_trades):
                continue
            if args.optimize_for == 'return':
                better = ret > best_ret
            elif args.optimize_for == 'trades':
                better = (trades > best_trades) or (trades == best_trades and ret > best_ret)
            else:  # ret_per_trade
                rpt_metric = (ret / max(1, trades))
                best_metric = (best_ret / max(1, best_trades)) if best_trades >= 0 else -1e9
                better = rpt_metric > best_metric
            if better:
                best_ret = ret; best_t = float(t); best_trades = trades
        buy_threshold = float(best_t)
        print(f"[TUNE] Best threshold={buy_threshold:.3f} return={best_ret*100:.2f}% trades={best_trades}")

    # Optional TP/SL tuning inside requested ranges
    def _parse_range(spec: str):
        try:
            parts = [float(x.strip()) for x in spec.split(',')]
            if len(parts) == 3:
                start, end, step = parts
                if step <= 0: step = (end - start) / 10.0 if end > start else 0.001
                vals = []
                v = start
                # ensure inclusive of end within epsilon
                while v <= end + 1e-12:
                    vals.append(round(v, 6))
                    v += step
                return vals
            elif len(parts) == 1:
                return [parts[0]]
        except Exception:
            pass
        return [0.02, 0.03, 0.04, 0.05]

    if int(args.tune_tpsl):
        tp_vals = _parse_range(args.tp_range)
        sl_vals = _parse_range(args.sl_range)
        best = {"ret": -1e9, "tp": tp_pct, "sl": sl_pct, "trades": -1}
        for tp in tp_vals:
            for sl in sl_vals:
                rpt, _ = simulate_trades_with_tp_sl(
                    opens, highs, lows, closes, probs,
                    threshold=buy_threshold,
                    start_capital=float(args.capital),
                    fee_pct=fee_pct, tp_pct=float(tp), sl_pct=float(sl),
                    low_stop_pct=float(args.low_stop_pct),
                    reenter_at_same_price=bool(int(args.reenter_same)),
                    confirm_bars=int(args.confirm_bars),
                    signal_exit=bool(int(args.signal_exit)),
                    exit_threshold=(float(args.exit_threshold) if args.exit_threshold is not None else float(buy_threshold)*0.9),
                )
                port = rpt.get("portfolio", {})
                ret = port.get("return", -1e9)
                trades = int(port.get("trades", 0))
                if trades < int(args.min_trades):
                    continue
                if args.optimize_for == 'return':
                    better = ret > best["ret"]
                elif args.optimize_for == 'trades':
                    better = (trades > best["trades"]) or (trades == best["trades"] and ret > best["ret"]) 
                else:
                    rpt_metric = (ret / max(1, trades))
                    best_metric = (best["ret"] / max(1, best["trades"])) if best["trades"] >= 0 else -1e9
                    better = rpt_metric > best_metric
                if better:
                    best.update({"ret": ret, "tp": float(tp), "sl": float(sl), "trades": trades})
        tp_pct, sl_pct = best["tp"], best["sl"]
        print(f"[TUNE] Best TP/SL in ranges: tp={tp_pct*100:.2f}%, sl={sl_pct*100:.2f}%, return={best['ret']*100:.2f}%, trades={best['trades']}")

    # Portfolio simulation with TP/SL + low-stop and reentry logic
    report, curve = simulate_trades_with_tp_sl(
        opens, highs, lows, closes, probs,
        threshold=buy_threshold,
        start_capital=float(args.capital),
        fee_pct=fee_pct, tp_pct=tp_pct, sl_pct=sl_pct,
        low_stop_pct=float(args.low_stop_pct),
        reenter_at_same_price=bool(int(args.reenter_same)),
        confirm_bars=int(args.confirm_bars),
        gate=gate,
    )
    print_portfolio_report(report, currency="$")

if __name__ == "__main__":
    main()
