import os
import time
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException

from utils import SignalGenerator

# ---------------------------------------
load_dotenv()

SYMBOL = os.getenv("TRADE_SYMBOL", "ETHUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
QUOTE_USDT = float(os.getenv("TRADE_QUANTITY_USDT", "15"))

API_KEY = os.getenv("BINANCE_KEY") or os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_SECRET") or os.getenv("BINANCE_TESTNET_SECRET")
def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

TESTNET = env_bool("TESTNET", False)
DRY_RUN = env_bool("DRY_RUN", True) 
def get_client() -> Client:
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing BINANCE credentials in env")
    return Client(API_KEY, API_SECRET, testnet=TESTNET)

def to_row(k) -> Dict[str, Any]:
    return {"date": k[0], "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}

def prefill(sig: SignalGenerator, client: Client) -> None:
    kl = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=min(sig.history_size, 1000))
    for k in kl: sig.history.append(to_row(k))

def latest(client: Client) -> Tuple[Dict[str, Any], int]:
    kl = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=2)
    last = to_row(kl[-1])
    return last, last["date"]

def place_market_quote_order(client: Client, side: str, symbol: str, quote_qty: float) -> Dict[str, Any]:
    if DRY_RUN:
        print(f"[DRY RUN] {side} {symbol} for {quote_qty} USDT")
        return {"status": "dry", "side": side, "symbol": symbol, "quoteOrderQty": quote_qty}
    try:
        return client.create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quoteOrderQty=str(quote_qty),
        )
    except BinanceAPIException as e:
        raise RuntimeError(f"Binance order failed: {e.status_code} {e.message}")

def trade_live(poll_seconds: int = 1) -> None:
    if not ENDPOINT_NAME:
        raise RuntimeError("ENDPOINT_NAME is not set")

    client = get_client()
    sig = SignalGenerator(endpoint_name=ENDPOINT_NAME)
    prefill(sig, client)

    position_open = False
    last_open_time = -1

    print(f"Live trading started on {SYMBOL} | DRY_RUN={DRY_RUN} TESTNET={TESTNET}")

    while True:
        try:
            bar, open_time = latest(client)
            if open_time == last_open_time:
                time.sleep(poll_seconds); 
                continue
            last_open_time = open_time

            res = sig.get_signal(bar)
            conf = res.get("confidence")
            signal = res.get("signal", 0)
            price = bar["close"]

            if not position_open and signal == 1 and conf is not None and conf >= sig.threshold:
                order = place_market_quote_order(client, SIDE_BUY, SYMBOL, QUOTE_USDT)
                print(f"🚀 BUY @ {price:.2f} | conf={conf:.3f} -> {order.get('status', 'ok')}")
                position_open = True
            elif position_open and signal == 0:
                order = place_market_quote_order(client, SIDE_SELL, SYMBOL, QUOTE_USDT)
                print(f"🏁 SELL @ {price:.2f} -> {order.get('status', 'ok')}")
                position_open = False

        except Exception as e:
            print(f"Error in trade loop: {e}")
            time.sleep(2)

        time.sleep(poll_seconds)

if __name__ == "__main__":
    trade_live()