import os
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from binance.client import Client
from dotenv import load_dotenv
from utils import SignalGenerator

# --- Setup ---
load_dotenv()
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)
client = Client(os.getenv('BINANCE_KEY'), os.getenv('BINANCE_SECRET'))

# --- Live data buffers ---
price_data = deque(maxlen=300)
time_data = deque(maxlen=300)
confidence_data = deque(maxlen=300)
signal_data = deque(maxlen=300)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))

def update(_):
    try:
        latest = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
        kline_data = {
            'date': pd.to_datetime(latest[0], unit='ms'),
            'open': float(latest[1]),
            'high': float(latest[2]),
            'low': float(latest[3]),
            'close': float(latest[4]),
            'volume': float(latest[5])
        }
        close_price = kline_data['close']
        current_time = kline_data['date']

        result = signal_gen.get_signal(kline_data)
        confidence = result.get('confidence', 0)
        signal = result.get('signal', 0)

        price_data.append(close_price)
        time_data.append(current_time)
        confidence_data.append(confidence)
        signal_data.append(signal)

        ax1.clear()
        ax2.clear()

        # Price
        ax1.plot(time_data, price_data, label='ETH Price', color='deepskyblue')
        ax1.set_ylabel("Price (USDT)")
        ax1.set_title("Live ETH/USDT with Signals and Confidence")
        ax1.grid(True)

        # Signal markers
        for i, sig in enumerate(signal_data):
            if sig == 1:
                ax1.scatter(time_data[i], price_data[i], color='lime', marker='^', s=100, label='BUY' if i == 0 else "")
            elif sig == -1:
                ax1.scatter(time_data[i], price_data[i], color='red', marker='v', s=100, label='SELL' if i == 0 else "")
            else:
                ax1.scatter(time_data[i], price_data[i], color='gray', marker='o', s=30, label='HOLD' if i == 0 else "")

        # Confidence
        ax2.plot(time_data, confidence_data, label='Confidence')
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Time")
        ax2.grid(True)

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # Console print (fixed)
        print(f"\r⏱ {current_time} | 💰 Price: ${close_price:.2f} | 🤖 Conf: {confidence if confidence else 0:.3f} | Signal: {signal}", end='')

    except Exception as e:
        print(f"\n❌ Error in update loop: {e}")

    return ax1.lines + ax2.lines

ani = animation.FuncAnimation(fig, update, interval=1000, blit=False)
plt.tight_layout()
plt.show()