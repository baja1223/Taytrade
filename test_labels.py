import pandas as pd
import numpy as np
import os

# --- 🔧 STEP 1: CONFIGURE YOUR TEST PARAMETERS HERE 🔧 ---
#
# Adjust these lists to test different values for profit threshold and risk-reward ratio.
#
profit_thresholds = [0.3, 0.5, 0.7, 1.0]  # in percent (e.g., 0.5 means 0.5%)
risk_ratios = [1.5, 2.0, 2.5, 3.0]        # risk-reward ratios
LOOKAHEAD_PERIOD = 10                     # number of future minutes to look ahead
DATA_FOLDER = "eth_1m_data_sample"        # folder containing sample CSV files
# ---

def find_first_csv(folder):
    """Finds the first .csv file in a given folder."""
    try:
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                return os.path.join(folder, file)
    except FileNotFoundError:
        return None
    return None

def analyze_and_tune_labels(df, profit_threshold_pct, risk_reward_ratio, lookahead_period):
    """
    Applies the labeling logic and returns a debug DataFrame and class balance.
    """
    print(f"\n🔬 Analyzing data with RR_Ratio={risk_reward_ratio} and Profit_Threshold={profit_threshold_pct}%...")

    # Calculate future highs and lows
    future_highs = df['high'].rolling(window=lookahead_period).max().shift(-lookahead_period)
    future_lows = df['low'].rolling(window=lookahead_period).min().shift(-lookahead_period)

    potential_profit = future_highs - df['close']
    potential_loss = df['close'] - future_lows

    # Dynamic profit threshold based on closing price
    profit_threshold_decimal = profit_threshold_pct / 100
    profit_threshold_dynamic = df['close'] * profit_threshold_decimal

    # Create labels based on conditions
    df['label'] = (
        (potential_profit > profit_threshold_dynamic) &
        (potential_loss > 0) &
        (potential_profit > risk_reward_ratio * potential_loss)
    ).astype(int)

    # Debug DataFrame for inspection
    debug_df = pd.DataFrame({
        'close': df['close'],
        'profit_threshold': profit_threshold_dynamic.round(4),
        'potential_profit': potential_profit.round(4),
        'potential_loss': potential_loss.round(4),
        'profit_check_pass': potential_profit > profit_threshold_dynamic,
        'rr_check_pass': potential_profit > risk_reward_ratio * potential_loss,
        'final_label': df['label']
    })

    # Class balance (0 = Hold, 1 = Buy)
    class_balance = df['label'].value_counts()

    return debug_df, class_balance

def main():
    """
    Main function to run the label tuning process.
    """
    print("--- Labeling Logic Investigation Script ---")

    sample_file = find_first_csv(DATA_FOLDER)
    if not sample_file:
        print(f"\n[ERROR] No .csv files found in the '{DATA_FOLDER}' folder.")
        print("Please create this folder and add at least one sample .csv file to it.")
        return

    print(f"Loading sample data from: {sample_file}")
    try:
        df = pd.read_csv(sample_file, header=None,
                         names=['date', 'open', 'high', 'low', 'close', 'volume'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
    except Exception as e:
        print(f"[ERROR] Could not load the CSV file: {e}")
        return

    results = []
    for pt in profit_thresholds:
        for rr in risk_ratios:
            print(f"\nTesting Profit Threshold: {pt}%, Risk-Reward Ratio: {rr}")
            debug_df, class_balance = analyze_and_tune_labels(
                df.copy(), pt, rr, LOOKAHEAD_PERIOD
            )
            buy_signals = class_balance.get(1, 0)
            hold_signals = class_balance.get(0, 0)
            total_signals = buy_signals + hold_signals
            buy_percentage = (buy_signals / total_signals) * 100 if total_signals > 0 else 0
            results.append({
                'profit_threshold': pt,
                'risk_reward_ratio': rr,
                'buy_signals': buy_signals,
                'hold_signals': hold_signals,
                'buy_percentage': buy_percentage
            })
            print(f"Buy Signals: {buy_signals}, Hold Signals: {hold_signals}, Buy Percentage: {buy_percentage:.2f}%")

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    print("\n--- Summary of Results ---")
    print(results_df)

    # Save results to CSV for further analysis
    results_df.to_csv('label_tuning_results.csv', index=False)
    print("Results saved to 'label_tuning_results.csv'")

if __name__ == "__main__":
    main()