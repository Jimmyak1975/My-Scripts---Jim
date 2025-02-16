import requests
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from datetime import timezone, timedelta

# =========================================
# Binance Futures API Settings (1-hour data)
# =========================================
BINANCE_FUTURES_URL = "https://fapi.binance.com"
SYMBOL = "LTCUSDT"
INTERVAL = "1h"  # Use 1-hour candlesticks directly
LIMIT = 1500     # Max klines per API request

def get_historical_data(symbol, interval, start_time_ms, limit=LIMIT):
    """
    Fetches historical kline data from Binance Futures starting at start_time_ms.
    Each kline is a list:
      [0] open_time, [1] open, [2] high, [3] low, [4] close, [5] volume, ...
    """
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return []

def collect_data(symbol, interval, days=365):
    """
    Collects historical data for 'symbol' and 'interval' over 'days' days.
    For 1-year lookback, days=365.
    """
    end_time = int(time.time() * 1000)  # current time in ms
    start_time = end_time - days * 24 * 3600 * 1000
    all_data = []
    current_start = start_time

    while current_start < end_time:
        data = get_historical_data(symbol, interval, current_start)
        if not data:
            break
        all_data.extend(data)
        # Advance current_start to the time of the last candle + 1 ms
        current_start = int(data[-1][0]) + 1
        print(f"Collected {len(data)} candles; moving to next batch...")
        time.sleep(0.2)  # pause to respect API limits

    return all_data

def process_data_for_hourly_diff(raw_data):
    """
    1. Converts raw 1-hour kline data into a DataFrame.
    2. Converts the Binance UTC timestamps to local time (UTC+2).
    3. Computes % change in 'close' from the *previous hour*, including across day boundaries.
    4. Returns a DataFrame with columns: [date, day_of_week, hour, hourly_pct_change].
    """
    df = pd.DataFrame(raw_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    # Convert open_time to datetime (UTC)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', utc=True)
    # Convert to local time (UTC+2)
    local_tz = timezone(timedelta(hours=2))
    df["open_time_local"] = df["open_time"].dt.tz_convert(local_tz)
    
    # Debug: print earliest and latest local times
    print("Earliest local open time:", df["open_time_local"].min())
    print("Latest local open time:", df["open_time_local"].max())
    
    # Convert numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sort by local open time
    df.sort_values(by="open_time_local", inplace=True)
    
    # For each row, use the previous row's close (across day boundaries)
    df["prev_close"] = df["close"].shift(1)
    
    # Compute hourly % change from the previous hour
    df["hourly_pct_change"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100
    
    # Extract date, hour, and day_of_week from local time
    df["date"] = df["open_time_local"].dt.date
    df["hour"] = df["open_time_local"].dt.hour
    df["day_of_week"] = df["open_time_local"].dt.day_name()
    
    return df[["date", "day_of_week", "hour", "hourly_pct_change"]]

def analyze_patterns_hourly_diff(df_hourly):
    """
    1. Groups data by (day_of_week, hour) to compute the average hourly_pct_change over the lookback period.
    2. Pivots the data so rows = hours (0-23) and columns = days (ordered Monday to Sunday).
    3. Ensures all 24 hours appear, even if some are missing.
    4. Creates a heatmap with a discrete color scale from -5% to +5% in 0.2% increments.
       Annotation text is colored green if positive, red if negative, and black if zero or NaN.
    """
    grouped = df_hourly.groupby(["day_of_week", "hour"])["hourly_pct_change"].mean().reset_index()
    pivot = grouped.pivot(index="hour", columns="day_of_week", values="hourly_pct_change")
    
    # Ensure all 24 hours (0..23) appear
    pivot = pivot.reindex(index=range(24))
    
    # Order the columns from Monday to Sunday
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(columns=[d for d in days_order if d in pivot.columns])
    
    # Define discrete boundaries every 0.2% from -5% to +5%
    boundaries = np.arange(-5, 5.2, 0.2)  # e.g., -5.0, -4.8, ..., 4.8, 5.0
    cmap = plt.get_cmap("RdYlGn", len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N)
    
    plt.figure(figsize=(12, 8))
    plt.title("Hourly % Change from Previous Hour (Avg) - LTCUSDT (Local Time UTC+2)\nLookback: 1 Year")
    
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "% Change", "ticks": boundaries}
    )
    
    plt.xlabel("Day of Week")
    plt.ylabel("Hour of Day (Local UTC+2)")
    plt.xticks(rotation=45)
    plt.yticks(range(24), range(24))
    plt.tight_layout()
    
    # Color annotation text: green if > 0, red if < 0, black if 0 or NaN.
    for text_obj in ax.texts:
        try:
            val = float(text_obj.get_text())
            if np.isnan(val):
                text_obj.set_color("black")
            elif val < 0:
                text_obj.set_color("red")
            elif val > 0:
                text_obj.set_color("green")
            else:
                text_obj.set_color("black")
        except ValueError:
            text_obj.set_color("black")
    
    plt.show()
    
    print("\nPivot Table of Hourly % Change (Average):")
    print(pivot)

if __name__ == "__main__":
    # Lookback period set to 1 year
    days_to_collect = 365
    raw_data = collect_data(SYMBOL, INTERVAL, days=days_to_collect)
    
    if raw_data:
        df_hourly = process_data_for_hourly_diff(raw_data)
        print("\nSample of hourly data with changes from previous hour:")
        print(df_hourly.head(10))
        analyze_patterns_hourly_diff(df_hourly)
    else:
        print("No data collected. Check your API or network settings.")
