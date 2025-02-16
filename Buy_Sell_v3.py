import requests
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from datetime import timezone, timedelta

# ======================================================
# User-Defined Year Period (1-Year Data)
# ======================================================
# Specify the start date for the 1-year period in "YYYY-MM-DD" format.
YEAR_START = "2024-02-05"  # Example: January 1, 2023
# Compute the end date as 364 days later (to cover 365 days total)
start_dt = pd.to_datetime(YEAR_START)
YEAR_END = (start_dt + pd.Timedelta(days=365-1)).strftime("%Y-%m-%d")
print(f"Analyzing year from {YEAR_START} to {YEAR_END}")

# Convert YEAR_START and YEAR_END to local timestamps (UTC+2)
local_tz = timezone(timedelta(hours=2))
year_start_local = pd.to_datetime(YEAR_START).tz_localize(local_tz)
start_time_ms = int(year_start_local.timestamp() * 1000)
# For YEAR_END, cover the entire day (set time to 23:59:59)
year_end_local = pd.to_datetime(YEAR_END, format="%Y-%m-%d").tz_localize(local_tz) + pd.Timedelta(hours=23, minutes=59, seconds=59)
end_time_ms = int(year_end_local.timestamp() * 1000)

# ======================================================
# Binance Futures API Settings (1-Hour Data)
# ======================================================
BINANCE_FUTURES_URL = "https://fapi.binance.com"
SYMBOL = "LTCUSDT"
INTERVAL = "1h"  # 1-hour candlesticks
LIMIT = 1500     # Max klines per API request

def get_historical_data(symbol, interval, start_time_ms, end_time_ms=None, limit=LIMIT):
    """
    Fetches historical kline data from Binance Futures starting at start_time_ms.
    Optionally, an end_time_ms can be specified.
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
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return []

def collect_data_for_period(symbol, interval, start_time_ms, end_time_ms):
    """
    Collects historical data for the given symbol and interval between start_time_ms and end_time_ms.
    To compute the first hour's change, we fetch one extra candle by subtracting one interval (1h) from start_time_ms.
    """
    extra_ms = 3600 * 1000  # 1 hour in ms
    query_start = start_time_ms - extra_ms
    all_data = []
    current_start = query_start

    while current_start < end_time_ms:
        data = get_historical_data(symbol, interval, current_start, end_time_ms=end_time_ms)
        if not data:
            break
        all_data.extend(data)
        current_start = int(data[-1][0]) + 1
        print(f"Collected {len(data)} candles; moving to next batch...")
        time.sleep(0.2)
        if current_start >= end_time_ms:
            break
    return all_data

def process_data_for_hourly_diff(raw_data):
    """
    1. Converts raw 1-hour kline data into a DataFrame.
    2. Converts Binance UTC timestamps to local time (UTC+2).
    3. Computes % change in 'close' from the previous hour (across day boundaries).
    4. Filters out rows with open_time_local before year_start_local.
    5. Returns a DataFrame with columns: [date, day_of_week, hour, hourly_pct_change].
    """
    df = pd.DataFrame(raw_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    # Convert open_time to datetime (UTC)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', utc=True)
    # Convert to local time (UTC+2)
    df["open_time_local"] = df["open_time"].dt.tz_convert(local_tz)
    
    # Debug: print earliest and latest local times in the fetched data
    print("Earliest local open time in data:", df["open_time_local"].min())
    print("Latest local open time in data:", df["open_time_local"].max())
    
    # Convert numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sort by local open time
    df.sort_values(by="open_time_local", inplace=True)
    
    # Compute previous close across entire dataset (even across day boundaries)
    df["prev_close"] = df["close"].shift(1)
    
    # Compute hourly % change from previous hour
    df["hourly_pct_change"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100
    
    # Extract date, hour, and day_of_week from local time
    df["date"] = df["open_time_local"].dt.date
    df["hour"] = df["open_time_local"].dt.hour
    df["day_of_week"] = df["open_time_local"].dt.day_name()
    
    # Filter out rows with open_time_local before year_start_local
    df = df[df["open_time_local"] >= year_start_local]
    
    return df[["date", "day_of_week", "hour", "hourly_pct_change"]]

def analyze_patterns_hourly_diff(df_hourly):
    """
    1. Groups data by (day_of_week, hour) to compute the average hourly_pct_change over the year.
    2. Pivots the data so rows = hours (0-23) and columns = days (ordered Monday to Sunday).
    3. Ensures all 24 hours appear in the pivot.
    4. Creates a heatmap with a discrete color scale from -5% to +5% in 0.2% increments.
       Annotation text is colored green if positive, red if negative, and black if zero/NaN.
    """
    grouped = df_hourly.groupby(["day_of_week", "hour"])["hourly_pct_change"].mean().reset_index()
    pivot = grouped.pivot(index="hour", columns="day_of_week", values="hourly_pct_change")
    
    # Ensure all 24 hours (0-23) appear
    pivot = pivot.reindex(index=range(24))
    
    # Order columns from Monday to Sunday
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(columns=[d for d in days_order if d in pivot.columns])
    
    # Define discrete boundaries every 0.2% from -5% to +5%
    boundaries = np.arange(-5, 5.2, 0.2)
    cmap = plt.get_cmap("RdYlGn", len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N)
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Hourly % Change from Previous Hour (Avg) - {SYMBOL} (Local Time UTC+2)\nYear: {YEAR_START} to {YEAR_END}")
    
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
    # Collect 1-year data (365 days)
    raw_data = collect_data_for_period(SYMBOL, INTERVAL, start_time_ms, end_time_ms)
    
    if raw_data:
        df_hourly = process_data_for_hourly_diff(raw_data)
        print("\nSample of hourly data with changes from previous hour:")
        print(df_hourly.head(10))
        analyze_patterns_hourly_diff(df_hourly)
    else:
        print("No data collected. Check your API or network settings.")
