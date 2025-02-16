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
# User Parameters: Week-to-Week Pattern Analysis
# ======================================================
# Set the start date for the first week (must be a Monday, in "YYYY-MM-DD" format).
WEEK_START = "2024-01-01"  # Example: January 2, 2023 (a Monday)
# Number of weeks to analyze (e.g., ~54 weeks for roughly 1 year)
NUM_WEEKS = 54

# Local timezone: fixed offset of UTC+2.
local_tz = timezone(timedelta(hours=2))

# ======================================================
# Binance Futures API Settings (1-hour data)
# ======================================================
BINANCE_FUTURES_URL = "https://fapi.binance.com"
SYMBOL = "LTCUSDT"
INTERVAL = "1h"  # Use 1-hour candlesticks
LIMIT = 1500     # Max klines per API request

def get_historical_data(symbol, interval, start_time_ms, end_time_ms=None, limit=LIMIT):
    """
    Fetches historical kline data from Binance Futures between [start_time_ms, end_time_ms].
    Each kline is a list:
      [0] open_time (ms), [1] open, [2] high, [3] low, [4] close, [5] volume, ...
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
        print(f"Error fetching data: {e}")
        return []

def collect_data_for_period(symbol, interval, start_time_ms, end_time_ms):
    """
    Collects historical data strictly between start_time_ms and end_time_ms.
    To compute the first hour's percentage change, we subtract one hour from start_time_ms.
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
        print(f"Collected {len(data)} candles; next start = {current_start}")
        time.sleep(0.2)
        if current_start >= end_time_ms:
            break
    return all_data

def process_data_for_hourly_diff(raw_data, period_start_local):
    """
    1. Converts raw 1-hour kline data into a DataFrame.
    2. Converts Binance UTC timestamps to local time (UTC+2).
    3. Computes the hourly percentage change from the previous hour (across day boundaries).
    4. Filters out rows with open_time_local before period_start_local.
    5. Returns a DataFrame with columns: [date, day_of_week, hour, hourly_pct_change].
    """
    df = pd.DataFrame(raw_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    # Convert open_time to UTC datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    # Convert to local time (UTC+2)
    df["open_time_local"] = df["open_time"].dt.tz_convert(local_tz)
    # Convert numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Sort by local time
    df.sort_values(by="open_time_local", inplace=True)
    # Compute previous close (across entire dataset)
    df["prev_close"] = df["close"].shift(1)
    # Compute hourly percentage change
    df["hourly_pct_change"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100
    # Extract date, hour, day_of_week from local time
    df["date"] = df["open_time_local"].dt.date
    df["hour"] = df["open_time_local"].dt.hour
    df["day_of_week"] = df["open_time_local"].dt.day_name()
    # Filter: keep only rows from period_start_local onward
    df = df[df["open_time_local"] >= period_start_local]
    return df[["date", "day_of_week", "hour", "hourly_pct_change"]]

def analyze_patterns_hourly_diff(df_hourly, week_range_str):
    """
    1. Groups the data by (day_of_week, hour) to compute the average hourly_pct_change.
    2. Pivots the data so rows are hours (0-23) and columns are days (ordered Monday to Sunday).
    3. Ensures all 24 hours appear.
    4. Creates a heatmap with discrete color bins from -5% to +5% in 0.2% increments.
       Annotation text is colored green if positive, red if negative, and black if zero/NaN.
    """
    grouped = df_hourly.groupby(["day_of_week", "hour"])["hourly_pct_change"].mean().reset_index()
    pivot = grouped.pivot(index="hour", columns="day_of_week", values="hourly_pct_change")
    pivot = pivot.reindex(index=range(24))  # Ensure all hours 0..23 are present
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(columns=[d for d in days_order if d in pivot.columns])
    boundaries = np.arange(-5, 5.2, 0.2)
    cmap = plt.get_cmap("RdYlGn", len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N)
    plt.figure(figsize=(12, 8))
    plt.title(f"Weekly Heatmap: {week_range_str}\nHourly % Change from Previous Hour (Avg) - {SYMBOL} (UTC+2)")
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "% Change", "ticks": boundaries}
    )
    plt.xlabel("Day of Week")
    plt.ylabel("Hour of Day (0-23, UTC+2)")
    plt.xticks(rotation=45)
    plt.yticks(range(24), range(24))
    plt.tight_layout()
    filename = f"weekly_heatmap_{week_range_str.replace(' ', '_').replace(':','-')}.png"
    plt.savefig(filename)
    print(f"Saved heatmap: {filename}")
    plt.close()
    print("\nPivot Table:")
    print(pivot)

if __name__ == "__main__":
    # Initialize starting week (must be a Monday)
    current_week_start = pd.to_datetime(WEEK_START).tz_localize(local_tz)
    for i in range(NUM_WEEKS):
        # Define the week's start and end: Monday 00:00 to Sunday 23:59:59
        week_start_local = current_week_start
        week_end_local = week_start_local + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
        week_range_str = f"{week_start_local.strftime('%Y-%m-%d %H:%M:%S')} to {week_end_local.strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"\nAnalyzing Week {i+1}: {week_range_str}")
        start_ms = int(week_start_local.timestamp() * 1000)
        end_ms = int(week_end_local.timestamp() * 1000)
        raw_data = collect_data_for_period(SYMBOL, INTERVAL, start_ms, end_ms)
        if raw_data:
            df_hourly = process_data_for_hourly_diff(raw_data, week_start_local)
            analyze_patterns_hourly_diff(df_hourly, week_range_str)
        else:
            print("No data collected for this week.")
        # Move to next Monday
        current_week_start = current_week_start + pd.Timedelta(weeks=1)
