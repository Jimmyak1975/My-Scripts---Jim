import requests
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timezone, timedelta

# ======================================================
# Lookback Period Settings: Past 54 Weeks (~1 year)
# ======================================================
LOOKBACK_WEEKS = 54
local_tz = timezone(timedelta(hours=2))

# Set end date as today (local midnight) and start date 54 weeks ago
end_dt = pd.Timestamp.now(tz=local_tz).normalize()  # Today at midnight (local)
start_dt = end_dt - pd.Timedelta(weeks=LOOKBACK_WEEKS)
LOOKBACK_START = start_dt.strftime("%Y-%m-%d")
LOOKBACK_END = end_dt.strftime("%Y-%m-%d")
print(f"Analyzing data from {LOOKBACK_START} to {LOOKBACK_END}")

# Convert to timestamps in milliseconds:
# For start, use midnight; for end, use 23:59:59 to cover the full day.
week_start_local = start_dt
start_time_ms = int(week_start_local.timestamp() * 1000)
week_end_local = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
end_time_ms = int(week_end_local.timestamp() * 1000)

# ======================================================
# Binance Futures API Settings (1-hour candlesticks)
# ======================================================
BINANCE_FUTURES_URL = "https://fapi.binance.com"
SYMBOL = "LTCUSDT"
INTERVAL = "1h"  # 1-hour candlesticks
LIMIT = 1500     # Maximum candles per API request

def get_historical_data(symbol, interval, start_time_ms, end_time_ms=None, limit=LIMIT):
    """
    Fetches historical kline data from Binance Futures.
    Each kline is a list: [open_time, open, high, low, close, volume, ...]
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
    Collects historical data from Binance for the specified period.
    An extra candle before the start is fetched to allow computation of the first hourly change.
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
        current_start = int(data[-1][0]) + 1  # move to next batch
        print(f"Collected {len(data)} candles; moving to next batch...")
        time.sleep(0.2)
        if current_start >= end_time_ms:
            break
    return all_data

def process_data_for_hourly_diff(raw_data):
    """
    Processes raw kline data into a DataFrame:
      - Converts timestamps from ms to datetime (local UTC+2).
      - Computes hourly % change (using previous candle's close).
      - Extracts date, hour, and day_of_week.
      - Removes the extra candle used for calculation.
    Returns a DataFrame with columns: [date, day_of_week, hour, hourly_pct_change].
    """
    df = pd.DataFrame(raw_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    # Convert open_time to UTC datetime and then to local (UTC+2)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', utc=True)
    df["open_time_local"] = df["open_time"].dt.tz_convert(local_tz)
    
    # Convert numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.sort_values(by="open_time_local", inplace=True)
    df["prev_close"] = df["close"].shift(1)
    df["hourly_pct_change"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100
    
    # Extract date, hour, and day_of_week
    df["date"] = df["open_time_local"].dt.date
    df["hour"] = df["open_time_local"].dt.hour
    df["day_of_week"] = df["open_time_local"].dt.day_name()
    
    # Filter out rows before our lookback start (the extra candle is dropped)
    df = df[df["open_time_local"] >= week_start_local]
    
    return df[["date", "day_of_week", "hour", "hourly_pct_change"]]

def analyze_hourly_patterns_yearly(df_hourly, min_count=5, threshold=60):
    """
    Analyzes hourly patterns over the past year.
    For each combination of day_of_week and hour:
      - Counts total observations, bullish (pct_change > 0), and bearish (pct_change < 0).
      - Computes the bullish and bearish ratios.
      - If bullish ratio >= threshold and sufficient samples, marks with a green up-arrow (▲) and percentage.
      - If bearish ratio >= threshold and sufficient samples, marks with a red down-arrow (▼) and percentage.
      - Otherwise, leaves the cell blank.
    Returns two DataFrames:
      - pivot_table: DataFrame with formatted text for display.
      - color_table: DataFrame with corresponding cell background colors.
    """
    grouped = df_hourly.groupby(["day_of_week", "hour"])["hourly_pct_change"].agg(
        total="count",
        bullish=lambda x: (x > 0).sum(),
        bearish=lambda x: (x < 0).sum()
    ).reset_index()
    grouped["bullish_ratio"] = grouped["bullish"] / grouped["total"] * 100
    grouped["bearish_ratio"] = grouped["bearish"] / grouped["total"] * 100

    # Debug: print the grouped data
    print("Grouped data by day and hour:")
    print(grouped)
    
    # Prepare empty DataFrames for the pivot table (formatted text) and colors
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_table = pd.DataFrame("", index=range(24), columns=days_order)
    color_table = pd.DataFrame("", index=range(24), columns=days_order)
    
    # Define colors for patterns and empty cells:
    bullish_color = "#90ee90"  # light green
    bearish_color = "#ffcccb"  # light red
    empty_color = "#d3d3d3"    # light gray
    
    # Fill in the pivot and color tables
    for _, row in grouped.iterrows():
        day = row["day_of_week"]
        hr = int(row["hour"])
        total = row["total"]
        bull_ratio = row["bullish_ratio"]
        bear_ratio = row["bearish_ratio"]

        cell_text = ""
        cell_color = empty_color

        if total >= min_count:
            if bull_ratio >= threshold:
                cell_text = f"▲ {bull_ratio:.0f}%"
                cell_color = bullish_color
            elif bear_ratio >= threshold:
                cell_text = f"▼ {bear_ratio:.0f}%"
                cell_color = bearish_color

        if day in pivot_table.columns:
            pivot_table.at[hr, day] = cell_text
            color_table.at[hr, day] = cell_color

    return pivot_table, color_table

def display_pattern_table(pivot_table, color_table):
    """
    Displays the pivot table of hourly patterns using matplotlib's table.
    The table uses a dark background with colored cells.
    """
    days = list(pivot_table.columns)
    hours = list(pivot_table.index)
    
    # Create a figure with a dark background
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#2f2f2f")  # dark background
    ax.axis('off')
    
    # Prepare cell text as list-of-lists
    table_data = pivot_table.values.tolist()
    
    # Create the table
    the_table = ax.table(
        cellText=table_data,
        rowLabels=[str(hr) for hr in hours],
        colLabels=days,
        loc='center',
        cellLoc='center'
    )
    
    # Adjust table properties: font size, colors, etc.
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    
    # Set header colors and cell colors based on color_table
    for (row, col), cell in the_table.get_celld().items():
        # Header cells: first row or first column
        if row == 0 or col < 0:
            cell.set_text_props(color="white", weight="bold")
            cell.set_facecolor("#4f4f4f")
        else:
            # Determine the corresponding hour and day
            hour = hours[row - 1]
            day = days[col]
            cell_color = color_table.at[hour, day]
            cell.set_facecolor(cell_color)
            cell.set_text_props(color="black")
    
    ax.set_title(f"Hourly Patterns for {SYMBOL}\nLookback: {LOOKBACK_START} to {LOOKBACK_END}", color="white", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    print("\nPattern Table (Text Version):")
    print(pivot_table)

if __name__ == "__main__":
    # Step 1: Collect raw historical data over the past 54 weeks
    raw_data = collect_data_for_period(SYMBOL, INTERVAL, start_time_ms, end_time_ms)
    
    if raw_data:
        # Step 2: Process raw data to compute hourly % change
        df_hourly = process_data_for_hourly_diff(raw_data)
        print("\nSample of processed hourly data:")
        print(df_hourly.head(10))
        
        # Step 3: Analyze patterns across days of week and hours
        # Adjust min_count and threshold if necessary.
        pivot_table, color_table = analyze_hourly_patterns_yearly(df_hourly, min_count=5, threshold=60)
        
        # Step 4: Display the table with colored cells
        display_pattern_table(pivot_table, color_table)
    else:
        print("No data collected. Please check your API/network settings.")
