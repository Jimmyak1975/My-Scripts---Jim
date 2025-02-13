import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from datetime import datetime, timedelta
from collections import deque

# ---------------- Configuration ----------------
SYMBOL = "TAOUSDT"  # Trading pair to monitor
API_URL = "https://api.binance.com/api/v3/depth"
FETCH_INTERVAL = 3       # seconds between API calls
LOOKBACK_SECONDS = 30    # window for the graph (in seconds)
TABLE_RECORD_SIZE = 10   # number of cumulative imbalance records to keep
REGRESSION_ALPHA = 0.3   # Smoothing factor for the regression (cumulative imbalance)

# ---------------- Data Storage ----------------
times = deque()          # For raw bid/ask chart (time stamps)
bids = deque()           # Best bid prices
asks = deque()           # Best ask prices
bid_volumes = deque()    # Bid volumes
ask_volumes = deque()    # Ask volumes

# For the regression (cumulative imbalance trend)
regression_values = deque(maxlen=LOOKBACK_SECONDS // FETCH_INTERVAL)
regression_times = deque(maxlen=LOOKBACK_SECONDS // FETCH_INTERVAL)

# For the table (stores the last TABLE_RECORD_SIZE cumulative values)
imbalance_record = deque(maxlen=TABLE_RECORD_SIZE)

# A variable to store the cumulative imbalance value.
cumulative_imbalance = 0

# Global variable for the regression line collection.
reg_collection = None

# ---------------- Data Fetching ----------------
def fetch_order_book():
    """Fetch current order book data from Binance for the given symbol."""
    try:
        params = {"symbol": SYMBOL, "limit": 5}
        response = requests.get(API_URL, params=params, timeout=5)
        data = response.json()
        best_bid_price = float(data["bids"][0][0])
        best_bid_volume = float(data["bids"][0][1])
        best_ask_price = float(data["asks"][0][0])
        best_ask_volume = float(data["asks"][0][1])
        return best_bid_price, best_ask_price, best_bid_volume, best_ask_volume
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None, None

def update_data():
    global cumulative_imbalance
    now = datetime.now()
    best_bid, best_ask, bid_vol, ask_vol = fetch_order_book()
    if best_bid is None or best_ask is None:
        return

    # Append new price and volume data for the raw chart.
    times.append(now)
    bids.append(best_bid)
    asks.append(best_ask)
    bid_volumes.append(bid_vol)
    ask_volumes.append(ask_vol)
    
    # Compute imbalance percentage using:
    # ((bid_vol - ask_vol) / (bid_vol + ask_vol)) * 100
    total_vol = bid_vol + ask_vol
    if total_vol > 0:
        imbalance = ((bid_vol - ask_vol) / total_vol) * 100
    else:
        imbalance = 0

    # --- Cumulative (regression) calculation ---
    if regression_values:
        cumulative_imbalance = (REGRESSION_ALPHA * imbalance) + ((1 - REGRESSION_ALPHA) * regression_values[-1])
    else:
        cumulative_imbalance = imbalance  # first value
    
    regression_values.append(cumulative_imbalance)
    regression_times.append(now)
    imbalance_record.append(cumulative_imbalance)
    
    # Remove raw data older than the lookback window.
    cutoff = now - timedelta(seconds=LOOKBACK_SECONDS)
    while times and times[0] < cutoff:
        times.popleft()
        bids.popleft()
        asks.popleft()
        bid_volumes.popleft()
        ask_volumes.popleft()

# ---------------- Plot Setup ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

# Top plot: Raw bid/ask chart.
line_bid, = ax1.plot([], [], label="Best Bid", color="green", marker="o")
line_ask, = ax1.plot([], [], label="Best Ask", color="red", marker="o")
# Regression line will be drawn via a LineCollection.
# Middle horizontal line (fixed at y = 0)
mid_line = ax1.axhline(0, color="black", linestyle="--", linewidth=2)

ax1.set_ylabel("Cumulative Imbalance (%)")
ax1.legend(loc="upper left")
ax1.set_title(f"Live Order Book for {SYMBOL}")

def init():
    current_time = datetime.now()
    ax1.set_xlim(current_time - timedelta(seconds=LOOKBACK_SECONDS), current_time)
    # Set fixed y-axis boundaries from -100 to 100.
    ax1.set_ylim(-100, 100)
    ax2.axis('off')  # Hide ax2 axes (for the table)
    return line_bid, line_ask, mid_line

def animate(frame):
    global reg_collection
    update_data()
    if not times:
        return line_bid, line_ask, mid_line

    current_time = datetime.now()
    ax1.set_xlim(current_time - timedelta(seconds=LOOKBACK_SECONDS), current_time)
    
    # Update raw bid/ask chart.
    times_list = list(times)
    line_bid.set_data(times_list, list(bids))
    line_ask.set_data(times_list, list(asks))
    
    # Update regression line (as segments with varying color)
    reg_times = list(regression_times)
    reg_values = list(regression_values)
    if len(reg_times) >= 2:
        x = mdates.date2num(reg_times)
        segments = []
        for i in range(len(x) - 1):
            seg = [[x[i], reg_values[i]], [x[i+1], reg_values[i+1]]]
            segments.append(seg)
        # Each segment's color is based on the value of the new point.
        colors = ["green" if reg_values[i+1] >= 0 else "red" for i in range(len(x) - 1)]
        if reg_collection is not None:
            reg_collection.remove()
        reg_collection = LineCollection(segments, colors=colors, linewidth=3)
        ax1.add_collection(reg_collection)
    else:
        if reg_collection is not None:
            reg_collection.remove()
            reg_collection = None

    # Set the y-axis fixed to [-100, 100] (so zero is in the middle)
    ax1.set_ylim(-100, 100)
    mid_line.set_ydata([0, 0])
    
    # ---- Update the Table (compact, centered numbers) ----
    ax2.clear()
    ax2.axis('off')
    table_data = [[f"{imb:.2f}%"] for imb in imbalance_record]
    table = ax2.table(cellText=table_data, colLabels=["Cumulative Imbalance (%)"],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8, 1.5)  # Compact the table
    
    # Adjust text color: green for positive, red for negative.
    for (row, col), cell in table.get_celld().items():
        if row > 0:  # skip header
            try:
                value = float(cell.get_text().get_text().replace("%", ""))
            except Exception:
                value = 0
            cell.get_text().set_color("green" if value >= 0 else "red")
            cell.set_facecolor("none")
    
    return line_bid, line_ask, reg_collection, mid_line

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              interval=FETCH_INTERVAL * 1000, blit=False,
                              cache_frame_data=False)

plt.show()
