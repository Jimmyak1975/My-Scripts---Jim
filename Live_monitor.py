import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from datetime import datetime, timedelta
from collections import deque

# ---------------- Configuration ----------------
SYMBOL = "TAOUSDT"  # Trading pair to monitor
ORDERBOOK_API_URL = "https://api.binance.com/api/v3/depth"
FETCH_INTERVAL = 5       # seconds between API calls (updated to 5 seconds)
LOOKBACK_SECONDS = 30    # window for the graph (in seconds)
TABLE_RECORD_SIZE = 10   # number of cumulative records to keep in the table
REGRESSION_ALPHA = 0.3   # Smoothing factor for regression calculations

# ---------------- Data Storage for Plot (Order Book) ----------------
times = deque()          # Timestamps for raw bid/ask data
bids = deque()           # Best bid prices
asks = deque()           # Best ask prices
bid_volumes = deque()    # Bid volumes
ask_volumes = deque()    # Ask volumes

# For the regression (cumulative imbalance from order book)
regression_values = deque(maxlen=LOOKBACK_SECONDS // FETCH_INTERVAL)
regression_times = deque(maxlen=LOOKBACK_SECONDS // FETCH_INTERVAL)
imbalance_record = deque(maxlen=TABLE_RECORD_SIZE)  # Column 1

# Global cumulative imbalance variable
cumulative_imbalance = 0

# ---------------- Data Storage for Extra Column ----------------
# For Cumulative Open Interest (%)
open_interest_values = deque(maxlen=LOOKBACK_SECONDS // FETCH_INTERVAL)
open_interest_record = deque(maxlen=TABLE_RECORD_SIZE)
cumulative_open_interest = 0
baseline_futures_oi = None

# Global variable for the regression line segments collection (plot)
reg_collection = None

# ---------------- Functions to Fetch Live Data ----------------
def fetch_order_book():
    """Fetch current order book data from Binance (spot)."""
    try:
        params = {"symbol": SYMBOL, "limit": 5}
        response = requests.get(ORDERBOOK_API_URL, params=params, timeout=5)
        data = response.json()
        best_bid_price = float(data["bids"][0][0])
        best_bid_volume = float(data["bids"][0][1])
        best_ask_price = float(data["asks"][0][0])
        best_ask_volume = float(data["asks"][0][1])
        return best_bid_price, best_ask_price, best_bid_volume, best_ask_volume
    except Exception as e:
        print(f"Error fetching order book data: {e}")
        return None, None, None, None

def fetch_futures_open_interest():
    """Fetch futures open interest from Binance Futures."""
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {"symbol": SYMBOL}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if "openInterest" not in data:
            print(f"openInterest not found in response: {data}")
            return None
        return float(data["openInterest"])
    except Exception as e:
        print(f"Error fetching futures OI data: {e}")
        return None

# ---------------- Update Data Function ----------------
def update_data():
    global cumulative_imbalance, baseline_futures_oi, cumulative_open_interest

    now = datetime.now()
    # Update Order Book Data (for plot)
    best_bid, best_ask, bid_vol, ask_vol = fetch_order_book()
    if best_bid is None or best_ask is None:
        return

    times.append(now)
    bids.append(best_bid)
    asks.append(best_ask)
    bid_volumes.append(bid_vol)
    ask_volumes.append(ask_vol)
    
    total_vol = bid_vol + ask_vol
    if total_vol > 0:
        imbalance = ((bid_vol - ask_vol) / total_vol) * 100
    else:
        imbalance = 0

    if regression_values:
        cumulative_imbalance = (REGRESSION_ALPHA * imbalance) + ((1 - REGRESSION_ALPHA) * regression_values[-1])
    else:
        cumulative_imbalance = imbalance
    regression_values.append(cumulative_imbalance)
    regression_times.append(now)
    imbalance_record.append(cumulative_imbalance)
    
    # Extra Metric: Futures Open Interest
    futures_oi = fetch_futures_open_interest()
    if futures_oi is not None:
        if baseline_futures_oi is None:
            baseline_futures_oi = futures_oi
        change_oi = ((futures_oi - baseline_futures_oi) / baseline_futures_oi) * 100
        if open_interest_values:
            cumulative_open_interest = (REGRESSION_ALPHA * change_oi) + ((1 - REGRESSION_ALPHA) * open_interest_values[-1])
        else:
            cumulative_open_interest = change_oi
        open_interest_values.append(cumulative_open_interest)
        open_interest_record.append(cumulative_open_interest)
    
    # Remove raw order book data older than the lookback window.
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
mid_line = ax1.axhline(0, color="black", linestyle="--", linewidth=2)

ax1.set_ylabel("Cumulative Imbalance (%)")
ax1.legend(loc="upper left")
ax1.set_title(f"Live Order Book for {SYMBOL}")
ax1.set_ylim(-100, 100)  # Plot remains untouched

def init():
    current_time = datetime.now()
    ax1.set_xlim(current_time - timedelta(seconds=LOOKBACK_SECONDS), current_time)
    ax2.axis('off')  # Hide table axis initially
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
    
    # Update regression line (as segments with varying color) for order book data.
    reg_times = list(regression_times)
    reg_values = list(regression_values)
    if len(reg_times) >= 2:
        x = mdates.date2num(reg_times)
        segments = []
        for i in range(len(x) - 1):
            segments.append([[x[i], reg_values[i]], [x[i+1], reg_values[i+1]]])
        colors = ["green" if reg_values[i+1] >= 0 else "red" for i in range(len(x) - 1)]
        if reg_collection is not None:
            reg_collection.remove()
        reg_collection = LineCollection(segments, colors=colors, linewidth=3)
        ax1.add_collection(reg_collection)
    else:
        if reg_collection is not None:
            reg_collection.remove()
            reg_collection = None

    ax1.set_ylim(-100, 100)
    mid_line.set_ydata([0, 0])
    
    # ---- Update the Table ----
    ax2.clear()
    ax2.axis('off')
    # Build table data with 2 columns:
    # Column 1: Cumulative Imbalance (%) (2-digit precision)
    # Column 2: Cumulative Open Interest (%) (4-digit precision)
    num_rows = len(imbalance_record)
    table_data = []
    for i in range(num_rows):
        row = [
            f"{imbalance_record[i]:.2f}%",
            f"{open_interest_record[i]:.4f}%" if i < len(open_interest_record) else ""
        ]
        table_data.append(row)
    
    col_labels = ["Cumulative Imbalance (%)", "Cumulative Open Interest (%)"]
    
    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8, 1.5)  # Compact the table
    
    # Color table text: green for positive, red for negative.
    for (row, col), cell in table.get_celld().items():
        if row > 0:  # Skip header row
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
