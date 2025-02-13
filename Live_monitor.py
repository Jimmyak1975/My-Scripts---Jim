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
FETCH_INTERVAL = 2       # seconds between API calls
LOOKBACK_SECONDS = 30    # base window for the graph (in seconds)
TABLE_RECORD_SIZE = 10   # number of cumulative records to keep in the table

# For the HA-smoothed values, we want enough data points to yield at least 10 segments.
# (Segments count = number of points - 1)
HA_MAX_POINTS = 16       # This will allow up to 15 segments on the HA plot

# ---------------- Global Variables for Heikin Ashi Smoothing ----------------
# For cumulative imbalance (Heikin Ashi components)
ha_imbalance_open = None
ha_imbalance_close = None
# For futures open interest (Heikin Ashi components)
ha_oi_open = None
ha_oi_close = None

# Deques for plotting Heikin Ashi–smoothed values.
ha_imbalance_values = deque(maxlen=HA_MAX_POINTS)
ha_imbalance_times = deque(maxlen=HA_MAX_POINTS)
imbalance_record = deque(maxlen=TABLE_RECORD_SIZE)  # For table display

# For futures open interest (%)
open_interest_values = deque(maxlen=LOOKBACK_SECONDS // FETCH_INTERVAL)
open_interest_record = deque(maxlen=TABLE_RECORD_SIZE)
baseline_futures_oi = None

# Global variable for the Heikin Ashi line segments collection (plot)
reg_collection = None

# ---------------- Data Storage for Raw Order Book Plot ----------------
# These deques hold the raw bid/ask data.
times = deque()
bids = deque()
asks = deque()
bid_volumes = deque()
ask_volumes = deque()

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
    global baseline_futures_oi, ha_imbalance_open, ha_imbalance_close, ha_oi_open, ha_oi_close

    now = datetime.now()
    # Update Order Book Data (for raw plot)
    best_bid, best_ask, bid_vol, ask_vol = fetch_order_book()
    if best_bid is None or best_ask is None:
        return

    times.append(now)
    bids.append(best_bid)
    asks.append(best_ask)
    bid_volumes.append(bid_vol)
    ask_volumes.append(ask_vol)
    
    # Calculate raw imbalance (%)
    total_vol = bid_vol + ask_vol
    if total_vol > 0:
        imbalance = ((bid_vol - ask_vol) / total_vol) * 100
    else:
        imbalance = 0

    # --- Heikin Ashi Calculation for Cumulative Imbalance ---
    # Treat each update as a degenerate candle (open, high, low, close all equal to imbalance)
    if ha_imbalance_open is None:
        ha_imbalance_open = imbalance
        ha_imbalance_close = imbalance
    else:
        ha_imbalance_open = (ha_imbalance_open + ha_imbalance_close) / 2
        ha_imbalance_close = imbalance
    ha_value = (ha_imbalance_open + ha_imbalance_close) / 2  # Smoothed HA value
    ha_imbalance_values.append(ha_value)
    ha_imbalance_times.append(now)
    imbalance_record.append(ha_value)
    
    # --- Extra Metric: Futures Open Interest (Heikin Ashi Calculation) ---
    futures_oi = fetch_futures_open_interest()
    if futures_oi is not None:
        if baseline_futures_oi is None:
            baseline_futures_oi = futures_oi
        change_oi = ((futures_oi - baseline_futures_oi) / baseline_futures_oi) * 100
        if ha_oi_open is None:
            ha_oi_open = change_oi
            ha_oi_close = change_oi
        else:
            ha_oi_open = (ha_oi_open + ha_oi_close) / 2
            ha_oi_close = change_oi
        ha_oi_value = (ha_oi_open + ha_oi_close) / 2
        open_interest_values.append(ha_oi_value)
        open_interest_record.append(ha_oi_value)
    
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
# The Heikin Ashi–smoothed cumulative imbalance will be drawn via a LineCollection.
mid_line = ax1.axhline(0, color="black", linestyle="--", linewidth=2)

ax1.set_ylabel("Cumulative Imbalance (%)")
ax1.legend(loc="upper left")
ax1.set_title(f"Live Order Book for {SYMBOL}")
ax1.set_ylim(-100, 100)

def init():
    current_time = datetime.now()
    # Stretch the x-axis 10% further than LOOKBACK_SECONDS
    effective_lookback = LOOKBACK_SECONDS * 1.1
    ax1.set_xlim(current_time - timedelta(seconds=effective_lookback), current_time)
    ax2.axis('off')  # Hide table axis initially
    return line_bid, line_ask, mid_line

def animate(frame):
    global reg_collection
    update_data()
    if not times:
        return line_bid, line_ask, mid_line

    current_time = datetime.now()
    effective_lookback = LOOKBACK_SECONDS * 1.1  # 10% more than the base lookback window
    ax1.set_xlim(current_time - timedelta(seconds=effective_lookback), current_time)
    
    # Update raw bid/ask chart.
    times_list = list(times)
    line_bid.set_data(times_list, list(bids))
    line_ask.set_data(times_list, list(asks))
    
    # --- Update Heikin Ashi–smoothed cumulative imbalance line ---
    ha_times = list(ha_imbalance_times)
    ha_values = list(ha_imbalance_values)
    if len(ha_times) >= 2:
        x = mdates.date2num(ha_times)
        segments = []
        for i in range(len(x) - 1):
            segments.append([[x[i], ha_values[i]], [x[i+1], ha_values[i+1]]])
        colors = ["green" if ha_values[i+1] >= 0 else "red" for i in range(len(x) - 1)]
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
