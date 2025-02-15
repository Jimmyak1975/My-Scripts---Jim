import requests
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- Configuration ----------------
SYMBOL = "LTCUSDT"  # Futures pair to monitor
ORDERBOOK_API_URL = "https://fapi.binance.com/fapi/v1/depth"
FETCH_INTERVAL = 0.5  # Fetch data every 0.5 seconds
TABLE_SIZE = 100  # Number of rows to display

# Colors for alternating rows (base dark theme)
BASE_COLOR_EVEN = "#2E2E2E"  # Dark gray
BASE_COLOR_ODD = "#1E1E1E"   # Almost black

# Highlight colors
HIGHLIGHT_YELLOW = "yellow"      # For >100,000
HIGHLIGHT_BLUE = "dodgerblue"    # For >300,000
HIGHLIGHT_ORANGE = "orange"      # For >500,000

# ---------------- Fetch Order Book Data ----------------
def fetch_order_book():
    """Fetch the current order book from Binance Futures and compute cumulative volumes."""
    try:
        params = {"symbol": SYMBOL, "limit": TABLE_SIZE}
        response = requests.get(ORDERBOOK_API_URL, params=params, timeout=5)
        data = response.json()

        # Calculate cumulative volumes for bids (from highest bid downward)
        bids = []
        cum_bid = 0.0
        for price, size in data["bids"]:
            row_value = float(price) * float(size)
            cum_bid += row_value
            bids.append(cum_bid)

        # Calculate cumulative volumes for asks (from lowest ask upward)
        asks = []
        cum_ask = 0.0
        for price, size in data["asks"]:
            row_value = float(price) * float(size)
            cum_ask += row_value
            asks.append(cum_ask)

        return bids, asks
    except Exception as e:
        print(f"Error fetching order book: {e}")
        return [], []

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(4, 12))  # Compact size
plt.get_current_fig_manager().set_window_title(SYMBOL)  # Set window title to the coin symbol
ax.axis('off')  # Hide axes

def update(frame):
    """Fetch data and update the table dynamically."""
    bids, asks = fetch_order_book()

    # Ensure table has exactly TABLE_SIZE rows
    bids = bids[:TABLE_SIZE] + [0] * (TABLE_SIZE - len(bids))
    asks = asks[:TABLE_SIZE] + [0] * (TABLE_SIZE - len(asks))

    # Prepare table data with formatted numbers
    table_data = []
    for i in range(TABLE_SIZE):
        ask_value = f"{int(round(asks[i])):,}" if asks[i] > 0 else ""
        bid_value = f"{int(round(bids[i])):,}" if bids[i] > 0 else ""
        table_data.append([ask_value, bid_value])

    # Table headers
    col_labels = ["ASK (Cumulative)", "BID (Cumulative)"]

    ax.clear()
    ax.axis('off')

    # Create the table; row 0 is the header
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')

    # Set font size and scale for a compact display
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.85, 0.85)

    # Apply thin borders to cells
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.2)

    # Set dark background for non-header cells with alternating colors
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(weight='bold', color="white")
        else:
            base_color = BASE_COLOR_EVEN if row % 2 == 0 else BASE_COLOR_ODD
            cell.set_facecolor(base_color)

    # Iterate over data rows to set text colors and apply highlights if thresholds are met
    for i in range(1, TABLE_SIZE + 1):
        cell_ask = table[i, 0]
        cell_bid = table[i, 1]

        try:
            ask_val = int(asks[i - 1]) if asks[i - 1] > 0 else 0
            bid_val = int(bids[i - 1]) if bids[i - 1] > 0 else 0
        except Exception as e:
            ask_val, bid_val = 0, 0

        # Set text colors: ask in red, bid in green
        cell_ask.set_text_props(color="red", weight='bold')
        cell_bid.set_text_props(color="green", weight='bold')

        # Highlighting logic for ASK side
        if ask_val > 500000:
            cell_ask.set_facecolor(HIGHLIGHT_ORANGE)
        elif ask_val > 300000:
            cell_ask.set_facecolor(HIGHLIGHT_BLUE)
        elif ask_val > 100000:
            cell_ask.set_facecolor(HIGHLIGHT_YELLOW)

        # Highlighting logic for BID side
        if bid_val > 500000:
            cell_bid.set_facecolor(HIGHLIGHT_ORANGE)
        elif bid_val > 300000:
            cell_bid.set_facecolor(HIGHLIGHT_BLUE)
        elif bid_val > 100000:
            cell_bid.set_facecolor(HIGHLIGHT_YELLOW)

ani = animation.FuncAnimation(fig, update, interval=FETCH_INTERVAL * 1000, cache_frame_data=False)

plt.show()
