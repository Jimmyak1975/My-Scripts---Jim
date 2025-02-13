import requests
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- Configuration ----------------
SYMBOL = "LTCUSDT"  # Futures pair to monitor
ORDERBOOK_API_URL = "https://fapi.binance.com/fapi/v1/depth"
FETCH_INTERVAL = 0.5  # Fetch data every 0.5 seconds
TABLE_SIZE = 100  # Number of rows to display

# ---------------- Fetch Order Book Data ----------------
def fetch_order_book():
    """Fetch the current order book from Binance Futures."""
    try:
        params = {"symbol": SYMBOL, "limit": TABLE_SIZE}
        response = requests.get(ORDERBOOK_API_URL, params=params, timeout=5)
        data = response.json()

        bids = [float(size) * float(price) for price, size in data["bids"]]
        asks = [float(size) * float(price) for price, size in data["asks"]]

        return bids, asks
    except Exception as e:
        print(f"Error fetching order book: {e}")
        return [], []

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(4, 12))  # Compact size
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
    col_labels = ["ASK", "BID"]

    ax.clear()
    ax.axis('off')

    # Create the table with tighter row spacing
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')

    # Formatting table
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Adjust font size for compact fit
    table.scale(0.85, 0.85)  # Reduce row height and width for tighter fit

    # Apply thinner lines for a cleaner look
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.2)  # Very thin table borders

    # Color formatting for bids (green) and asks (red) + bold numbers + highlight rules
    for i in range(1, TABLE_SIZE + 1):
        cell_ask = table[i, 0]
        cell_bid = table[i, 1]

        # Get values as numbers
        try:
            ask_val = int(asks[i - 1]) if asks[i - 1] > 0 else 0
            bid_val = int(bids[i - 1]) if bids[i - 1] > 0 else 0
        except:
            ask_val, bid_val = 0, 0

        # Color for ASK (Red)
        cell_ask.set_text_props(color="red", fontweight='bold')

        # Color for BID (Green)
        cell_bid.set_text_props(color="green", fontweight='bold')

        # Highlighting Logic:
        if ask_val > 300000:
            cell_ask.set_facecolor("lightblue")  # 🔵 Highlight in blue
        elif ask_val > 100000:
            cell_ask.set_facecolor("yellow")  # 🟡 Highlight in yellow

        if bid_val > 300000:
            cell_bid.set_facecolor("lightblue")  # 🔵 Highlight in blue
        elif bid_val > 100000:
            cell_bid.set_facecolor("yellow")  # 🟡 Highlight in yellow

ani = animation.FuncAnimation(fig, update, interval=FETCH_INTERVAL * 1000, cache_frame_data=False)

plt.show()
