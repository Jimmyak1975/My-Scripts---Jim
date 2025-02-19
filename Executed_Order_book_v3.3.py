import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- Configuration ----------------
SYMBOL = "LTCUSDT"  # Futures pair to monitor
AGG_TRADES_API_URL = "https://fapi.binance.com/fapi/v1/aggTrades"  # Aggregated trades endpoint
FETCH_INTERVAL = 1  # Fetch data every 1 second
TABLE_SIZE = 100  # Show the top 100 trades for each side in the table

# Colors for alternating rows (base dark theme)
BASE_COLOR_EVEN = "#2E2E2E"  # Dark gray
BASE_COLOR_ODD = "#1E1E1E"   # Almost black

# Highlight colors based on trade value thresholds
HIGHLIGHT_YELLOW = "yellow"      # For values > 100,000
HIGHLIGHT_ORANGE = "orange"      # For values > 200,000
HIGHLIGHT_BLUE   = "blue"        # For values > 300,000

# ---------------- Fetch Aggregated Trades Data ----------------
def fetch_aggregated_trades():
    """Fetch recent aggregated trades for the given symbol, up to 1000."""
    try:
        params = {"symbol": SYMBOL, "limit": 1000}  # fetch a larger batch of trades
        response = requests.get(AGG_TRADES_API_URL, params=params, timeout=10)
        trades = response.json()
        return trades
    except Exception as e:
        print(f"Error fetching aggregated trades: {e}")
        return []

# ---------------- Process Trades to Show the Largest 100 in Each Side ----------------
def process_trades(trades):
    """
    1. Calculate the USD value of each aggregated trade.
    2. Separate trades into SELL and BUY based on the 'm' flag.
       (In aggTrades, if m is True then the buyer is the market maker, so it's treated as a SELL.)
    3. Sort each list (SELL and BUY) by trade value in descending order.
    4. Return the top 100 trade values for each side.
    """
    sell_trades = []
    buy_trades = []
    for trade in trades:
        price = float(trade["p"])
        qty = float(trade["q"])
        value = price * qty
        if trade.get("m"):
            sell_trades.append(value)
        else:
            buy_trades.append(value)
    
    sell_trades.sort(reverse=True)
    buy_trades.sort(reverse=True)
    
    top_sell = sell_trades[:TABLE_SIZE]
    top_buy = buy_trades[:TABLE_SIZE]
    
    return top_sell, top_buy

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(4, 12))
try:
    fig.canvas.manager.set_window_title(SYMBOL)
except Exception:
    pass  # Some backends may not support set_window_title

ax.axis('off')  # Hide axes

def update(frame):
    """Fetch aggregated trades, process to get top 100 largest BUY and SELL trades, and display them in a table."""
    trades = fetch_aggregated_trades()
    sell_values, buy_values = process_trades(trades)
    
    # Ensure each list has TABLE_SIZE rows by padding with zeros if needed.
    max_rows = TABLE_SIZE
    while len(sell_values) < max_rows:
        sell_values.append(0)
    while len(buy_values) < max_rows:
        buy_values.append(0)
    
    sell_values = sell_values[:max_rows]
    buy_values  = buy_values[:max_rows]
    
    # Build table data: format each trade value for display.
    table_data = []
    for i in range(max_rows):
        sell_val = f"{int(round(sell_values[i])):,}" if sell_values[i] > 0 else ""
        buy_val  = f"{int(round(buy_values[i])):,}" if buy_values[i] > 0 else ""
        table_data.append([sell_val, buy_val])
    
    # Table headers are "SELL" and "BUY"
    col_labels = ["SELL", "BUY"]
    
    ax.clear()
    ax.axis('off')
    
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.85, 0.85)
    
    # Apply thin borders and base alternating background colors.
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.2)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header row styling.
            cell.set_facecolor("#404040")
            cell.set_text_props(weight='bold', color="white")
        else:
            base_color = BASE_COLOR_EVEN if row % 2 == 0 else BASE_COLOR_ODD
            cell.set_facecolor(base_color)
            # Set text colors: red for SELL column, green for BUY column.
            if col == 0:
                cell.set_text_props(color="red", weight='bold')
            elif col == 1:
                cell.set_text_props(color="green", weight='bold')
    
    # Apply highlighting based on thresholds:
    # For each row, check the underlying trade value and override the cell background if thresholds are met.
    for i in range(1, max_rows + 1):
        # SELL side
        cell_sell = table[i, 0]
        sell_val = sell_values[i - 1]
        if sell_val > 300000:
            cell_sell.set_facecolor(HIGHLIGHT_BLUE)
        elif sell_val > 200000:
            cell_sell.set_facecolor(HIGHLIGHT_ORANGE)
        elif sell_val > 100000:
            cell_sell.set_facecolor(HIGHLIGHT_YELLOW)
        
        # BUY side
        cell_buy = table[i, 1]
        buy_val = buy_values[i - 1]
        if buy_val > 300000:
            cell_buy.set_facecolor(HIGHLIGHT_BLUE)
        elif buy_val > 200000:
            cell_buy.set_facecolor(HIGHLIGHT_ORANGE)
        elif buy_val > 100000:
            cell_buy.set_facecolor(HIGHLIGHT_YELLOW)

ani = animation.FuncAnimation(fig, update, interval=FETCH_INTERVAL * 1000, cache_frame_data=False)
plt.show()
