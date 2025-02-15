import requests
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- Configuration ----------------
SYMBOL = "LTCUSDT"  # Futures pair to monitor
TRADES_API_URL = "https://fapi.binance.com/fapi/v1/trades"  # Endpoint for recent trades
FETCH_INTERVAL = 1  # Fetch data every 1 second
TABLE_SIZE = 100  # Maximum number of rows to display

# Colors for alternating rows (base dark theme)
BASE_COLOR_EVEN = "#2E2E2E"  # Dark gray
BASE_COLOR_ODD = "#1E1E1E"   # Almost black

# Highlight colors (optional thresholds, adjust as needed)
HIGHLIGHT_YELLOW = "yellow"      # For >100,000
HIGHLIGHT_BLUE = "dodgerblue"    # For >300,000
HIGHLIGHT_ORANGE = "orange"      # For >500,000

# ---------------- Fetch Executed Trades Data ----------------
def fetch_executed_trades():
    """Fetch recent executed trades for the given symbol."""
    try:
        params = {"symbol": SYMBOL, "limit": TABLE_SIZE * 2}  # fetch more to allow separate sorting
        response = requests.get(TRADES_API_URL, params=params, timeout=5)
        trades = response.json()
        # Binance returns most recent trades first; reverse to process oldest first
        trades.reverse()
        return trades
    except Exception as e:
        print(f"Error fetching executed trades: {e}")
        return []

# ---------------- Process Trades to Cumulative Totals ----------------
def process_trades(trades):
    """
    Separates trades into BUY and SELL lists, and computes cumulative USD value for each side.
    Returns:
        cumulative_sells: list of cumulative values for SELL trades (ask side)
        cumulative_buys: list of cumulative values for BUY trades (bid side)
    """
    sell_values = []
    buy_values = []
    
    # Binance's "isBuyerMaker" flag: if True, the buyer is the maker, so the executed trade is a sell.
    for trade in trades:
        price = float(trade["price"])
        qty = float(trade["qty"])
        value = price * qty
        if trade.get("isBuyerMaker"):
            sell_values.append(value)
        else:
            buy_values.append(value)
    
    # Compute cumulative totals
    cumulative_sells = []
    cum = 0.0
    for val in sell_values:
        cum += val
        cumulative_sells.append(cum)
    
    cumulative_buys = []
    cum = 0.0
    for val in buy_values:
        cum += val
        cumulative_buys.append(cum)
    
    return cumulative_sells, cumulative_buys

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(4, 12))  # Compact size
plt.get_current_fig_manager().set_window_title(SYMBOL)  # Set window title to the coin symbol
ax.axis('off')  # Hide axes

def update(frame):
    """Fetch executed trades, process cumulative totals, and update the table."""
    trades = fetch_executed_trades()
    cumulative_sells, cumulative_buys = process_trades(trades)
    
    # Determine maximum rows needed (at most TABLE_SIZE)
    max_rows = TABLE_SIZE
    # Pad shorter lists with zeros or empty strings
    while len(cumulative_sells) < max_rows:
        cumulative_sells.append(0)
    while len(cumulative_buys) < max_rows:
        cumulative_buys.append(0)
    cumulative_sells = cumulative_sells[:max_rows]
    cumulative_buys = cumulative_buys[:max_rows]
    
    # Prepare table data (left: SELL (ask), right: BUY (bid))
    table_data = []
    for i in range(max_rows):
        sell_val = f"{int(round(cumulative_sells[i])):,}" if cumulative_sells[i] > 0 else ""
        buy_val = f"{int(round(cumulative_buys[i])):,}" if cumulative_buys[i] > 0 else ""
        table_data.append([sell_val, buy_val])
    
    # Table headers
    col_labels = ["SELL (Cumulative)", "BUY (Cumulative)"]
    
    ax.clear()
    ax.axis('off')
    
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.85, 0.85)
    
    # Apply thin borders
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
    
    # Optional: apply highlighting if cumulative volume exceeds thresholds
    for i in range(1, max_rows + 1):
        cell_sell = table[i, 0]
        cell_buy = table[i, 1]
        try:
            sell_val = int(cumulative_sells[i - 1]) if cumulative_sells[i - 1] > 0 else 0
            buy_val = int(cumulative_buys[i - 1]) if cumulative_buys[i - 1] > 0 else 0
        except Exception as e:
            sell_val, buy_val = 0, 0
        
        cell_sell.set_text_props(color="red", weight='bold')
        cell_buy.set_text_props(color="green", weight='bold')
        
        if sell_val > 500000:
            cell_sell.set_facecolor(HIGHLIGHT_ORANGE)
        elif sell_val > 300000:
            cell_sell.set_facecolor(HIGHLIGHT_BLUE)
        elif sell_val > 100000:
            cell_sell.set_facecolor(HIGHLIGHT_YELLOW)
        
        if buy_val > 500000:
            cell_buy.set_facecolor(HIGHLIGHT_ORANGE)
        elif buy_val > 300000:
            cell_buy.set_facecolor(HIGHLIGHT_BLUE)
        elif buy_val > 100000:
            cell_buy.set_facecolor(HIGHLIGHT_YELLOW)

ani = animation.FuncAnimation(fig, update, interval=FETCH_INTERVAL * 1000, cache_frame_data=False)
plt.show()
