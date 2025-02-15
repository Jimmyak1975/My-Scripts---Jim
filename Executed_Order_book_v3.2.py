import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- Configuration ----------------
SYMBOL = "LTCUSDT"  # Futures pair to monitor
TRADES_API_URL = "https://fapi.binance.com/fapi/v1/trades"  # Endpoint for recent trades
FETCH_INTERVAL = 1  # Fetch data every 1 second
TABLE_SIZE = 100  # Show the top 100 trades in the table

# Colors for alternating rows (base dark theme)
BASE_COLOR_EVEN = "#2E2E2E"  # Dark gray
BASE_COLOR_ODD = "#1E1E1E"   # Almost black

# Highlight colors based on cumulative thresholds
HIGHLIGHT_YELLOW = "yellow"      # For cumulative > 200,000
HIGHLIGHT_BLUE   = "dodgerblue"  # For cumulative > 300,000
HIGHLIGHT_ORANGE = "orange"      # For cumulative > 500,000

# ---------------- Fetch Executed Trades Data ----------------
def fetch_executed_trades():
    """Fetch recent executed trades for the given symbol, up to 1000."""
    try:
        params = {"symbol": SYMBOL, "limit": 1000}  # fetch a bigger batch
        response = requests.get(TRADES_API_URL, params=params, timeout=10)
        trades = response.json()
        return trades
    except Exception as e:
        print(f"Error fetching executed trades: {e}")
        return []

# ---------------- Process Trades to Show the Largest 100 ----------------
def process_trades(trades):
    """
    1. Calculate the USD value of each trade.
    2. Sort by trade value (descending).
    3. Slice the top 100 trades.
    4. Separate into SELL vs BUY.
    5. Compute cumulative totals in descending order of size.
    """
    # Convert raw data to a list of (side, value) for easy sorting.
    # side = "SELL" if isBuyerMaker=True, else "BUY"
    processed = []
    for trade in trades:
        price = float(trade["price"])
        qty = float(trade["qty"])
        value = price * qty
        side = "SELL" if trade.get("isBuyerMaker") else "BUY"
        processed.append((side, value))

    # Sort trades by value descending
    processed.sort(key=lambda x: x[1], reverse=True)

    # Take the top 100 biggest trades
    top_trades = processed[:TABLE_SIZE]

    # Separate into SELL and BUY, preserving the descending order
    sell_values = [t[1] for t in top_trades if t[0] == "SELL"]
    buy_values  = [t[1] for t in top_trades if t[0] == "BUY"]

    # Compute cumulative sums for SELL
    cumulative_sells = []
    cum_sell = 0.0
    for val in sell_values:
        cum_sell += val
        cumulative_sells.append(cum_sell)

    # Compute cumulative sums for BUY
    cumulative_buys = []
    cum_buy = 0.0
    for val in buy_values:
        cum_buy += val
        cumulative_buys.append(cum_buy)

    return cumulative_sells, cumulative_buys

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(4, 12))
try:
    fig.canvas.manager.set_window_title(SYMBOL)
except Exception:
    pass  # Some backends may not support set_window_title

ax.axis('off')  # Hide axes

def update(frame):
    """Fetch trades, process to get top 100 largest trades, display in table."""
    trades = fetch_executed_trades()
    cumulative_sells, cumulative_buys = process_trades(trades)

    # We want to display up to TABLE_SIZE rows in the final table.
    # But note that the number of SELL or BUY trades in the top 100 might be fewer than 100 total.
    # We'll pad each list so we can build a consistent table of TABLE_SIZE rows.
    max_rows = TABLE_SIZE

    while len(cumulative_sells) < max_rows:
        cumulative_sells.append(0)
    while len(cumulative_buys) < max_rows:
        cumulative_buys.append(0)

    # We'll just display the first TABLE_SIZE from each, though they should already be exactly TABLE_SIZE length now.
    cumulative_sells = cumulative_sells[:max_rows]
    cumulative_buys  = cumulative_buys[:max_rows]

    # Build table data
    # The top row in each list is the largest trade, the bottom row is the smaller ones in the top 100.
    # So row 0 = largest trade, row 1 = second largest, etc.
    table_data = []
    for i in range(max_rows):
        sell_val = f"{int(round(cumulative_sells[i])):,}" if cumulative_sells[i] > 0 else ""
        buy_val  = f"{int(round(cumulative_buys[i])):,}" if cumulative_buys[i] > 0 else ""
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
            # Header row
            cell.set_facecolor("#404040")
            cell.set_text_props(weight='bold', color="white")
        else:
            base_color = BASE_COLOR_EVEN if row % 2 == 0 else BASE_COLOR_ODD
            cell.set_facecolor(base_color)
    
    # Apply highlighting based on cumulative thresholds
    for i in range(1, max_rows + 1):
        cell_sell = table[i, 0]
        cell_buy  = table[i, 1]
        try:
            sell_val_int = int(cumulative_sells[i - 1]) if cumulative_sells[i - 1] > 0 else 0
            buy_val_int  = int(cumulative_buys[i - 1]) if cumulative_buys[i - 1] > 0 else 0
        except Exception:
            sell_val_int, buy_val_int = 0, 0
        
        cell_sell.set_text_props(color="red",   weight='bold')
        cell_buy.set_text_props(color="green", weight='bold')
        
        # Highlight SELL side
        if sell_val_int > 500000:
            cell_sell.set_facecolor(HIGHLIGHT_ORANGE)
        elif sell_val_int > 300000:
            cell_sell.set_facecolor(HIGHLIGHT_BLUE)
        elif sell_val_int > 200000:
            cell_sell.set_facecolor(HIGHLIGHT_YELLOW)
        
        # Highlight BUY side
        if buy_val_int > 500000:
            cell_buy.set_facecolor(HIGHLIGHT_ORANGE)
        elif buy_val_int > 300000:
            cell_buy.set_facecolor(HIGHLIGHT_BLUE)
        elif buy_val_int > 200000:
            cell_buy.set_facecolor(HIGHLIGHT_YELLOW)

ani = animation.FuncAnimation(fig, update, interval=FETCH_INTERVAL * 1000, cache_frame_data=False)
plt.show()
