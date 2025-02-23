import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import concurrent.futures
from datetime import datetime  # Added for blinking timing

# ---------------- Configuration ----------------
SYMBOL = "ONDOUSDT"         # Litecoin vs USDT pair
FETCH_INTERVAL = 2         # Fetch every 2 seconds (faster updates)
TABLE_SIZE = 50            # Display the top 50 trades per side
TOTAL_FETCH_LIMIT = 1000   # Number of trades to fetch for computing totals
VOLUME_THRESHOLD = 10      # Filter trades > 10 LTC

# API endpoints
EXCHANGES = {
    "Binance": "https://fapi.binance.com/fapi/v1/aggTrades",
    "Coinbase": "https://api.exchange.coinbase.com/products/LTC-USD/trades",
    "Kraken": "https://api.kraken.com/0/public/Trades?pair=LTCUSD"
}

# Colors
BASE_COLOR_EVEN = "#2E2E2E"
BASE_COLOR_ODD = "#1E1E1E"
# Highlight thresholds:
HIGHLIGHT_YELLOW = "yellow"  # > $50,000
HIGHLIGHT_ORANGE = "orange"  # > $100,000
HIGHLIGHT_BLUE = "blue"      # > $200,000
HIGHLIGHT_RED = "purple"     # > $500,000  <-- Changed to purple

# ---------------- Fetch Trades ----------------
def fetch_exchange_trades(exchange, url):
    trades = []
    try:
        if exchange == "Binance":
            params = {"symbol": SYMBOL, "limit": TOTAL_FETCH_LIMIT}
            response = requests.get(url, params=params, timeout=10).json()
            for trade in response:
                price = float(trade["p"])
                qty = float(trade["q"])
                value = price * qty
                is_sell = trade.get("m")
                if qty >= VOLUME_THRESHOLD:
                    trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
        elif exchange == "Coinbase":
            response = requests.get(url, params={"limit": TOTAL_FETCH_LIMIT}, timeout=10).json()
            if not isinstance(response, list):
                print(f"Coinbase API error: {response}")
                return trades
            for trade in response:
                price = float(trade["price"])
                qty = float(trade["size"])
                value = price * qty
                is_sell = trade["side"] == "sell"
                if qty >= VOLUME_THRESHOLD:
                    trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
        elif exchange == "Kraken":
            response = requests.get(url, timeout=10).json()
            if "result" in response and "LTCUSD" in response["result"]:
                trades_data = response["result"]["LTCUSD"]
                for trade in trades_data:
                    price = float(trade[0])
                    qty = float(trade[1])
                    value = price * qty
                    is_sell = trade[2] == "s"
                    if qty >= VOLUME_THRESHOLD:
                        trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
            else:
                print(f"Kraken response missing 'result' or 'LTCUSD': {response}")
    except Exception as e:
        print(f"Error fetching from {exchange}: {e}")
    return trades

def fetch_trades():
    """Fetch recent trades from multiple exchanges concurrently."""
    all_trades = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(EXCHANGES)) as executor:
        future_to_exchange = {executor.submit(fetch_exchange_trades, exchange, url): exchange
                              for exchange, url in EXCHANGES.items()}
        for future in concurrent.futures.as_completed(future_to_exchange):
            exchange = future_to_exchange[future]
            try:
                trades = future.result()
                all_trades.extend(trades)
            except Exception as exc:
                print(f"Error fetching from {exchange}: {exc}")
    return all_trades

# ---------------- Process Trades ----------------
def process_trades(trades):
    """
    Separates trades into sells and buys, sorts them in descending order,
    and calculates:
      - The total value for each side (using the full fetched set)
      - The display lists of trades (top TABLE_SIZE for each side)
    """
    sell_trades = [t["value"] for t in trades if t["is_sell"]]
    buy_trades  = [t["value"] for t in trades if not t["is_sell"]]
    
    # Compute grand totals from the full fetched trades
    total_sell = sum(sell_trades)
    total_buy  = sum(buy_trades)
    
    # Sort the trades in descending order (high to low)
    sell_trades.sort(reverse=True)
    buy_trades.sort(reverse=True)
    
    # For display, take the top TABLE_SIZE trades from each side
    display_sell = sell_trades[:TABLE_SIZE]
    display_buy  = buy_trades[:TABLE_SIZE]
    
    return total_sell, total_buy, display_sell, display_buy

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(4, 10))
fig.canvas.manager.set_window_title(f"{SYMBOL} Live Trades")
ax.axis('off')

def update(frame):
    """Fetch, process, and display the latest trades along with grand totals."""
    # Determine if we are in the blink period:
    now = datetime.now()
    # Blink if we're between HH:57:00 and HH:57:15
    blink = (now.minute == 57 and now.second < 15)
    # Toggle based on current second (to alternate the blink)
    blink_toggle = (now.second % 2 == 0) if blink else False

    trades = fetch_trades()
    total_sell, total_buy, sell_values, buy_values = process_trades(trades)
    
    # Ensure display lists have exactly TABLE_SIZE items (pad with zeros if needed)
    max_rows = TABLE_SIZE
    sell_values += [0] * (max_rows - len(sell_values))
    buy_values  += [0] * (max_rows - len(buy_values))
    sell_values = sell_values[:max_rows]
    buy_values  = buy_values[:max_rows]
    
    # Create the totals row with BUY first then SELL (formatted with commas)
    totals_row = [f"{int(round(total_buy)):,}" if total_buy > 0 else "",
                  f"{int(round(total_sell)):,}" if total_sell > 0 else ""]
    
    # Create rows for individual trades with BUY value first then SELL value
    trade_rows = [
        [f"{int(round(bv)):,}" if bv > 0 else "",
         f"{int(round(sv)):,}" if sv > 0 else ""]
        for sv, bv in zip(sell_values, buy_values)
    ]
    
    # Insert the totals row at the top (below the header)
    table_data = [totals_row] + trade_rows
    
    ax.clear()
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=["BUY", "SELL"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.85, 0.85)
    
    # Format table cells normally
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.2)
        row, col = key
        if row == 0:  # Header row
            cell.set_facecolor("#404040")
            cell.set_text_props(weight='bold', color="white")
        elif row == 1:  # Totals row highlighted in pink
            cell.set_facecolor("pink")
            cell.set_text_props(weight='bold')
        else:
            # For trade rows (rows starting at index 2), alternate row colors
            base_color = BASE_COLOR_EVEN if (row % 2 == 0) else BASE_COLOR_ODD
            cell.set_facecolor(base_color)
            # Now, col==0 is BUY (green) and col==1 is SELL (red)
            cell.set_text_props(color="green" if col == 0 else "red", weight='bold')
    
    # Apply highlight thresholds on individual trade cells (skip header and totals row)
    for i in range(2, len(table_data) + 1):
        trade_index = i - 2  # Adjust index for our display lists
        # For each row, first column is BUY (using buy_values) and second column is SELL (using sell_values)
        for col, val in [(0, buy_values[trade_index]), (1, sell_values[trade_index])]:
            cell = table[i, col]
            if val > 500000:
                cell.set_facecolor(HIGHLIGHT_RED)
            elif val > 200000:
                cell.set_facecolor(HIGHLIGHT_BLUE)
            elif val > 100000:
                cell.set_facecolor(HIGHLIGHT_ORANGE)
            elif val > 50000:
                cell.set_facecolor(HIGHLIGHT_YELLOW)
    
    # If in blink period, override trade cell colors to create a blinking effect
    if blink:
        for i in range(2, len(table_data) + 1):  # Only trade rows
            for col in [0, 1]:
                cell = table[i, col]
                if blink_toggle:
                    cell.set_facecolor("white")
                    
ani = animation.FuncAnimation(fig, update, interval=FETCH_INTERVAL * 1000, cache_frame_data=False)
plt.show()
