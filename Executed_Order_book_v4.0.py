import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import concurrent.futures
from datetime import datetime  # For blinking timing

# ---------------- Configuration ----------------
# Using SOL vs USDT pair for most exchanges; Bitstamp uses SOL vs USD
SYMBOL = "SOLUSDT"         
FETCH_INTERVAL = 2         # Fetch every 2 seconds
TABLE_SIZE = 50            # Display the top 50 trades per side
TOTAL_FETCH_LIMIT = 1000   # Number of trades to fetch for computing totals
VOLUME_THRESHOLD = 10      # Filter trades > 10 SOL

# Mapping of a common symbol to exchange-specific symbols for SOL
EXCHANGE_SYMBOLS = {
    "Binance": "SOLUSDT",
    "Coinbase": "SOL-USD",
    "Kraken": "SOLUSD",
    "Bitfinex": "tSOLUSD",
    "Huobi Global": "solusdt",
    "OKX": "SOL-USDT",
    "KuCoin": "SOL-USDT",     # using hyphen
    "Bitstamp": "solusd",     # Updated to SOL vs USD (lowercase)
    "Gemini": "SOLUSD",       # Gemini trades against USD
    "Gate.io": "SOL_USDT"     # using underscore
}

# API endpoints using the dynamic symbol mapping
EXCHANGES = {
    "Binance": f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={EXCHANGE_SYMBOLS['Binance']}",
    "Coinbase": f"https://api.exchange.coinbase.com/products/{EXCHANGE_SYMBOLS['Coinbase']}/trades",
    "Kraken": f"https://api.kraken.com/0/public/Trades?pair={EXCHANGE_SYMBOLS['Kraken']}",
    "Bitfinex": f"https://api-pub.bitfinex.com/v2/trades/{EXCHANGE_SYMBOLS['Bitfinex']}/hist",
    "Huobi Global": f"https://api.huobi.pro/market/history/trade?symbol={EXCHANGE_SYMBOLS['Huobi Global']}",
    "OKX": f"https://www.okx.com/api/v5/market/trades?instId={EXCHANGE_SYMBOLS['OKX']}",
    "KuCoin": f"https://api.kucoin.com/api/v1/market/histories?symbol={EXCHANGE_SYMBOLS['KuCoin']}",
    "Bitstamp": f"https://www.bitstamp.net/api/v2/transactions/{EXCHANGE_SYMBOLS['Bitstamp']}/",
    "Gemini": f"https://api.gemini.com/v1/trades/{EXCHANGE_SYMBOLS['Gemini'].lower()}",
    "Gate.io": f"https://api.gateio.ws/api/v4/spot/trades?currency_pair={EXCHANGE_SYMBOLS['Gate.io']}"
}

# Colors
BASE_COLOR_EVEN = "#2E2E2E"
BASE_COLOR_ODD = "#1E1E1E"
# Highlight thresholds:
HIGHLIGHT_YELLOW = "yellow"  # > $50,000
HIGHLIGHT_ORANGE = "orange"  # > $100,000
HIGHLIGHT_BLUE = "blue"      # > $200,000
HIGHLIGHT_RED = "purple"     # > $500,000 (changed to purple)

# ---------------- Fetch Trades ----------------
def fetch_exchange_trades(exchange, url):
    trades = []
    try:
        if exchange == "Binance":
            params = {"limit": TOTAL_FETCH_LIMIT}
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
            if "result" in response and EXCHANGE_SYMBOLS["Kraken"] in response["result"]:
                trades_data = response["result"][EXCHANGE_SYMBOLS["Kraken"]]
                for trade in trades_data:
                    price = float(trade[0])
                    qty = float(trade[1])
                    value = price * qty
                    is_sell = trade[2] == "s"
                    if qty >= VOLUME_THRESHOLD:
                        trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
            else:
                print(f"Kraken response missing 'result' or '{EXCHANGE_SYMBOLS['Kraken']}': {response}")
                
        elif exchange == "Bitfinex":
            params = {"limit": TOTAL_FETCH_LIMIT}
            response = requests.get(url, params=params, timeout=10).json()
            for trade in response:
                if len(trade) >= 4:
                    price = float(trade[3])
                    qty = abs(float(trade[2]))
                    value = price * qty
                    is_sell = float(trade[2]) < 0
                    if qty >= VOLUME_THRESHOLD:
                        trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
                        
        elif exchange == "Huobi Global":
            response = requests.get(url, timeout=10).json()
            if "data" in response and len(response["data"]) > 0:
                trade_group = response["data"][0]
                trades_data = trade_group.get("data", [])
                for trade in trades_data:
                    price = float(trade["price"])
                    qty = float(trade["amount"])
                    value = price * qty
                    is_sell = trade["direction"] == "sell"
                    if qty >= VOLUME_THRESHOLD:
                        trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
            else:
                print(f"Huobi response error: {response}")
                
        elif exchange == "OKX":
            response = requests.get(url, timeout=10).json()
            if "data" in response and response["data"]:
                trades_data = response["data"][0].get("trades", [])
                for trade in trades_data:
                    price = float(trade["price"])
                    qty = float(trade["size"])
                    value = price * qty
                    is_sell = trade["side"].lower() == "sell"
                    if qty >= VOLUME_THRESHOLD:
                        trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
            else:
                print(f"OKX response error: {response}")
                
        elif exchange == "KuCoin":
            response = requests.get(url, timeout=10).json()
            if response.get("code") == "200000" and "data" in response:
                for trade in response["data"]:
                    price = float(trade["price"])
                    qty = float(trade["size"])
                    value = price * qty
                    is_sell = trade["side"].lower() == "sell"
                    if qty >= VOLUME_THRESHOLD:
                        trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
            else:
                print(f"KuCoin API error: {response}")
                
        elif exchange == "Bitstamp":
            response = requests.get(url, timeout=10).json()
            if isinstance(response, list):
                for trade in response:
                    price = float(trade["price"])
                    qty = float(trade["amount"])
                    value = price * qty
                    # Bitstamp: "type": "0" for buy, "1" for sell
                    is_sell = int(trade["type"]) == 1
                    if qty >= VOLUME_THRESHOLD:
                        trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
            else:
                print(f"Bitstamp API error: {response}")
                    
        elif exchange == "Gemini":
            response = requests.get(url, timeout=10).json()
            for trade in response:
                price = float(trade["price"])
                qty = float(trade["amount"])
                value = price * qty
                is_sell = trade["type"].lower() == "sell"
                if qty >= VOLUME_THRESHOLD:
                    trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
                    
        elif exchange == "Gate.io":
            response = requests.get(url, timeout=10).json()
            for trade in response:
                price = float(trade["price"])
                qty = float(trade["amount"])  # Changed from 'size' to 'amount'
                value = price * qty
                is_sell = trade["side"].lower() == "sell"
                if qty >= VOLUME_THRESHOLD:
                    trades.append({"exchange": exchange, "value": value, "is_sell": is_sell})
                    
    except Exception as e:
        print(f"Error fetching from {exchange}: {e}")
    return trades

def fetch_trades():
    """Fetch recent trades from multiple exchanges concurrently."""
    all_trades = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(EXCHANGES)) as executor:
        future_to_exchange = {
            executor.submit(fetch_exchange_trades, exchange, url): exchange
            for exchange, url in EXCHANGES.items()
        }
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
    # Separate trades into sells and buys, then sort and compute totals
    sell_trades = [t["value"] for t in trades if t["is_sell"]]
    buy_trades  = [t["value"] for t in trades if not t["is_sell"]]
    
    total_sell = sum(sell_trades)
    total_buy  = sum(buy_trades)
    
    sell_trades.sort(reverse=True)
    buy_trades.sort(reverse=True)
    
    display_sell = sell_trades[:TABLE_SIZE]
    display_buy  = buy_trades[:TABLE_SIZE]
    
    return total_sell, total_buy, display_sell, display_buy

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(4, 10))
fig.canvas.manager.set_window_title(f"{SYMBOL} Live Trades")

# ✅ ONLY CHANGE: dark-grey background for the entire window
fig.patch.set_facecolor(BASE_COLOR_ODD)
ax.set_facecolor(BASE_COLOR_ODD)

ax.axis('off')

def update(frame):
    # Determine if we are in the blink period:
    now = datetime.now()
    blink = (now.minute == 57 and now.second < 15)
    blink_toggle = (now.second % 2 == 0) if blink else False

    trades = fetch_trades()
    total_sell, total_buy, sell_values, buy_values = process_trades(trades)
    
    max_rows = TABLE_SIZE
    sell_values += [0] * (max_rows - len(sell_values))
    buy_values  += [0] * (max_rows - len(buy_values))
    sell_values = sell_values[:max_rows]
    buy_values  = buy_values[:max_rows]
    
    totals_row = [f"{int(round(total_buy)):,}" if total_buy > 0 else "",
                  f"{int(round(total_sell)):,}" if total_sell > 0 else ""]
    
    trade_rows = [
        [f"{int(round(bv)):,}" if bv > 0 else "",
         f"{int(round(sv)):,}" if sv > 0 else ""]
        for sv, bv in zip(sell_values, buy_values)
    ]
    
    table_data = [totals_row] + trade_rows
    
    ax.clear()

    # ✅ ONLY CHANGE: re-apply after clear() so white doesn't come back
    fig.patch.set_facecolor(BASE_COLOR_ODD)
    ax.set_facecolor(BASE_COLOR_ODD)

    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=["BUY", "SELL"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.85, 0.85)
    
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
            base_color = BASE_COLOR_EVEN if (row % 2 == 0) else BASE_COLOR_ODD
            cell.set_facecolor(base_color)
            cell.set_text_props(color="green" if col == 0 else "red", weight='bold')
    
    for i in range(2, len(table_data) + 1):
        trade_index = i - 2
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
    
    if blink:
        for i in range(2, len(table_data) + 1):
            for col in [0, 1]:
                cell = table[i, col]
                if blink_toggle:
                    cell.set_facecolor("white")
                    
ani = animation.FuncAnimation(fig, update, interval=FETCH_INTERVAL * 1000, cache_frame_data=False)
plt.show()
