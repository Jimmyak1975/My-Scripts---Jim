import requests
import time
import datetime

# ======================================
# Configuration – Replace with your data
# ======================================
TELEGRAM_BOT_TOKEN = "7721789198:AAFQbDqk7Ln5O-O3eGwYh05MZMCdVunfMHI"
TELEGRAM_CHAT_ID = "7052327528"  # e.g., an integer or string

# Binance Futures API base URL
BINANCE_FUTURES_URL = "https://fapi.binance.com"

# List of coins on Binance Futures to monitor.
# Make sure the symbol names match Binance Futures (here assumed as USDT pairs)
COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "MATICUSDT", "SOLUSDT", "DOTUSDT", "LTCUSDT",
    "LINKUSDT", "AVAXUSDT", "ATOMUSDT", "XLMUSDT", "ALGOUSDT",
    "ICPUSDT", "ETCUSDT", "VETUSDT", "FILUSDT", "TRXUSDT",
    # Additional coins
    "TAOUSDT",   # TAO
    "ENAUSDT",   # ENA
    "GMTUSDT",   # GMT
    "ONDOUSDT",  # ONDO
    "1000SHIBUSDT",  # 1000SHIB
    "1000PEPEUSDT",  # 1000PEPE
    "HBARUSDT",  # HBAR
    "SUIUSDT",   # SUI
    "LPTUSDT"    # LPT
]

# Remove any duplicates if they occur
COINS = list(dict.fromkeys(COINS))

# This dict will store the last triggered candle's open_time for each coin.
# This is used to ensure we only trigger one notification per 1-min candle.
last_triggered_candle = {}

# ======================================
# Utility Functions
# ======================================

def send_telegram_message(message: str):
    """
    Sends a message to your Telegram chat using the Bot API.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def get_current_kline(symbol: str) -> dict:
    """
    Fetches the latest 1-minute kline for the symbol.
    Returns a dictionary with:
      - open_time (in milliseconds)
      - open price (as float)
      - current price (using the 'close' field which updates in real time)
    If error occurs, returns None.
    """
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": 1
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        kline = response.json()[0]
        # kline format:
        # [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume, etc.
        return {
            "open_time": int(kline[0]),
            "open": float(kline[1]),
            "current": float(kline[4])
        }
    except Exception as e:
        print(f"Error getting current kline for {symbol}: {e}")
        return None

def get_historical_klines(symbol: str, start_time_ms: int) -> list:
    """
    Fetches historical 1-minute klines for the given symbol starting at start_time_ms.
    For a 4-hour lookback, we fetch up to 240 candles.
    """
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_time_ms,
        "limit": 240  # roughly 4 hours of 1-min candles
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting historical klines for {symbol}: {e}")
        return []

def format_percentage_change(percent: float) -> str:
    """
    Returns a formatted string with an emoji for the direction.
    """
    if percent > 0:
        return f"{percent:.2f}% 🟢"
    else:
        return f"{percent:.2f}% 🔴"

# ======================================
# Historical Summary Notification
# ======================================

def send_historical_summaries():
    """
    For each coin, look back over the past 4 hours (from when the script starts)
    and check each 1-min candle for a price change of >= 1% (from open to close).
    If found, send a one-time summary message per coin to Telegram.
    """
    now_ms = int(time.time() * 1000)
    four_hours_ago_ms = now_ms - (4 * 3600 * 1000)
    for symbol in COINS:
        klines = get_historical_klines(symbol, four_hours_ago_ms)
        # List to store events as strings.
        events = []
        for k in klines:
            open_time = int(k[0])
            open_price = float(k[1])
            close_price = float(k[4])
            # Calculate percentage change for that candle.
            if open_price == 0:
                continue
            percent_change = ((close_price - open_price) / open_price) * 100
            if abs(percent_change) >= 1:
                # Convert open time to human-readable format (HH:MM)
                time_str = datetime.datetime.fromtimestamp(open_time/1000).strftime("%H:%M")
                if percent_change > 0:
                    event_str = f"+{percent_change:.2f}% at {time_str}"
                else:
                    event_str = f"{percent_change:.2f}% at {time_str}"
                events.append(event_str)
        if events:
            message = f"Historical summary for {symbol} (past 4h): " + ", ".join(events)
            print("Sending historical summary:", message)
            send_telegram_message(message)
        else:
            print(f"No historical events for {symbol} in the past 4 hours.")

# ======================================
# Real-Time Monitoring (1-min candles)
# ======================================

def real_time_monitor():
    """
    Continuously monitors the current 1-min candle for each coin.
    If the price (compared to the candle's open) moves by ≥ 0.5% (in either direction)
    in real time—and if a trigger for that candle hasn't already been sent—then a trigger
    notification is sent. The trigger is allowed to repeat if a new candle meets the criteria.
    """
    global last_triggered_candle
    print("Starting real-time monitoring...")
    while True:
        for symbol in COINS:
            kline = get_current_kline(symbol)
            if kline is None:
                continue

            open_time = kline["open_time"]
            open_price = kline["open"]
            current_price = kline["current"]

            # Calculate the percent change relative to the candle's open
            if open_price == 0:
                continue
            percent_change = ((current_price - open_price) / open_price) * 100

            # Check if this candle has already triggered a notification
            last_trigger = last_triggered_candle.get(symbol)
            # Trigger if change exceeds 0.5% and notification hasn't been sent for this candle
            if last_trigger != open_time and abs(percent_change) >= 0.5:
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"{time_str} - {symbol}: {format_percentage_change(percent_change)} (current candle)"
                print("Sending trigger notification:", message)
                send_telegram_message(message)
                last_triggered_candle[symbol] = open_time
        # Poll every 5 seconds (adjust as needed)
        time.sleep(5)

# ======================================
# Script Entry Point
# ======================================

if __name__ == "__main__":
    try:
        # First, send one-time historical summary notifications per coin.
        print("Sending historical summary notifications...")
        send_historical_summaries()
        # Then, start the real-time monitoring.
        real_time_monitor()
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
