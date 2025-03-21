import requests
import time
import datetime
import json
import websocket
import os
import sys
import threading

# ======================================
# Configuration – Replace with your data
# ======================================
TELEGRAM_BOT_TOKEN = "7721789198:AAFQbDqk7Ln5O-O3eGwYh05MZMCdVunfMHI"
TELEGRAM_CHAT_ID = "7052327528"  # e.g., an integer or string
BINANCE_FUTURES_URL = "https://fapi.binance.com"

# List of coins to monitor (only 5 coins)
COINS = ["SOLUSDT", "XRPUSDT"]

# Global dictionary for accumulation tracking.
# For each symbol, we store:
#   - baseline: the price at the start of the accumulation window (using the candle's open)
#   - window_start: the timestamp when the window started (in seconds)
#   - baseline_time: the formatted time when the window started (for notifications)
#   - last_notif_timestamp: last time a notification was sent (for cooldown)
#   - last_update: the last processed candle open time (to avoid duplicate processing)
cumulative_data = {}

# ======================================
# Utility Functions
# ======================================

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def format_percentage_change(percent: float, symbol: str = None) -> str:
    return f"🟣{percent:+.2f}%"

# ======================================
# WebSocket Event Handlers
# ======================================

def on_message(ws, message):
    try:
        data = json.loads(message)
        if "data" not in data or data["data"].get("e") != "kline":
            return
        
        kline_data = data["data"]["k"]
        symbol = data["data"]["s"]
        
        # Process every update—even if the candle is not closed.
        current_price = float(kline_data["c"])
        update_time = time.time()
        
        # Use the candle's open time (in seconds) as a unique identifier.
        candle_open_time = kline_data["t"] / 1000
        
        # Initialize accumulation window if needed.
        if symbol not in cumulative_data:
            baseline = float(kline_data["o"])  # candle open as baseline
            cumulative_data[symbol] = {
                "baseline": baseline,
                "window_start": candle_open_time,
                "baseline_time": datetime.datetime.fromtimestamp(candle_open_time).strftime("%H:%M"),
                "last_notif_timestamp": 0,
                "last_update": candle_open_time
            }
        
        symbol_data = cumulative_data[symbol]
        # Avoid processing duplicate updates for the same candle.
        if candle_open_time == symbol_data.get("last_update"):
            pass
        else:
            symbol_data["last_update"] = candle_open_time
        
        baseline = symbol_data["baseline"]
        cumulative_percent = ((current_price - baseline) / baseline) * 100
        
        threshold = 0.2    # 0.2% trigger threshold
        cooldown = 180     # 180 seconds (3 minutes) cooldown
        max_window = 120   # 2-minute maximum accumulation window
        
        # First, check if the threshold is reached (even before candle closes).
        if abs(cumulative_percent) >= threshold:
            if update_time - symbol_data["last_notif_timestamp"] >= cooldown:
                symbol_data["last_notif_timestamp"] = update_time
                start_time = symbol_data["baseline_time"]
                end_time = datetime.datetime.fromtimestamp(update_time).strftime("%H:%M")
                symbol_name = symbol.replace("USDT", "")
                message_text = f"{symbol_name} - {start_time} to {end_time}\n{format_percentage_change(cumulative_percent, symbol)}"
                print("Sending trigger notification:", message_text)
                send_telegram_message(message_text)
                # Reset the accumulation window.
                symbol_data["baseline"] = current_price
                symbol_data["window_start"] = update_time
                symbol_data["baseline_time"] = datetime.datetime.fromtimestamp(update_time).strftime("%H:%M")
            return
        
        # Then, if no threshold is reached, check if the window exceeded the maximum.
        if update_time - symbol_data["window_start"] >= max_window:
            # Reset the window without triggering notification.
            symbol_data["baseline"] = float(kline_data["o"])  # Start new window using current candle's open.
            symbol_data["window_start"] = candle_open_time
            symbol_data["baseline_time"] = datetime.datetime.fromtimestamp(candle_open_time).strftime("%H:%M")
            return
        
    except Exception as e:
        print("Error in on_message:", e)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, code, msg):
    print("WebSocket closed:", code, msg)

def on_open(ws):
    print("WebSocket connection opened.")
    cumulative_data.clear()  # Clear state to avoid stale data.

# ======================================
# WebSocket Reconnection Logic
# ======================================

def run_websocket_loop(max_runtime=14400, reconnect_delay=5):
    # Build the combined stream URL.
    streams = "/".join([coin.lower() + "@kline_1m" for coin in COINS])
    ws_url = f"wss://fstream.binance.com/stream?streams={streams}"
    start_time = time.time()
    
    # Loop until the overall max runtime is reached.
    while time.time() - start_time < max_runtime:
        print("Starting WebSocket connection...")
        ws = websocket.WebSocketApp(ws_url,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close,
                                    on_open=on_open)
        # Run the WebSocket connection (this call blocks until the connection closes).
        ws.run_forever(ping_interval=20, ping_timeout=10)
        # If we reach here, the connection closed. Wait and then reconnect.
        if time.time() - start_time >= max_runtime:
            break
        print(f"WebSocket disconnected. Reconnecting in {reconnect_delay} seconds...")
        time.sleep(reconnect_delay)
    
    print("Max runtime reached. Exiting run_websocket_loop.")

# ======================================
# Script Entry Point – Live Monitoring Only
# ======================================

if __name__ == "__main__":
    try:
        print("Starting real-time monitoring via WebSocket...")
        run_websocket_loop(max_runtime=14400, reconnect_delay=5)
        print("Four hours elapsed. Restarting the script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
