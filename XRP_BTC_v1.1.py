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

# Binance Futures API base URL (for REST calls)
BINANCE_FUTURES_URL = "https://fapi.binance.com"

# List of coins on Binance Futures to monitor (MATICUSDT removed)
COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "DOTUSDT", "LTCUSDT",
    "SUIUSDT", "TRXUSDT", "XLMUSDT", "HBARUSDT", "AVAXUSDT"
]

# Remove duplicates if they occur
COINS = list(dict.fromkeys(COINS))

# Global dictionary for real-time cumulative tracking.
cumulative_data = {}

# Global variable for BTC’s current price (updated in real time).
btc_current = None

# ======================================
# Utility Functions (REST-based)
# ======================================

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def send_message_in_chunks(message: str, max_len: int = 4096):
    lines = message.split("\n")
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) + 1 > max_len:
            send_telegram_message(chunk)
            chunk = line
        else:
            chunk = chunk + "\n" + line if chunk else line
    if chunk:
        send_telegram_message(chunk)

def get_historical_klines(symbol: str, start_time_ms: int) -> list:
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1m", "startTime": start_time_ms, "limit": 360}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting historical klines for {symbol}: {e}")
        return []

def format_percentage_change(percent: float) -> str:
    if percent > 0:
        return f"🟢{percent:.2f}%"
    else:
        return f"🔴{percent:.2f}%"

def send_historical_summaries():
    threshold = 0.5  # 0.5% threshold for historical events
    max_duration = 60 * 60 * 1000  # 60 minutes in milliseconds
    now_ms = int(time.time() * 1000)
    six_hours_ago_ms = now_ms - (6 * 3600 * 1000)
    summary_list = []

    for symbol in COINS:
        klines = get_historical_klines(symbol, six_hours_ago_ms)
        if not klines:
            print(f"No historical data for {symbol}")
            continue

        events = []
        baseline_price = float(klines[0][1])
        baseline_time = int(klines[0][0])
        for k in klines:
            candle_time = int(k[0])
            candle_close = float(k[4])
            if (candle_time - baseline_time) > max_duration:
                baseline_price = candle_close
                baseline_time = candle_time
                continue
            cumulative_change = ((candle_close - baseline_price) / baseline_price) * 100
            if abs(cumulative_change) >= threshold:
                start_str = datetime.datetime.fromtimestamp(baseline_time / 1000).strftime("%H:%M")
                end_str = datetime.datetime.fromtimestamp(candle_time / 1000).strftime("%H:%M")
                formatted_change = format_percentage_change(cumulative_change)
                event_str = f"{start_str}-{end_str}: {formatted_change}"
                events.append(event_str)
                baseline_price = candle_close
                baseline_time = candle_time

        if events:
            summary_list.append((symbol, events))

    summary_list.sort(key=lambda x: len(x[1]), reverse=True)

    if summary_list:
        aggregated_parts = []
        for (symbol, events) in summary_list:
            symbol_name = symbol.replace("USDT", "")
            header = f"HS {symbol_name}"
            coin_message = header + "\n" + "\n".join(events)
            aggregated_parts.append(coin_message)
        aggregated_message = "\n\n".join(aggregated_parts)
        print("Sending aggregated historical summary:\n", aggregated_message)
        send_message_in_chunks(aggregated_message)
    else:
        print("No significant historical events in the past 6 hours.")

# ======================================
# Real-Time Monitoring via WebSocket
# ======================================

def on_message(ws, message):
    global btc_current
    try:
        data = json.loads(message)
        if "data" in data and data["data"].get("e") == "kline":
            kline_data = data["data"]["k"]
            symbol = data["data"]["s"]
            current_price = float(kline_data["c"])
            
            if symbol == "BTCUSDT":
                btc_current = current_price
            
            if symbol not in cumulative_data:
                cumulative_data[symbol] = {
                    "baseline": current_price,
                    "direction": 0,
                    "baseline_time": datetime.datetime.now().strftime("%H:%M"),
                    "baseline_timestamp": time.time(),
                    "last_notif_timestamp": 0
                }
            
            baseline = cumulative_data[symbol]["baseline"]
            last_direction = cumulative_data[symbol]["direction"]
            new_direction = 1 if current_price > baseline else (-1 if current_price < baseline else 0)
            
            if last_direction != 0 and new_direction != last_direction:
                cumulative_data[symbol]["baseline"] = current_price
                cumulative_data[symbol]["direction"] = new_direction
                cumulative_data[symbol]["baseline_time"] = datetime.datetime.now().strftime("%H:%M")
                cumulative_data[symbol]["baseline_timestamp"] = time.time()
            else:
                now_ts = time.time()
                if now_ts - cumulative_data[symbol]["baseline_timestamp"] > 3600:
                    cumulative_data[symbol]["baseline"] = current_price
                    cumulative_data[symbol]["direction"] = new_direction
                    cumulative_data[symbol]["baseline_time"] = datetime.datetime.now().strftime("%H:%M")
                    cumulative_data[symbol]["baseline_timestamp"] = now_ts
                    return

                cumulative_percent = ((current_price - baseline) / baseline) * 100
                if abs(cumulative_percent) >= 0.5:  # 0.5% threshold
                    if now_ts - cumulative_data[symbol]["last_notif_timestamp"] < 600:
                        return
                    
                    start_time = cumulative_data[symbol]["baseline_time"]
                    end_time = datetime.datetime.now().strftime("%H:%M")
                    symbol_name = symbol.replace("USDT", "")
                    message_text = f"{symbol_name} - {start_time} to {end_time}\n{format_percentage_change(cumulative_percent)}"
                    
                    time_diff = now_ts - cumulative_data[symbol]["baseline_timestamp"]
                    if (symbol != "BTCUSDT" and cumulative_percent > 0 and
                        time_diff <= 3600 and btc_current is not None and "BTCUSDT" in cumulative_data):
                        btc_baseline = cumulative_data["BTCUSDT"]["baseline"]
                        btc_cumulative = ((btc_current - btc_baseline) / btc_baseline) * 100
                        if btc_cumulative <= 0:
                            message_text += " 🔵"
                    
                    print("Sending trigger notification:", message_text)
                    send_telegram_message(message_text)
                    cumulative_data[symbol]["baseline"] = current_price
                    cumulative_data[symbol]["direction"] = new_direction
                    cumulative_data[symbol]["baseline_time"] = datetime.datetime.now().strftime("%H:%M")
                    cumulative_data[symbol]["baseline_timestamp"] = now_ts
                    cumulative_data[symbol]["last_notif_timestamp"] = now_ts
    except Exception as e:
        print("Error in on_message:", e)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed:", close_status_code, close_msg)

def on_open(ws):
    print("WebSocket connection opened.")

def run_websocket(max_runtime=3600):
    """
    Constructs the combined WebSocket URL for all coin 1-minute kline streams and runs the connection.
    Uses a separate thread to allow for a graceful shutdown after max_runtime seconds.
    """
    streams = "/".join([coin.lower() + "@kline_1m" for coin in COINS])
    ws_url = f"wss://fstream.binance.com/stream?streams={streams}"
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)

    ws_thread = threading.Thread(target=ws.run_forever, kwargs={'ping_interval':20, 'ping_timeout':10})
    ws_thread.daemon = True
    ws_thread.start()

    start_time = time.time()
    while ws_thread.is_alive():
        if time.time() - start_time > max_runtime:
            print("Max runtime reached. Closing WebSocket...")
            ws.close()  # This should force ws.run_forever() to exit.
            break
        time.sleep(1)
    ws_thread.join()
    print("WebSocket thread closed. Exiting run_websocket.")

# ======================================
# Script Entry Point with Auto-Restart Every Hour
# ======================================

if __name__ == "__main__":
    try:
        print("Sending historical summary notifications...")
        send_historical_summaries()
        print("Starting real-time monitoring via WebSocket...")
        run_websocket(max_runtime=3600)  # Run for 1 hour
        print("One hour elapsed. Restarting the script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
