import asyncio
import sys
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from binance import AsyncClient
from binance.client import Client
from telegram import Bot

# =========================
# 1. Force Windows Selector Event Loop (on Windows)
# =========================
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =========================
# 2. Logging Configuration
# =========================
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('futures_1_2_3m_fast.log', mode='a'),
        logging.StreamHandler()
    ]
)

# =========================
# 3. Telegram Configuration
# =========================
TELEGRAM_BOT_TOKEN = '7721789198:AAFQbDqk7Ln5O-O3eGwYh05MZMCdVunfMHI'  # Replace with your actual token
TELEGRAM_CHAT_ID = '7052327528'      # Replace with your actual chat ID
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# =========================
# 4. Global Parameters
# =========================
TOP_GAINERS_LIMIT = 20              # Number of top FUTURES gainers to track
TOP_GAINERS_UPDATE_INTERVAL = 180   # 3 minutes
POLL_INTERVAL = 15                  # 15-second polling interval
PRICE_CHANGE_THRESHOLD = 0.5        # 0.5% immediate price change

# Timeframes in minutes and their corresponding Binance intervals
TIMEFRAMES = [1, 2, 3]
BINANCE_INTERVALS = {
    1: '1m',
    2: '2m',  # Note: Binance does not support 2m; we'll emulate it
    3: '3m'
}

# Data structure: data_frames[symbol][timeframe] => DataFrame
data_frames = {}
TOKENS = []
last_gainers_update = datetime.now(timezone.utc) - timedelta(minutes=5)

# Once triggered for a token, we skip all future alerts
alerted_tokens = set()
client: AsyncClient = None

# =========================
# 5. Aggregation for 2-Min Bars
# =========================
def aggregate_2m_from_1m(df_1m: pd.DataFrame):
    """
    Combine pairs of consecutive 1-min bars -> 2-min bars.
    - timestamp => second bar's
    - open => first bar's open
    - high => max
    - low => min
    - close => second bar's close
    - volume => sum
    """
    if len(df_1m) < 2:
        return pd.DataFrame()

    df_1m = df_1m.sort_values('timestamp').reset_index(drop=True).copy()
    df_1m['group_id'] = df_1m.index // 2

    agg = df_1m.groupby('group_id').agg({
        'timestamp': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)

    agg.sort_values('timestamp', inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg

# =========================
# 6. MA + MACD Calculation
# =========================
def calculate_ma_and_macd(df: pd.DataFrame):
    if len(df) < 2:
        return df
    # MAs
    df['MA3'] = df['close'].rolling(window=3).mean()
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA25'] = df['close'].rolling(window=25).mean()

    if len(df) >= 26:
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

# =========================
# 7. Confidence Calculation (Weighted)
# =========================
def calculate_confidence(df: pd.DataFrame):
    """
    Weighted approach:
      - MAs bullish => +3, MAs bearish => +3
      - MACD above signal => +5 bullish, else +5 bearish
    total => 8 possible
    scale => 0..100
    """
    if len(df) < 2:
        return 50.0

    bullish_score = 0
    bearish_score = 0

    # MAs
    if 'MA3' in df.columns and 'MA7' in df.columns and 'MA25' in df.columns:
        ma3 = df['MA3'].iloc[-1]
        ma7 = df['MA7'].iloc[-1]
        ma25 = df['MA25'].iloc[-1]
        if pd.notna(ma3) and pd.notna(ma7) and pd.notna(ma25):
            if ma3 > ma7 > ma25:
                bullish_score += 3
            elif ma3 < ma7 < ma25:
                bearish_score += 3

    # MACD
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        macd_val = df['MACD'].iloc[-1]
        sig_val = df['Signal_Line'].iloc[-1]
        if pd.notna(macd_val) and pd.notna(sig_val):
            if macd_val > sig_val:
                bullish_score += 5
            else:
                bearish_score += 5

    total = bullish_score + bearish_score
    if total == 0:
        return 50.0

    conf = (bullish_score / total) * 100.0
    return round(conf, 2)

# =========================
# 8. Candlestick Patterns
# =========================
def detect_candlestick_patterns(df: pd.DataFrame):
    alerts = []
    if len(df) < 2:
        return alerts

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    body = latest['close'] - latest['open']
    lower_wick = latest['open'] - latest['low'] if body > 0 else latest['close'] - latest['low']
    upper_wick = latest['high'] - latest['close'] if body > 0 else latest['high'] - latest['open']

    # Hammer (Bullish)
    if body > 0 and lower_wick > 2 * abs(body) and (latest['high'] - max(latest['close'], latest['open'])) < 0.1 * abs(body):
        alerts.append("Hammer detected!")

    # Bearish Engulfing
    if (
        body < 0
        and prev['close'] > prev['open']
        and latest['open'] > prev['close']
        and latest['close'] < prev['open']
    ):
        alerts.append("Bearish Engulfing detected!")

    return alerts

# =========================
# 9. Volume Spike
# =========================
def detect_volume_spike(df: pd.DataFrame):
    alerts = []
    if len(df) < 3:
        return alerts

    latest = df.iloc[-1]
    avg_win = 3
    avg_volume = df['volume'].rolling(window=avg_win).mean().iloc[-1]
    if latest['volume'] > 2 * avg_volume:
        alerts.append("Volume Spike detected!")
    return alerts

# =========================
# 10. Immediate 0.5% Change
# =========================
def immediate_price_change_alert(df: pd.DataFrame):
    if len(df) < 2:
        return (False, 0.0)

    latest_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    if prev_close == 0:
        return (False, 0.0)
    pct_change = ((latest_close - prev_close) / prev_close) * 100.0
    if abs(pct_change) >= PRICE_CHANGE_THRESHOLD:
        return (True, pct_change)
    return (False, pct_change)

# =========================
# 11. Telegram Sending
# =========================
async def send_telegram_message(text: str):
    try:
        # Remove emojis from log messages to avoid encoding issues
        log_text = text.replace("🟢B", "[BUY]").replace("🔴S", "[SELL]")
        logging.info(f"Sending Telegram message: {log_text}")  # Log without emojis
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)  # Send with emojis
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

# =========================
# 12. Fetch 1m or 3m klines from Futures
# =========================
async def fetch_futures_klines(symbol: str, interval_str: str, limit=50):
    try:
        klines = await client.futures_klines(
            symbol=symbol,
            interval=interval_str,
            limit=limit
        )
        rows = []
        for k in klines:
            open_t = pd.to_datetime(k[0], unit='ms')
            open_p = float(k[1])
            high_p = float(k[2])
            low_p = float(k[3])
            close_p = float(k[4])
            vol = float(k[5])
            rows.append({
                'timestamp': open_t,
                'open': open_p,
                'high': high_p,
                'low': low_p,
                'close': close_p,
                'volume': vol
            })
        df = pd.DataFrame(rows)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching {interval_str} futures klines for {symbol}: {e}")
        return pd.DataFrame()

# =========================
# 13. Format Buy/Sell Message
# =========================
def format_buy_sell_message(symbol: str, timeframe: int, price_now: float, direction: str, confidence: float, pct_change: float = 0.0):
    """
    direction => 'buy' => 🟢B, 'sell' => 🔴S
    Include 'Price Change: XX%' in the message
    """
    if direction == 'buy':
        top_line = "🟢B"
    else:
        top_line = "🔴S"

    msg = (
        f"{top_line}\n"
        f"{symbol}\n"
        f"Timeframe: {timeframe}m\n"
        f"Confidence Level: {confidence:.2f}%\n"
        f"Price Change: {pct_change:.2f}%\n"
        f"Current Price: {price_now}"
    )
    return msg

# =========================
# 14. Fetch Futures Ticker for Top Gainers
# =========================
async def fetch_futures_top_gainers(client: AsyncClient, limit=20):
    try:
        logging.info("Fetching top gainers from Binance FUTURES (24h).")
        tickers = await client.futures_ticker()
        df_tickers = pd.DataFrame(tickers)
        df_tickers = df_tickers[df_tickers['symbol'].str.endswith('USDT')]
        df_tickers['priceChangePercent'] = pd.to_numeric(df_tickers['priceChangePercent'], errors='coerce')
        df_tickers.dropna(subset=['priceChangePercent'], inplace=True)
        df_tickers.sort_values(by='priceChangePercent', ascending=False, inplace=True)

        top_n = df_tickers.head(limit)
        top_symbols = top_n['symbol'].tolist()
        logging.info(f"Top Gainers (FUTURES): {top_symbols}")
        return top_symbols
    except Exception as e:
        logging.error(f"Error fetching FUTURES top gainers: {e}")
        return []

# =========================
# 15. Analyze Symbol
# =========================
async def analyze_symbol(symbol: str):
    """
    Analyze a single symbol for trading signals.
    """
    try:
        # Initialize data_frames for the symbol if not present
        if symbol not in data_frames:
            data_frames[symbol] = {}

        # Fetch 1m and 3m klines
        df_1m = await fetch_futures_klines(symbol, BINANCE_INTERVALS[1], limit=50)
        df_3m = await fetch_futures_klines(symbol, BINANCE_INTERVALS[3], limit=50)

        if df_1m.empty or df_3m.empty:
            logging.warning(f"No data fetched for {symbol}. Skipping analysis.")
            return

        # Aggregate 2m data from 1m
        df_2m = aggregate_2m_from_1m(df_1m)

        # Update data_frames
        data_frames[symbol][1] = df_1m
        data_frames[symbol][2] = df_2m
        data_frames[symbol][3] = df_3m

        # Analyze each timeframe
        for timeframe in TIMEFRAMES:
            if timeframe not in BINANCE_INTERVALS:
                logging.warning(f"Timeframe {timeframe}m not supported. Skipping.")
                continue

            df = data_frames[symbol].get(timeframe)
            if df is None or df.empty:
                logging.warning(f"No data for {symbol} at {timeframe}m. Skipping.")
                continue

            # Calculate MA and MACD
            df = calculate_ma_and_macd(df)
            data_frames[symbol][timeframe] = df

            # Calculate confidence
            confidence = calculate_confidence(df)

            # Detect patterns
            candlestick_alerts = detect_candlestick_patterns(df)
            volume_spike_alerts = detect_volume_spike(df)
            immediate_change, pct_change = immediate_price_change_alert(df)

            # Current price
            current_price = df['close'].iloc[-1]

            alerts = []
            if candlestick_alerts:
                alerts.extend(candlestick_alerts)
            if volume_spike_alerts:
                alerts.extend(volume_spike_alerts)
            if immediate_change:
                alerts.append(f"Immediate Price Change: {pct_change:.2f}%")

            # Determine if a buy or sell signal should be sent based on confidence
            # You can adjust the thresholds as needed
            if confidence > 70 and 'Hammer detected!' in alerts:
                direction = 'buy'
            elif confidence < 30 and 'Bearish Engulfing detected!' in alerts:
                direction = 'sell'
            elif immediate_change:
                direction = 'buy' if pct_change > 0 else 'sell'
            else:
                direction = None

            if direction:
                message = format_buy_sell_message(
                    symbol=symbol,
                    timeframe=timeframe,
                    price_now=current_price,
                    direction=direction,
                    confidence=confidence,
                    pct_change=pct_change
                )
                await send_telegram_message(message)
                alerted_tokens.add(symbol)
                logging.info(f"Alert sent for {symbol}: {direction.upper()}")

                # Once an alert is sent, skip further alerts for this token
                break

    except Exception as e:
        logging.error(f"Error analyzing {symbol}: {e}")

# =========================
# 16. Main Polling Loop
# =========================
async def main_loop():
    global TOKENS, last_gainers_update

    while True:
        now = datetime.now(timezone.utc)

        # Update top gainers every 3 minutes
        if (now - last_gainers_update).total_seconds() >= TOP_GAINERS_UPDATE_INTERVAL:
            new_tokens = await fetch_futures_top_gainers(client, limit=TOP_GAINERS_LIMIT)
            if new_tokens:
                TOKENS = new_tokens
            last_gainers_update = now

        if not TOKENS:
            logging.warning("No FUTURES tokens in top gainers. Sleeping, then retrying.")
            await asyncio.sleep(POLL_INTERVAL)
            continue

        tasks = []
        for symbol in TOKENS:
            if symbol in alerted_tokens:
                continue
            tasks.append(analyze_symbol(symbol))
        await asyncio.gather(*tasks)

        logging.info(f"Completed cycle for {len(TOKENS)} FUTURES tokens. Sleeping {POLL_INTERVAL}s...")
        await asyncio.sleep(POLL_INTERVAL)

# =========================
# 17. Main Function
# =========================
async def main():
    global client
    client = await AsyncClient.create(api_key=None, api_secret=None)  # Add your API keys if needed
    try:
        await main_loop()
    finally:
        await client.close_connection()

# =========================
# 18. Entry Point
# =========================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script terminated by user.")
    except Exception as e:
        logging.error(f"Unexpected error in __main__: {e}")
