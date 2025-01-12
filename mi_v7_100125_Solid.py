import sys
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from binance import AsyncClient
from telegram import Bot
from aiohttp import ClientTimeout
from asyncio import Semaphore

# =========================
# 1. Event Loop Configuration (Windows)
# =========================
if sys.platform.startswith('win'):
    # Configure asyncio to use SelectorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =========================
# 2. Logging Configuration
# =========================
def setup_logging():
    """
    Configure logging settings.
    Logs are written to both a file and the console with a specified format.
    """
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more verbose output
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('futures_1_2_3m_fast.log', mode='a'),
            logging.StreamHandler()
        ]
    )

setup_logging()  # Initialize logging at the start

# =========================
# 3. Telegram Configuration
# =========================
class TelegramConfig:
    """
    Configuration class for Telegram bot credentials.
    Replace the placeholder strings with your actual Telegram Bot Token and Chat ID.
    """
    TELEGRAM_BOT_TOKEN = '7721789198:AAFQbDqk7Ln5O-O3eGwYh05MZMCdVunfMHI'  # Replace with your actual token
    TELEGRAM_CHAT_ID = '7052327528'      # Replace with your actual chat ID

# Initialize Telegram bot
bot = Bot(token=TelegramConfig.TELEGRAM_BOT_TOKEN)

# =========================
# 4. Global Parameters
# =========================
class GlobalParams:
    """
    Class to hold all global parameters and configurations.
    """
    TOP_GAINERS_LIMIT = 20              # Number of top FUTURES gainers to track
    TOP_GAINERS_UPDATE_INTERVAL = 180   # 3 minutes in seconds
    POLL_INTERVAL = 15                  # 15-second polling interval
    PRICE_CHANGE_THRESHOLD = 0.5        # 0.5% immediate price change
    
    # Timeframes in minutes and their corresponding Binance intervals
    TIMEFRAMES = [1, 2, 3]
    BINANCE_INTERVALS = {
        1: '1m',
        2: '2m',  # Note: Binance does not support 2m; we'll emulate it
        3: '3m'
    }
    
    MAX_CONCURRENT_FETCHES = 5          # Maximum number of concurrent klines fetches

# =========================
# 5. Data Structures
# =========================
data_frames = {}          # Structure: data_frames[symbol][timeframe] => DataFrame
TOKENS = []               # List to hold top gainers symbols
alerted_tokens = set()    # Set to keep track of symbols already alerted
last_gainers_update = datetime.now(timezone.utc) - timedelta(minutes=5)  # Initialize last update time

# =========================
# 6. Data Processing Functions
# =========================
def aggregate_2m_from_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Combine pairs of consecutive 1-minute bars into 2-minute bars.

    Parameters:
        df_1m (pd.DataFrame): DataFrame containing 1-minute klines.

    Returns:
        pd.DataFrame: Aggregated 2-minute klines.
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

def calculate_ma_and_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Moving Averages (MA) and MACD indicators.

    Parameters:
        df (pd.DataFrame): DataFrame containing klines.

    Returns:
        pd.DataFrame: DataFrame with MA and MACD indicators added.
    """
    if len(df) < 2:
        return df
    # Moving Averages
    df['MA3'] = df['close'].rolling(window=3).mean()
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA25'] = df['close'].rolling(window=25).mean()

    if len(df) >= 26:
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def calculate_confidence(df: pd.DataFrame) -> float:
    """
    Calculate the confidence level for buy/sell signals based on MA and MACD indicators.

    Parameters:
        df (pd.DataFrame): DataFrame containing klines with MA and MACD indicators.

    Returns:
        float: Confidence level scaled between 0 and 100.
    """
    if len(df) < 2:
        return 50.0

    bullish_score = 0
    bearish_score = 0

    # Moving Averages
    if all(col in df.columns for col in ['MA3', 'MA7', 'MA25']):
        ma3 = df['MA3'].iloc[-1]
        ma7 = df['MA7'].iloc[-1]
        ma25 = df['MA25'].iloc[-1]
        if pd.notna(ma3) and pd.notna(ma7) and pd.notna(ma25):
            if ma3 > ma7 > ma25:
                bullish_score += 3
            elif ma3 < ma7 < ma25:
                bearish_score += 3

    # MACD
    if all(col in df.columns for col in ['MACD', 'Signal_Line']):
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

    confidence = (bullish_score / total) * 100.0
    return round(confidence, 2)

def detect_candlestick_patterns(df: pd.DataFrame) -> list:
    """
    Detect specific candlestick patterns in the latest two klines.

    Parameters:
        df (pd.DataFrame): DataFrame containing klines.

    Returns:
        list: List of detected candlestick pattern alerts.
    """
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

def detect_volume_spike(df: pd.DataFrame) -> list:
    """
    Detect a volume spike where the current volume is more than twice the average of the last 3 periods.

    Parameters:
        df (pd.DataFrame): DataFrame containing klines.

    Returns:
        list: List containing a volume spike alert if detected.
    """
    alerts = []
    if len(df) < 3:
        return alerts

    latest = df.iloc[-1]
    avg_win = 3
    avg_volume = df['volume'].rolling(window=avg_win).mean().iloc[-1]
    if latest['volume'] > 2 * avg_volume:
        alerts.append("Volume Spike detected!")
    return alerts

def immediate_price_change_alert(df: pd.DataFrame, threshold=0.5) -> tuple:
    """
    Check for an immediate price change exceeding the defined threshold.

    Parameters:
        df (pd.DataFrame): DataFrame containing klines.
        threshold (float): Percentage threshold for price change.

    Returns:
        tuple: (bool indicating if threshold exceeded, percentage change)
    """
    if len(df) < 2:
        return (False, 0.0)

    latest_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    if prev_close == 0:
        return (False, 0.0)
    pct_change = ((latest_close - prev_close) / prev_close) * 100.0
    if abs(pct_change) >= threshold:
        return (True, pct_change)
    return (False, pct_change)

# =========================
# 7. Utility Functions
# =========================
def format_buy_sell_message(symbol: str, timeframe: int, price_now: float, direction: str, confidence: float, pct_change: float = 0.0) -> str:
    """
    Format the buy/sell message to be sent via Telegram.

    Parameters:
        symbol (str): Trading symbol (e.g., BTCUSDT).
        timeframe (int): Timeframe in minutes.
        price_now (float): Current price.
        direction (str): 'buy' or 'sell'.
        confidence (float): Confidence level percentage.
        pct_change (float, optional): Price change percentage. Defaults to 0.0.

    Returns:
        str: Formatted message string.
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

async def send_telegram_message(text: str):
    """
    Send a message via Telegram bot.

    Parameters:
        text (str): Message text.
    """
    try:
        # Remove emojis from log messages to avoid encoding issues
        log_text = text.replace("🟢B", "[BUY]").replace("🔴S", "[SELL]")
        logging.info(f"Sending Telegram message: {log_text}")  # Log without emojis
        await bot.send_message(chat_id=TelegramConfig.TELEGRAM_CHAT_ID, text=text)  # Send with emojis
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

# =========================
# 8. Data Fetching Functions
# =========================
async def fetch_futures_top_gainers(client: AsyncClient, spot_tickers: list, limit=20) -> list:
    """
    Fetch the top gainers from Binance Futures based on 24-hour price change percentage
    and ensure they are also active in the Spot market.

    Parameters:
        client (AsyncClient): Instance of Binance AsyncClient.
        spot_tickers (list): List of Spot ticker dictionaries.
        limit (int, optional): Number of top gainers to fetch. Defaults to 20.

    Returns:
        list: List of top gainers symbols that are active in both Spot and Futures.
    """
    try:
        logging.info("Fetching top gainers from Binance FUTURES (24h).")
        tickers = await client.futures_ticker()
        df_tickers = pd.DataFrame(tickers)
        df_tickers = df_tickers[df_tickers['symbol'].str.endswith('USDT')]
        df_tickers['priceChangePercent'] = pd.to_numeric(df_tickers['priceChangePercent'], errors='coerce')
        df_tickers.dropna(subset=['priceChangePercent'], inplace=True)
        df_tickers.sort_values(by='priceChangePercent', ascending=False, inplace=True)

        # Fetch exchange info to get list of valid Futures symbols
        exchange_info = await client.futures_exchange_info()
        futures_symbols = set(symbol['symbol'] for symbol in exchange_info['symbols'])

        # Extract Spot USDT symbols
        spot_usdt_symbols = set(ticker['symbol'] for ticker in spot_tickers if ticker['symbol'].endswith('USDT'))

        # Intersection: Symbols that are in Futures exchange info and in Spot USDT symbols
        valid_symbols = spot_usdt_symbols.intersection(futures_symbols)

        # Filter top gainers that are also in Spot and have Futures contracts
        df_top = df_tickers[df_tickers['symbol'].isin(valid_symbols)].head(limit)
        top_symbols = df_top['symbol'].tolist()
        logging.info(f"Top Gainers (FUTURES & SPOT): {top_symbols}")
        return top_symbols
    except Exception as e:
        logging.error(f"Error fetching top gainers: {e}")
        return []

async def fetch_futures_klines(symbol: str, interval_str: str, client: AsyncClient, limit=50, retries=3, backoff_factor=2) -> pd.DataFrame:
    """
    Fetch klines (candlestick data) from Binance Futures for a given symbol and interval with retry logic.
    
    Parameters:
        symbol (str): Trading symbol (e.g., BTCUSDT).
        interval_str (str): Kline interval (e.g., '1m', '3m').
        client (AsyncClient): Instance of Binance AsyncClient.
        limit (int, optional): Number of klines to fetch. Defaults to 50.
        retries (int, optional): Number of retry attempts. Defaults to 3.
        backoff_factor (int, optional): Factor by which the wait time increases after each retry. Defaults to 2.
    
    Returns:
        pd.DataFrame: DataFrame containing fetched klines or empty DataFrame on failure.
    """
    for attempt in range(1, retries + 1):
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
        except asyncio.TimeoutError:
            logging.warning(f"TimeoutError on attempt {attempt} fetching klines for {symbol} at {interval_str}m.")
        except Exception as e:
            logging.exception(f"Error fetching {interval_str} futures klines for {symbol}: {e}")
        
        if attempt < retries:
            wait_time = backoff_factor ** attempt
            logging.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    logging.error(f"Failed to fetch klines for {symbol} after {retries} attempts.")
    return pd.DataFrame()

# =========================
# 9. Analyze Symbol Function
# =========================
async def analyze_symbol(symbol: str, client: AsyncClient, bot: Bot, chat_id: str, alerted_tokens: set, semaphore: Semaphore):
    """
    Analyze a single symbol for trading signals and send alerts if conditions are met.

    Parameters:
        symbol (str): Trading symbol (e.g., BTCUSDT).
        client (AsyncClient): Instance of Binance AsyncClient.
        bot (Bot): Instance of Telegram Bot.
        chat_id (str): Telegram chat ID to send alerts to.
        alerted_tokens (set): Set of symbols already alerted to prevent duplicates.
        semaphore (Semaphore): Semaphore to limit concurrent fetches.
    """
    async with semaphore:
        try:
            # Initialize data_frames for the symbol if not present
            if symbol not in data_frames:
                data_frames[symbol] = {}

            # Fetch 1m and 3m klines
            df_1m = await fetch_futures_klines(symbol, GlobalParams.BINANCE_INTERVALS[1], client, limit=50)
            df_3m = await fetch_futures_klines(symbol, GlobalParams.BINANCE_INTERVALS[3], client, limit=50)

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
            for timeframe in GlobalParams.TIMEFRAMES:
                if timeframe not in GlobalParams.BINANCE_INTERVALS:
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
                immediate_change, pct_change = immediate_price_change_alert(df, threshold=GlobalParams.PRICE_CHANGE_THRESHOLD)

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
                direction = None
                if confidence > 70 and 'Hammer detected!' in alerts:
                    direction = 'buy'
                elif confidence < 30 and 'Bearish Engulfing detected!' in alerts:
                    direction = 'sell'
                elif immediate_change:
                    direction = 'buy' if pct_change > 0 else 'sell'

                if direction and symbol not in alerted_tokens:
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
# 10. Main Polling Loop
# =========================
async def main_loop(client: AsyncClient, bot: Bot, chat_id: str):
    """
    The main polling loop that periodically fetches top gainers and analyzes each symbol.

    Parameters:
        client (AsyncClient): Instance of Binance AsyncClient.
        bot (Bot): Instance of Telegram Bot.
        chat_id (str): Telegram chat ID to send alerts to.
    """
    global TOKENS, last_gainers_update

    semaphore = Semaphore(GlobalParams.MAX_CONCURRENT_FETCHES)  # Limit concurrent fetches

    while True:
        now = datetime.now(timezone.utc)

        # Update top gainers every TOP_GAINERS_UPDATE_INTERVAL seconds
        if (now - last_gainers_update).total_seconds() >= GlobalParams.TOP_GAINERS_UPDATE_INTERVAL:
            # Fetch tickers to pass spot_tickers to the gainers function
            futures_tickers, spot_tickers = await fetch_tickers(client)
            if not futures_tickers or not spot_tickers:
                logging.error("Failed to fetch tickers from Binance.")
                await asyncio.sleep(GlobalParams.POLL_INTERVAL)
                continue

            new_tokens = await fetch_futures_top_gainers(client, spot_tickers, limit=GlobalParams.TOP_GAINERS_LIMIT)
            if new_tokens:
                TOKENS = new_tokens
            last_gainers_update = now

        if not TOKENS:
            logging.warning("No FUTURES tokens in top gainers. Sleeping, then retrying.")
            await asyncio.sleep(GlobalParams.POLL_INTERVAL)
            continue

        tasks = []
        for symbol in TOKENS:
            if symbol in alerted_tokens:
                continue
            tasks.append(analyze_symbol(symbol, client, bot, chat_id, alerted_tokens, semaphore))
        await asyncio.gather(*tasks)

        logging.info(f"Completed cycle for {len(TOKENS)} FUTURES tokens. Sleeping {GlobalParams.POLL_INTERVAL}s...")
        await asyncio.sleep(GlobalParams.POLL_INTERVAL)

# =========================
# 11. Fetch Tickers Function
# =========================
async def fetch_tickers(client: AsyncClient):
    """
    Fetches Futures and Spot tickers from Binance.

    Parameters:
        client (AsyncClient): An instance of Binance AsyncClient.

    Returns:
        tuple: A tuple containing two lists - futures_tickers and spot_tickers.
    """
    try:
        logging.info("Fetching Binance Futures tickers.")
        futures_tickers = await client.futures_ticker()

        logging.info("Fetching Binance Spot tickers.")
        spot_tickers = await client.get_all_tickers()

        logging.info(f"Fetched {len(futures_tickers)} Futures tickers and {len(spot_tickers)} Spot tickers.")
        return futures_tickers, spot_tickers
    except Exception as e:
        logging.error(f"Error fetching tickers: {e}")
        return [], []

# =========================
# 12. Main Function
# =========================
async def main():
    """
    The main function that initializes the Binance client and starts the polling loop.
    """
    # Initialize Binance client with increased timeout
    timeout = ClientTimeout(total=60)  # 60 seconds total timeout
    client = await AsyncClient.create(
        api_key=None, 
        api_secret=None, 
        requests_params={'timeout': timeout}  # Pass timeout inside requests_params
    )  # Add your API keys if needed
    try:
        await main_loop(client, bot, TelegramConfig.TELEGRAM_CHAT_ID)
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
    finally:
        await client.close_connection()
        logging.info("Binance client connection closed.")

# =========================
# 13. Entry Point
# =========================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script terminated by user.")
    except Exception as e:
        logging.error(f"Unexpected error in __main__: {e}")
