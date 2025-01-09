import sys
import asyncio
import logging
import json
import sqlite3
import signal
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import psutil  # Added psutil for memory monitoring

from binance import AsyncClient
from telegram import Bot

# =========================
# 1. Sentry Integration
# =========================
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration

# Initialize Sentry with adjusted sampling rate and fingerprinting
sentry_logging = LoggingIntegration(
    level=logging.INFO,        # Capture info and above as breadcrumbs
    event_level=logging.ERROR  # Send errors as events
)

sentry_sdk.init(
    dsn="https://d6324dff11af3c2f1e2aa160fd9efa6c@o4508608695566336.ingest.us.sentry.io/4508608760643584",  # Replace with your actual Sentry DSN
    integrations=[sentry_logging, AsyncioIntegration()],
    traces_sample_rate=0.2  # Capture 20% of transactions to reduce load
)

# =========================
# 2. Event Loop Configuration (Windows)
# =========================
if sys.platform.startswith('win'):
    # Configure asyncio to use SelectorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =========================
# 3. Structured Logging Configuration
# =========================
class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    def format(self, record):
        log_record = {
            'time': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'funcName': record.funcName,
        }
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging():
    """
    Configure logging with JSONFormatter for structured logs.
    Logs are written to both a file and the console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose output

    # File Handler
    fh = logging.FileHandler('futures_gainers_losers.log', mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(JSONFormatter())
    logger.addHandler(ch)

setup_logging()  # Initialize logging at the start

# =========================
# 4. Telegram Configuration
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
# 5. Global Parameters
# =========================
class GlobalParams:
    """
    Class to hold all global parameters and configurations.
    """
    TOP_GAINERS_LIMIT = 30              # Number of top FUTURES gainers to track
    TOP_LOSERS_LIMIT = 30               # Number of top FUTURES losers to track
    TOP_GAINERS_UPDATE_INTERVAL = 180   # 3 minutes in seconds
    POLL_INTERVAL = 15                  # 15-second polling interval
    PRICE_CHANGE_THRESHOLD = 0.5        # 0.5% immediate price change

    # Timeframes in minutes and their corresponding Binance intervals
    TIMEFRAMES = [1, 2]
    BINANCE_INTERVALS = {
        1: '1m'
    }

# =========================
# 6. SQLite Database Setup (Enhanced)
# =========================
class AlertedTokensDB:
    """
    Class to handle SQLite operations for persisting alerted tokens,
    trades, and indicators.
    """
    def __init__(self, db_path='alerted_tokens.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.initialize_tables()

    def initialize_tables(self):
        """
        Create the necessary tables if they don't exist.
        """
        try:
            # Existing alerted_tokens table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerted_tokens (
                    symbol TEXT PRIMARY KEY
                )
            ''')
            
            # New trades table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe INTEGER,
                    direction TEXT,
                    confidence REAL,
                    price_change REAL,
                    current_price REAL,
                    timestamp TEXT
                )
            ''')

            # New indicators table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe INTEGER,
                    timestamp TEXT,
                    MA3 REAL,
                    MA7 REAL,
                    MA25 REAL,
                    EMA12 REAL,
                    EMA26 REAL,
                    MACD REAL,
                    Signal_Line REAL,
                    confidence REAL
                )
            ''')

            self.conn.commit()
            logging.info("SQLite database tables initialized.")
        except Exception as e:
            logging.error(f"Error initializing SQLite tables: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)  # Capture exception in Sentry

    def add_alerted_token(self, symbol: str):
        """
        Add a symbol to the alerted_tokens table.
        """
        try:
            self.cursor.execute('INSERT OR IGNORE INTO alerted_tokens (symbol) VALUES (?)', (symbol,))
            self.conn.commit()
            logging.info(f"Symbol {symbol} added to alerted_tokens.")
        except Exception as e:
            logging.error(f"Error adding symbol {symbol} to alerted_tokens: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)  # Capture exception in Sentry

    def is_token_alerted(self, symbol: str) -> bool:
        """
        Check if a symbol has already been alerted.
        """
        try:
            self.cursor.execute('SELECT symbol FROM alerted_tokens WHERE symbol = ?', (symbol,))
            return self.cursor.fetchone() is not None
        except Exception as e:
            logging.error(f"Error checking if symbol {symbol} is alerted: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)  # Capture exception in Sentry
            return False

    def load_alerted_tokens(self) -> set:
        """
        Load all alerted tokens from the database into a set.
        """
        try:
            self.cursor.execute('SELECT symbol FROM alerted_tokens')
            rows = self.cursor.fetchall()
            return set(row[0] for row in rows)
        except Exception as e:
            logging.error(f"Error loading alerted tokens: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)  # Capture exception in Sentry
            return set()

    def log_trade(self, symbol: str, timeframe: int, direction: str, confidence: float, price_change: float, current_price: float, timestamp: str):
        """
        Log a trade signal to the trades table.
        """
        try:
            self.cursor.execute('''
                INSERT INTO trades (symbol, timeframe, direction, confidence, price_change, current_price, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timeframe, direction, confidence, price_change, current_price, timestamp))
            self.conn.commit()
            logging.info(f"Trade logged for {symbol} at {timeframe}m: {direction.upper()}")
        except Exception as e:
            logging.error(f"Error logging trade for {symbol}: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)

    def log_indicator(self, symbol: str, timeframe: int, timestamp: str, indicators: dict, confidence: float):
        """
        Log indicator values to the indicators table.
        """
        try:
            self.cursor.execute('''
                INSERT INTO indicators (
                    symbol, timeframe, timestamp, MA3, MA7, MA25, EMA12, EMA26, MACD, Signal_Line, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                timeframe,
                timestamp,
                indicators.get('MA3'),
                indicators.get('MA7'),
                indicators.get('MA25'),
                indicators.get('EMA12'),
                indicators.get('EMA26'),
                indicators.get('MACD'),
                indicators.get('Signal_Line'),
                confidence
            ))
            self.conn.commit()
            logging.info(f"Indicators logged for {symbol} at {timeframe}m.")
        except Exception as e:
            logging.error(f"Error logging indicators for {symbol}: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)  # Capture exception in Sentry

    def close(self):
        """
        Close the SQLite connection.
        """
        try:
            self.conn.close()
            logging.info("SQLite database connection closed.")
        except Exception as e:
            logging.error(f"Error closing SQLite database: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)  # Capture exception in Sentry

# =========================
# 7. Data Structures
# =========================
data_frames = {}  # Structure: data_frames[symbol][timeframe] => DataFrame
TOKENS = []       # List to hold top gainers and losers symbols

# =========================
# 8. Data Processing Functions
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
# 9. Utility Functions
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

# Implemented Telegram message queue and rate limiter
telegram_queue = asyncio.Queue()
telegram_semaphore = asyncio.Semaphore(30)  # Added semaphore for rate limiting

async def telegram_worker():
    """
    Worker coroutine to send Telegram messages from the queue at a controlled rate.
    """
    while True:
        message = await telegram_queue.get()
        await send_telegram_message(message)
        telegram_queue.task_done()

async def send_telegram_message(text: str, retries=3, backoff=1):
    """
    Send a message via Telegram bot with retry mechanism and rate limiting.

    Parameters:
        text (str): Message text.
        retries (int): Number of retry attempts.
        backoff (int): Initial backoff time in seconds.
    """
    async with telegram_semaphore:
        for attempt in range(1, retries + 1):
            try:
                # Remove emojis from log messages to avoid encoding issues
                log_text = text.replace("🟢B", "[BUY]").replace("🔴S", "[SELL]")
                logging.info(f"Sending Telegram message: {log_text}")  # Log without emojis
                await bot.send_message(chat_id=TelegramConfig.TELEGRAM_CHAT_ID, text=text)  # Send with emojis
                return  # Success, exit the function
            except Exception as e:
                logging.error(f"Attempt {attempt} - Error sending Telegram message: {e}", exc_info=True)
                if attempt < retries:
                    wait_time = backoff * (2 ** (attempt - 1))  # Exponential backoff
                    logging.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"All {retries} attempts failed to send Telegram message.")
                    sentry_sdk.capture_exception(e, fingerprint=["send_telegram_message"])

# =========================
# 10. Data Fetching Functions
# =========================
async def fetch_futures_top_symbols(client: AsyncClient, gainers_limit=30, losers_limit=30, retries=3, backoff=1) -> list:
    """
    Fetch the top gainers and losers from Binance Futures based on 24-hour price change percentage with retry.

    Parameters:
        client (AsyncClient): Instance of Binance AsyncClient.
        gainers_limit (int, optional): Number of top gainers to fetch. Defaults to 30.
        losers_limit (int, optional): Number of top losers to fetch. Defaults to 30.
        retries (int): Number of retry attempts.
        backoff (int): Initial backoff time in seconds.

    Returns:
        list: List of top gainers and losers symbols.
    """
    for attempt in range(1, retries + 1):
        try:
            logging.info("Fetching top gainers and losers from Binance FUTURES (24h).")
            tickers = await client.futures_ticker()
            df_tickers = pd.DataFrame(tickers)
            df_tickers = df_tickers[df_tickers['symbol'].str.endswith('USDT')]  # Ensure USDT-paired
            df_tickers['priceChangePercent'] = pd.to_numeric(df_tickers['priceChangePercent'], errors='coerce')
            df_tickers.dropna(subset=['priceChangePercent'], inplace=True)
            df_tickers.sort_values(by='priceChangePercent', ascending=False, inplace=True)

            top_gainers = df_tickers.head(gainers_limit)['symbol'].tolist()
            top_losers = df_tickers.tail(losers_limit)['symbol'].tolist()
            top_symbols = list(set(top_gainers + top_losers))  # Combine and remove duplicates
            logging.info(f"Top Gainers: {top_gainers}")
            logging.info(f"Top Losers: {top_losers}")
            return top_symbols
        except Exception as e:
            logging.error(f"Attempt {attempt} - Error fetching top symbols: {e}", exc_info=True)
            if attempt < retries:
                wait_time = backoff * (2 ** (attempt - 1))
                logging.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"All {retries} attempts failed to fetch top symbols.")
                sentry_sdk.capture_exception(e, fingerprint=["fetch_futures_top_symbols"])
                return []

async def fetch_futures_klines(symbol: str, interval_str: str, client: AsyncClient, limit=50, retries=3, backoff=1) -> pd.DataFrame:
    """
    Fetch klines (candlestick data) from Binance Futures for a given symbol and interval with retry.

    Parameters:
        symbol (str): Trading symbol (e.g., BTCUSDT).
        interval_str (str): Kline interval (e.g., '1m', '3m').
        client (AsyncClient): Instance of Binance AsyncClient.
        limit (int, optional): Number of klines to fetch. Defaults to 50.
        retries (int): Number of retry attempts.
        backoff (int): Initial backoff time in seconds.

    Returns:
        pd.DataFrame: DataFrame containing fetched klines.
    """
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Fetching {interval_str} futures klines for {symbol}.")
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
            logging.error(f"Attempt {attempt} - Error fetching {interval_str} futures klines for {symbol}: {e}", exc_info=True)
            if attempt < retries:
                wait_time = backoff * (2 ** (attempt - 1))
                logging.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"All {retries} attempts failed to fetch klines for {symbol} at {interval_str}.")
                sentry_sdk.capture_exception(e, fingerprint=["fetch_futures_klines", symbol, interval_str])
                return pd.DataFrame()

# =========================
# 11. Analyze Symbols in Batch (Enhanced)
# =========================
async def analyze_symbols_batch(symbols: list, client: AsyncClient, bot: Bot, chat_id: str, alerted_tokens: set, db: AlertedTokensDB):
    """
    Analyze multiple symbols in batch for trading signals and send alerts if conditions are met.
    
    Additionally, log detailed information about indicators and trades to the database.
    """
    try:
        # Fetch klines for all symbols and supported intervals in parallel
        fetch_tasks = []
        symbol_timeframe_map = {}  # To track which symbol and timeframe each task corresponds to

        for symbol in symbols:
            if db.is_token_alerted(symbol):
                logging.info(f"Symbol {symbol} already alerted. Skipping.")
                continue
            for timeframe in GlobalParams.TIMEFRAMES:
                if timeframe not in GlobalParams.BINANCE_INTERVALS:
                    continue  # Skip unsupported timeframes
                interval = GlobalParams.BINANCE_INTERVALS[timeframe]
                task = fetch_futures_klines(symbol, interval, client, limit=50)
                fetch_tasks.append(task)
                symbol_timeframe_map[len(fetch_tasks)-1] = (symbol, timeframe)

        # Execute all fetch tasks concurrently
        fetched_data = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Organize fetched data
        symbol_data = {}
        for idx, data in enumerate(fetched_data):
            symbol, timeframe = symbol_timeframe_map.get(idx, (None, None))
            if not symbol or not timeframe:
                continue
            if symbol not in symbol_data:
                symbol_data[symbol] = {}
            if isinstance(data, Exception):
                logging.error(f"Error fetching data for {symbol} at {timeframe}m: {data}")
                sentry_sdk.capture_exception(data, fingerprint=["fetch_futures_klines", symbol, timeframe])
                continue
            if not data.empty:
                symbol_data[symbol][timeframe] = data
            else:
                logging.warning(f"No data for {symbol} at {timeframe}m.")

        # Aggregate 2m klines from 1m data
        for symbol in symbol_data:
            if 1 in symbol_data[symbol]:
                df_1m = symbol_data[symbol][1]
                df_2m = aggregate_2m_from_1m(df_1m)
                if not df_2m.empty:
                    symbol_data[symbol][2] = df_2m

        # Calculate indicators for all fetched and aggregated data
        for symbol, timeframes in symbol_data.items():
            for timeframe, df in timeframes.items():
                # Skip aggregation since 2m is already aggregated
                df = calculate_ma_and_macd(df)
                symbol_data[symbol][timeframe] = df

                # Log indicators
                latest_timestamp = df['timestamp'].iloc[-1].isoformat()
                indicators = {
                    'MA3': df['MA3'].iloc[-1],
                    'MA7': df['MA7'].iloc[-1],
                    'MA25': df['MA25'].iloc[-1],
                    'EMA12': df['EMA12'].iloc[-1] if 'EMA12' in df.columns else None,
                    'EMA26': df['EMA26'].iloc[-1] if 'EMA26' in df.columns else None,
                    'MACD': df['MACD'].iloc[-1] if 'MACD' in df.columns else None,
                    'Signal_Line': df['Signal_Line'].iloc[-1] if 'Signal_Line' in df.columns else None
                }
                confidence = calculate_confidence(df)
                db.log_indicator(symbol, timeframe, latest_timestamp, indicators, confidence)

        # Aggregate alerts to send in a single message
        aggregated_alerts = []

        # Analyze signals for each symbol
        for symbol, timeframes in symbol_data.items():
            for timeframe, df in timeframes.items():
                confidence = calculate_confidence(df)
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

                # Determine if a buy or sell signal should be sent based on confidence and alerts
                direction = None
                if confidence > 70 and 'Hammer detected!' in alerts:
                    direction = 'buy'
                elif confidence < 30 and 'Bearish Engulfing detected!' in alerts:
                    direction = 'sell'
                elif immediate_change:
                    direction = 'buy' if pct_change > 0 else 'sell'

                if direction and abs(pct_change) >= GlobalParams.PRICE_CHANGE_THRESHOLD:
                    message = format_buy_sell_message(
                        symbol=symbol,
                        timeframe=timeframe,
                        price_now=current_price,
                        direction=direction,
                        confidence=confidence,
                        pct_change=pct_change
                    )
                    aggregated_alerts.append(message)
                    alerted_tokens.add(symbol)
                    db.add_alerted_token(symbol)  # Persist the alerted token
                    db.log_trade(symbol, timeframe, direction, confidence, pct_change, current_price, datetime.now(timezone.utc).isoformat())
                    logging.info(f"Alert prepared for {symbol}: {direction.upper()} at {pct_change:.2f}% change.")
                    # Removed break to allow multiple alerts per symbol across different timeframes

        # Send aggregated alerts
        if aggregated_alerts:
            aggregated_message = "\n\n".join(aggregated_alerts)
            await telegram_queue.put(aggregated_message)

    except Exception as e:
        logging.error(f"Error in analyze_symbols_batch: {e}", exc_info=True)
        sentry_sdk.capture_exception(e, fingerprint=["analyze_symbols_batch"])

# =========================
# 12. Main Polling Loop with Batch Processing
# =========================
async def main_loop(client: AsyncClient, bot: Bot, chat_id: str, db: AlertedTokensDB):
    """
    The main polling loop that periodically fetches top gainers and losers and analyzes all symbols in batches.
    
    Additionally, logs detailed information about indicators and trades to the database.
    """
    global TOKENS

    # Load alerted tokens once at the start
    alerted_tokens = db.load_alerted_tokens()

    # Initialize last_symbols_update
    last_symbols_update = datetime.now(timezone.utc) - timedelta(seconds=GlobalParams.TOP_GAINERS_UPDATE_INTERVAL)

    # Start Telegram worker
    asyncio.create_task(telegram_worker())

    # Start memory monitoring
    asyncio.create_task(monitor_memory())

    while True:
        try:
            now = datetime.now(timezone.utc)

            # Update top symbols every TOP_GAINERS_UPDATE_INTERVAL seconds
            if not TOKENS or (now - last_symbols_update).total_seconds() >= GlobalParams.TOP_GAINERS_UPDATE_INTERVAL:
                new_tokens = await fetch_futures_top_symbols(
                    client,
                    gainers_limit=GlobalParams.TOP_GAINERS_LIMIT,
                    losers_limit=GlobalParams.TOP_LOSERS_LIMIT
                )
                if new_tokens:
                    TOKENS = new_tokens
                last_symbols_update = now

            if not TOKENS:
                logging.warning("No FUTURES tokens in top gainers or losers. Sleeping, then retrying.")
                await asyncio.sleep(GlobalParams.POLL_INTERVAL)
                continue

            # Analyze symbols in batch
            await analyze_symbols_batch(TOKENS, client, bot, chat_id, alerted_tokens, db)

            logging.info(f"Completed batch analysis for {len(TOKENS)} FUTURES tokens. Sleeping {GlobalParams.POLL_INTERVAL}s...")
            logging.info("Heartbeat: Script is running smoothly.")  # Heartbeat log as a breadcrumb

            # Log heartbeat as a breadcrumb for Sentry
            sentry_sdk.add_breadcrumb(
                category="heartbeat",
                message="Script is running smoothly.",
                level=logging.INFO,
            )

            await asyncio.sleep(GlobalParams.POLL_INTERVAL)

        except Exception as e:
            logging.error(f"Error in main_loop: {e}", exc_info=True)
            sentry_sdk.capture_exception(e, fingerprint=["main_loop"])
            await asyncio.sleep(GlobalParams.POLL_INTERVAL)  # Prevent tight loop in case of continuous errors

# Added memory monitoring function
async def monitor_memory():
    """
    Monitor and log the script's memory usage periodically.
    """
    process = psutil.Process()
    while True:
        mem = process.memory_info().rss / (1024 * 1024)  # in MB
        logging.info(f"Memory Usage: {mem:.2f} MB")
        sentry_sdk.add_breadcrumb(
            category="memory",
            message=f"Memory Usage: {mem:.2f} MB",
            level=logging.INFO,
        )
        await asyncio.sleep(60)  # Log every minute

# =========================
# 13. Graceful Shutdown
# =========================
async def shutdown(client: AsyncClient, db: AlertedTokensDB):
    """
    Gracefully shutdown the application by closing client connections and the database.
    """
    logging.info("Shutting down gracefully...")
    try:
        await client.close_connection()
    except Exception as e:
        logging.error(f"Error closing Binance client: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)

    try:
        db.close()
    except Exception as e:
        logging.error(f"Error closing SQLite database: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)

    # Ensure all Telegram messages are sent before exiting
    await telegram_queue.join()

    # Ensure all Sentry events are sent before exiting
    sentry_sdk.flush(timeout=2)
    logging.info("Shutdown complete.")
    sys.exit(0)

def handle_signal(signal_num, frame, client, db):
    """
    Handle termination signals to initiate graceful shutdown.
    """
    logging.info(f"Received signal {signal_num}. Initiating shutdown...")
    asyncio.create_task(shutdown(client, db))

# =========================
# 14. Main Function
# =========================
async def main():
    """
    The main function that initializes the Binance client and starts the polling loop.
    """
    # Implement retry mechanism for Binance client creation
    retries = 5
    backoff = 5
    client = None
    for attempt in range(1, retries + 1):
        try:
            client = await AsyncClient.create()
            logging.info("Binance AsyncClient created successfully.")
            break
        except Exception as e:
            logging.error(f"Attempt {attempt} - Error initializing Binance client: {e}", exc_info=True)
            sentry_sdk.capture_exception(e, fingerprint=["AsyncClient.create"])
            if attempt < retries:
                logging.info(f"Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
            else:
                logging.critical("Failed to initialize Binance client after multiple attempts.")
                return

    # Initialize SQLite database
    db = AlertedTokensDB()

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig, lambda sig=sig: handle_signal(sig, None, client, db)
            )
        except NotImplementedError:
            logging.warning(f"Signal handler for {sig} not implemented on this platform.")

    try:
        await main_loop(client, bot, TelegramConfig.TELEGRAM_CHAT_ID, db)
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}", exc_info=True)
        sentry_sdk.capture_exception(e, fingerprint=["main"])
    finally:
        await client.close_connection()
        db.close()
        logging.info("Binance client and database connections closed.")

# =========================
# 15. Entry Point
# =========================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script terminated by user.")
    except Exception as e:
        logging.error(f"Unexpected error in __main__: {e}", exc_info=True)
        sentry_sdk.capture_exception(e, fingerprint=["__main__"])
