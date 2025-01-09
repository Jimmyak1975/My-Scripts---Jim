import sys
import asyncio
import aiohttp
import aiosqlite
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.error import TelegramError

# ===============================
# 1. Set Event Loop Policy on Windows
# ===============================
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ===============================
# 2. Configure Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("gainers_losers_dormants.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# 3. Configuration Parameters
# ===============================

# Binance API Configuration
BINANCE_API_URL = "https://api.binance.com"
API_ENDPOINT_24HR_TICKER = "/api/v3/ticker/24hr"
API_ENDPOINT_KLINES = "/api/v3/klines"

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "7721789198:AAFQbDqk7Ln5O-O3eGwYh05MZMCdVunfMHI"  # Replace with your Telegram Bot Token
TELEGRAM_CHAT_ID = "7052327528"      # Replace with your Telegram Chat ID

# Alert Parameters
GAINERS_COUNT = 20
THRESHOLD_PERCENT = 0.5  # Percentage change to trigger alerts
RESET_PERIOD_HOURS = 2   # Hours before resetting alert state
MA_PERIODS = [7, 25, 99]  # Moving Average periods
TIMEFRAME_MINUTES = 1    # Timeframe for monitoring (not directly used here)
ENGLISH_TOKENS_EXCLUDE = []  # Symbols to exclude from monitoring

# ===============================
# 4. Initialize Telegram Bot
# ===============================
bot = Bot(token=TELEGRAM_BOT_TOKEN)
logger.info("Successfully initialized Telegram bot.")

# ===============================
# 5. Database Configuration
# ===============================
DB_NAME = 'alert_state.db'

async def init_db():
    """Initialize the SQLite database for managing alert states."""
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS alerted_tokens (
                    symbol TEXT PRIMARY KEY,
                    last_alert TIMESTAMP
                )
            ''')
            await db.commit()
            logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

async def reset_alerts_db():
    """Reset the alert states in the database."""
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute('DELETE FROM alerted_tokens')
            await db.commit()
            logger.info("Alerts reset in the database.")
    except Exception as e:
        logger.error(f"Error resetting alerts in database: {e}")

async def should_alert(symbol):
    """Determine if an alert should be sent for a given symbol."""
    now = datetime.utcnow()
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            cursor = await db.execute(
                'SELECT last_alert FROM alerted_tokens WHERE symbol = ?',
                (symbol,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                # No previous alert, proceed and record the alert
                await db.execute(
                    'INSERT INTO alerted_tokens (symbol, last_alert) VALUES (?, ?)',
                    (symbol, now)
                )
                await db.commit()
                return True
            last_alert = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
            if now - last_alert >= timedelta(hours=RESET_PERIOD_HOURS):
                # Reset the alert
                await db.execute(
                    'UPDATE alerted_tokens SET last_alert = ? WHERE symbol = ?',
                    (now, symbol)
                )
                await db.commit()
                return True
            return False
    except Exception as e:
        logger.error(f"Error accessing alert database for symbol {symbol}: {e}")
        return False

# ===============================
# 6. Helper Functions
# ===============================

def clean_data(df):
    """Clean the DataFrame by removing NaN or infinite values."""
    if df.empty:
        logger.warning("DataFrame is empty. Skipping.")
        return df
    df.replace([np.inf, -np.inf, None], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def calculate_moving_averages(df, ma_periods=[7, 25, 99]):
    """
    Calculate moving averages for specified periods.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' prices.
        ma_periods (list): List of periods for moving averages.

    Returns:
        dict: Dictionary of moving averages.
    """
    ma_values = {}
    for period in ma_periods:
        if len(df) >= period:
            ma = df['Close'].rolling(window=period).mean().iloc[-1]
            ma_values[f'MA{period}'] = ma
        else:
            ma_values[f'MA{period}'] = np.nan  # Not enough data
    return ma_values

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD and Signal Line.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' prices.
        fast_period (int): Fast EMA period.
        slow_period (int): Slow EMA period.
        signal_period (int): Signal line EMA period.

    Returns:
        dict: MACD and Signal Line values.
    """
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return {'MACD': macd.iloc[-1], 'Signal_Line': signal.iloc[-1]}

def detect_candlestick_patterns(df):
    """
    Detect candlestick patterns such as Engulfing and Large Candles.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Open', 'Close', 'High', 'Low'.

    Returns:
        dict: Detected patterns.
    """
    patterns = {}
    if len(df) < 2:
        patterns['Engulfing'] = False
        patterns['Large_Candle'] = False
        return patterns

    # Previous candle
    prev_open = df['Open'].iloc[-2]
    prev_close = df['Close'].iloc[-2]

    # Current candle
    curr_open = df['Open'].iloc[-1]
    curr_close = df['Close'].iloc[-1]
    curr_high = df['High'].iloc[-1]
    curr_low = df['Low'].iloc[-1]

    # Engulfing Pattern
    engulfing = False
    if (curr_close > curr_open) and (prev_close < prev_open):
        if (curr_close > prev_open) and (curr_open < prev_close):
            engulfing = True
    elif (curr_close < curr_open) and (prev_close > prev_open):
        if (curr_close < prev_open) and (curr_open > prev_close):
            engulfing = True
    patterns['Engulfing'] = engulfing

    # Large Candle (e.g., candle size >= 3x previous candle)
    prev_candle_size = abs(prev_close - prev_open)
    curr_candle_size = abs(curr_close - curr_open)
    large_candle = False
    if prev_candle_size > 0:
        if curr_candle_size >= 3 * prev_candle_size:
            large_candle = True
    patterns['Large_Candle'] = large_candle

    return patterns

def determine_confidence_level(indicators):
    """
    Determine the confidence level based on technical indicators.

    Parameters:
        indicators (dict): Dictionary containing technical indicators.

    Returns:
        float: Confidence level percentage.
    """
    confidence = 0
    total_factors = 0

    # MACD
    if 'MACD' in indicators and 'Signal_Line' in indicators:
        if indicators['MACD'] > indicators['Signal_Line']:
            confidence += 25  # Bullish signal
        else:
            confidence += 15  # Bearish signal
        total_factors += 25

    # Moving Averages
    if 'Moving_Averages' in indicators:
        ma7 = indicators['Moving_Averages'].get('MA7', np.nan)
        ma25 = indicators['Moving_Averages'].get('MA25', np.nan)
        ma99 = indicators['Moving_Averages'].get('MA99', np.nan)
        current_close = indicators.get('close', np.nan)
        if not np.isnan(ma7) and not np.isnan(ma25) and not np.isnan(ma99) and not np.isnan(current_close):
            if ma7 > ma25 > ma99 and current_close > ma7:
                confidence += 20  # Strong uptrend
            elif ma7 < ma25 < ma99 and current_close < ma7:
                confidence += 20  # Strong downtrend
            elif ma7 > ma25 > ma99:
                confidence += 10  # Potential uptrend
            elif ma7 < ma25 < ma99:
                confidence += 10  # Potential downtrend
            total_factors += 20

    # Candlestick Patterns
    if 'Candlestick_Patterns' in indicators:
        if indicators['Candlestick_Patterns'].get('Engulfing') or indicators['Candlestick_Patterns'].get('Large_Candle'):
            confidence += 20  # Pattern detected
        total_factors += 20

    # Volume Spike (Implementing a simple condition: current volume > average volume)
    if 'Volume' in indicators:
        average_volume = indicators.get('average_volume', 0)
        current_volume = indicators.get('Volume', 0)
        if average_volume > 0 and current_volume > average_volume:
            confidence += 15  # Volume spike
        total_factors += 15

    # Normalize confidence to 100%
    confidence_level = (confidence / total_factors) * 100 if total_factors > 0 else 50
    return confidence_level

# ===============================
# 7. Notification Function
# ===============================

async def send_notification(symbol, percent_change, current_price, indicators, alert_type, confidence_level):
    """
    Send a Telegram notification.

    Parameters:
        symbol (str): Cryptocurrency symbol.
        percent_change (float): Percentage price change.
        current_price (float): Current price.
        indicators (dict): Technical indicators.
        alert_type (str): Type of alert (e.g., Gainer, Loser).
        confidence_level (float): Confidence level percentage.
    """
    try:
        message = f"🚨 *{alert_type} Alert!*\n"
        message += f"- *Symbol:* {symbol}\n"
        message += f"- *Price Change:* {percent_change:.2f}%\n"
        message += f"- *Current Price:* {current_price:.4f} USDT\n\n"

        if 'Moving_Averages' in indicators:
            message += "*Moving Averages:*\n"
            for ma, value in indicators['Moving_Averages'].items():
                message += f"• {ma}: {value:.4f}\n"

        if 'MACD' in indicators and 'Signal_Line' in indicators:
            message += "*MACD Indicators:*\n"
            message += f"• MACD: {indicators['MACD']:.4f}\n"
            message += f"• Signal Line: {indicators['Signal_Line']:.4f}\n"

        if 'Candlestick_Patterns' in indicators:
            patterns = []
            if indicators['Candlestick_Patterns'].get('Engulfing'):
                patterns.append("Engulfing")
            if indicators['Candlestick_Patterns'].get('Large_Candle'):
                patterns.append("Large Candle")
            if patterns:
                message += "*Candlestick Patterns Detected:*\n" + ", ".join(patterns) + "\n"

        message += f"\n*Confidence Level:* {confidence_level:.0f}%"

        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        logger.info(f"Sent {alert_type} alert for {symbol}.")
    except TelegramError as te:
        logger.error(f"Failed to send Telegram message: {te}")
    except Exception as e:
        logger.error(f"Unexpected error in send_notification: {e}")

# ===============================
# 8. Main Monitoring Function
# ===============================

async def monitor_binance():
    """
    Main function to monitor Binance for top gainers and send alerts.
    """
    await init_db()

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Fetch 24-hour ticker data
                async with session.get(BINANCE_API_URL + API_ENDPOINT_24HR_TICKER) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching data: HTTP {response.status}")
                        await asyncio.sleep(60)  # Wait before retrying
                        continue
                    tickers = await response.json()

                # Filter USDT tickers and exclude specified tokens
                usdt_tickers = [
                    ticker for ticker in tickers
                    if ticker['symbol'].endswith('USDT') and ticker['symbol'] not in ENGLISH_TOKENS_EXCLUDE
                ]

                # Sort gainers based on priceChangePercent
                sorted_gainers = sorted(
                    usdt_tickers,
                    key=lambda x: float(x['priceChangePercent']),
                    reverse=True
                )[:GAINERS_COUNT]
                sorted_gainers_symbols = [ticker['symbol'] for ticker in sorted_gainers]
                logger.info(f"Top {GAINERS_COUNT} Gainers: {sorted_gainers_symbols}")

                # Process each gainer
                for ticker in sorted_gainers:
                    symbol = ticker['symbol']
                    percent_change = float(ticker['priceChangePercent'])
                    current_price = float(ticker['lastPrice'])
                    current_volume = float(ticker['volume'])

                    # Check if the percent change exceeds the threshold
                    if percent_change < THRESHOLD_PERCENT:
                        continue  # Skip if below threshold

                    # Determine alert type (Gainer)
                    alert_type = 'Gainer'

                    # Check if an alert should be sent
                    if not await should_alert(symbol):
                        logger.info(f"Alert already sent recently for {symbol}. Skipping.")
                        continue

                    # Fetch historical klines for technical indicators
                    KLINE_INTERVAL = "1m"
                    KLINE_LIMIT = max(MA_PERIODS) + 1  # Ensure enough data points

                    params = {
                        'symbol': symbol,
                        'interval': KLINE_INTERVAL,
                        'limit': KLINE_LIMIT
                    }

                    async with session.get(BINANCE_API_URL + API_ENDPOINT_KLINES, params=params) as kline_response:
                        if kline_response.status != 200:
                            logger.error(f"Error fetching klines for {symbol}: HTTP {kline_response.status}")
                            continue
                        klines = await kline_response.json()

                    # Convert klines to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Close Time', 'Quote Asset Volume', 'Number of Trades',
                        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
                    ])
                    df['Open'] = df['Open'].astype(float)
                    df['High'] = df['High'].astype(float)
                    df['Low'] = df['Low'].astype(float)
                    df['Close'] = df['Close'].astype(float)
                    df['Volume'] = df['Volume'].astype(float)
                    df.set_index('Close Time', inplace=True)
                    df = clean_data(df)
                    if df.empty:
                        logger.warning(f"No data available for {symbol}. Skipping.")
                        continue

                    # Calculate technical indicators
                    ma_values = calculate_moving_averages(df, MA_PERIODS)
                    macd_values = calculate_macd(df)
                    candlestick_patterns = detect_candlestick_patterns(df)

                    # Calculate average volume over the MA periods for volume spike detection
                    average_volume = df['Volume'].mean()

                    indicators = {
                        'Moving_Averages': ma_values,
                        'MACD': macd_values['MACD'],
                        'Signal_Line': macd_values['Signal_Line'],
                        'Candlestick_Patterns': candlestick_patterns,
                        'Volume': current_volume,
                        'average_volume': average_volume,  # For volume spike analysis
                        'close': current_price  # Current closing price
                    }

                    # Calculate confidence level
                    confidence_level = determine_confidence_level(indicators)

                    # Send notification
                    await send_notification(
                        symbol=symbol,
                        percent_change=percent_change,
                        current_price=current_price,
                        indicators=indicators,
                        alert_type=alert_type,
                        confidence_level=confidence_level
                    )

                # Wait for the next interval
                await asyncio.sleep(60)  # Wait for 1 minute

            except aiohttp.ClientError as ce:
                logger.error(f"HTTP Client Error: {ce}")
                await asyncio.sleep(60)  # Wait before retrying
            except asyncio.CancelledError:
                logger.info("Monitoring task cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

# ===============================
# 9. Scheduled Reset Task
# ===============================

async def scheduled_reset():
    """
    Scheduled task to reset alerts periodically.
    """
    while True:
        now = datetime.utcnow()
        # Calculate the next reset time aligned to the reset period
        next_reset = now + timedelta(hours=RESET_PERIOD_HOURS)
        next_reset = next_reset.replace(minute=0, second=0, microsecond=0)
        wait_seconds = (next_reset - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        await reset_alerts_db()

# ===============================
# 10. Main Function
# ===============================

async def main():
    """
    Main entry point for the monitoring script.
    """
    # Initialize database
    await init_db()

    # Create tasks
    monitor_task = asyncio.create_task(monitor_binance())
    reset_task = asyncio.create_task(scheduled_reset())

    # Run tasks concurrently
    await asyncio.gather(monitor_task, reset_task)

# ===============================
# 11. Entry Point
# ===============================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script terminated by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
