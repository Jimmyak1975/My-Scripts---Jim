import sys
import asyncio
from datetime import datetime, timedelta
import ccxt
import logging
from telegram import Bot
import pandas as pd
import numpy as np
import aiosqlite
from binance import AsyncClient, BinanceSocketManager

# ===============================
# 1. Set Event Loop Policy on Windows
# ===============================
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ===============================
# 2. Configure Logging
# ===============================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("gainers_losers_dormants.log"),
        logging.StreamHandler()
    ]
)

# ===============================
# 3. Binance API Configuration
# ===============================
BINANCE_API_KEY = "VxIKafSgpDsXoXXV0tHSIMI8jCczHrv8VMbBIIYrFvtmkMqSq0JobCjWt3E56Y9p"
BINANCE_SECRET_KEY = "UdkjWyVGYWcl5u05bRJKO4XyQQ7eQ76j5mJhsFdkfGaAGiwojLcx6aSjMiR2Dn3S"

# ===============================
# 4. Telegram Bot Configuration
# ===============================
TELEGRAM_BOT_TOKEN = "7721789198:AAFQbDqk7Ln5O-O3eGwYh05MZMCdVunfMHI"
TELEGRAM_CHAT_ID = "7052327528"

# ===============================
# 5. Parameters
# ===============================
GAINERS_COUNT = 30
LOSERS_COUNT = 30
THRESHOLDS = [0.5, 1.0]  # Adjusted for sensitivity
TIMEFRAMES = [1, 3]  # in minutes
RESET_PERIOD = 2  # in hours

# ===============================
# 6. Initialize Binance Exchange and Telegram Bot
# ===============================
try:
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
    })
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    logging.info("Successfully initialized Binance exchange and Telegram bot.")
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    exit(1)

# ===============================
# 7. State Management Database
# ===============================
DB_NAME = 'alert_state.db'

async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS alerted_tokens (
                symbol TEXT,
                timeframe INTEGER,
                last_alert TIMESTAMP,
                PRIMARY KEY (symbol, timeframe)
            )
        ''')
        await db.commit()

async def reset_alerts_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute('DELETE FROM alerted_tokens')
        await db.commit()
        logging.info("Alerts reset in the database.")

async def should_alert_db(symbol, timeframe):
    now = datetime.utcnow()
    async with aiosqlite.connect(DB_NAME) as db:
        cursor = await db.execute(
            'SELECT last_alert FROM alerted_tokens WHERE symbol = ? AND timeframe = ?',
            (symbol, timeframe)
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            await db.execute(
                'INSERT INTO alerted_tokens (symbol, timeframe, last_alert) VALUES (?, ?, ?)',
                (symbol, timeframe, now)
            )
            await db.commit()
            return True
        last_alert = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
        if now - last_alert >= timedelta(hours=RESET_PERIOD):
            await db.execute(
                'UPDATE alerted_tokens SET last_alert = ? WHERE symbol = ? AND timeframe = ?',
                (now, symbol, timeframe)
            )
            await db.commit()
            return True
        return False

# ===============================
# 8. State Management Variables
# ===============================
HPP_TOKENS = []  # Placeholder for high-profit-potential tokens

# ===============================
# 9. Helper Functions
# ===============================
def clean_data(df):
    """Ensure data is clean and free from None or NaN values."""
    if df.empty:
        logging.warning("DataFrame is empty. Skipping.")
        return df
    df.replace([np.inf, -np.inf, None], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculates Bollinger Bands for the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' prices.
        period (int): Number of periods to calculate the moving average.
        std_dev (int): Number of standard deviations for the bands.
        
    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands added.
    """
    df['MA20'] = df['close'].rolling(window=period).mean()
    df['STD20'] = df['close'].rolling(window=period).std()
    df['UpperBand'] = df['MA20'] + (df['STD20'] * std_dev)
    df['LowerBand'] = df['MA20'] - (df['STD20'] * std_dev)
    return df

def calculate_atr(df, period=14):
    """
    Calculates Average True Range (ATR) for the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        period (int): Number of periods to calculate ATR.
        
    Returns:
        pd.DataFrame: DataFrame with ATR added.
    """
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    return df

def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    """
    Calculates Stochastic Oscillator (%K and %D) for the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        k_period (int): Number of periods to calculate %K.
        d_period (int): Number of periods to calculate %D.
        
    Returns:
        pd.DataFrame: DataFrame with Stochastic Oscillator added.
    """
    df['LowestLow'] = df['low'].rolling(window=k_period).min()
    df['HighestHigh'] = df['high'].rolling(window=k_period).max()
    df['%K'] = ((df['close'] - df['LowestLow']) / (df['HighestHigh'] - df['LowestLow'])) * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    df.drop(['LowestLow', 'HighestHigh'], axis=1, inplace=True)
    return df

def generate_notification(symbol, timeframe, percent_change, current_price, indicators, alert_type):
    """
    Generates a formatted notification message for Telegram.
    
    Parameters:
        symbol (str): Cryptocurrency symbol.
        timeframe (int): Timeframe in minutes.
        percent_change (float): Percentage price change.
        current_price (float): Current price in USDT.
        indicators (dict): Dictionary containing indicator values.
        alert_type (str): Type of alert ('Gainer', 'Loser', 'Sudden Change').
        
    Returns:
        str: Formatted notification message.
    """
    hpp_prefix = "HPP " if symbol in HPP_TOKENS else ""
    notification = (
        f"{hpp_prefix}🚨 *{alert_type} Alert!*\n"
        f"- *Symbol:* {symbol}\n"
        f"- *Timeframe:* {timeframe}m\n"
        f"- *Price Change:* {percent_change:.2f}%\n"
        f"- *Price Now:* {current_price:.4f} USDT\n"
    )
    
    if indicators:
        if 'Bollinger Bands' in indicators:
            notification += f"- *Bollinger Bands:* Upper {indicators['Bollinger Bands']['UpperBand']:.4f}, Lower {indicators['Bollinger Bands']['LowerBand']:.4f}\n"
        if 'ATR' in indicators:
            notification += f"- *ATR:* {indicators['ATR']:.4f}\n"
        if 'Stochastic Oscillator' in indicators:
            notification += f"- *Stochastic Oscillator:* %K {indicators['Stochastic Oscillator']['%K']:.2f}, %D {indicators['Stochastic Oscillator']['%D']:.2f}\n"
    
    return notification

async def send_telegram_message(message):
    """
    Sends a message via Telegram bot.
    
    Parameters:
        message (str): The message to send.
    """
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

async def process_symbol(symbol, sorted_gainers, sorted_losers):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df = clean_data(df)
        if df.empty:
            return

        # Calculate Indicators
        df = calculate_bollinger_bands(df)
        df = calculate_atr(df)
        df = calculate_stochastic_oscillator(df)

        for timeframe in TIMEFRAMES:
            try:
                if len(df) < timeframe:
                    logging.warning(f"Not enough data for {symbol} in {timeframe}m timeframe. Skipping.")
                    continue
                percent_change = ((df['close'].iloc[-1] - df['close'].iloc[-timeframe]) / df['close'].iloc[-timeframe]) * 100
                if any(percent_change >= threshold for threshold in THRESHOLDS):
                    # Determine alert type
                    if symbol in sorted_gainers:
                        alert_type = 'Gainer'
                    elif symbol in sorted_losers:
                        alert_type = 'Loser'
                    else:
                        alert_type = 'Sudden Change'
                    
                    # Gather indicator values
                    indicators = {}
                    if not df['UpperBand'].isna().iloc[-1] and not df['LowerBand'].isna().iloc[-1]:
                        indicators['Bollinger Bands'] = {
                            'UpperBand': df['UpperBand'].iloc[-1],
                            'LowerBand': df['LowerBand'].iloc[-1]
                        }
                    if not df['ATR'].isna().iloc[-1]:
                        indicators['ATR'] = df['ATR'].iloc[-1]
                    if not df['%K'].isna().iloc[-1] and not df['%D'].isna().iloc[-1]:
                        indicators['Stochastic Oscillator'] = {
                            '%K': df['%K'].iloc[-1],
                            '%D': df['%D'].iloc[-1]
                        }
                    
                    # Check if alert should be sent
                    if await should_alert_db(symbol, timeframe):
                        notification = generate_notification(
                            symbol,
                            timeframe,
                            percent_change,
                            df['close'].iloc[-1],
                            indicators,
                            alert_type
                        )
                        await send_telegram_message(notification)
            except Exception as e:
                logging.error(f"Error processing {symbol} for timeframe {timeframe}: {e}")
    except Exception as e:
        logging.error(f"Error processing {symbol}: {e}")

async def monitor_fixed_symbols(sorted_gainers, sorted_losers):
    tasks = []
    for symbol in sorted_gainers + sorted_losers:
        tasks.append(process_symbol(symbol, sorted_gainers, sorted_losers))
    await asyncio.gather(*tasks)

async def monitor_sudden_changes(all_symbols, sorted_gainers, sorted_losers):
    tasks = []
    for symbol in all_symbols:
        if symbol in sorted_gainers or symbol in sorted_losers:
            continue  # Already monitored
        tasks.append(process_symbol(symbol, sorted_gainers, sorted_losers))
    await asyncio.gather(*tasks)

async def websocket_listener(client):
    bm = BinanceSocketManager(client)
    socket = bm.symbol_ticker_socket()
    async with socket as s:
        async for msg in s:
            symbol = msg['s']  # Symbol
            price = float(msg['c'])  # Current price
            # Implement logic to handle incoming price updates
            # This would require maintaining the previous prices to calculate percent changes
            # and triggering alerts as necessary
            pass  # Placeholder for websocket handling logic

async def main():
    await init_db()
    
    client = await AsyncClient.create(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
    bm = BinanceSocketManager(client)
    
    # Start the websocket listener in the background
    # Note: Implementing full websocket data handling requires significant changes
    # This is a placeholder for integrating websocket data
    # Uncomment the following lines if you plan to implement websocket logic
    # socket = bm.stream_ticker_socket()
    # asyncio.create_task(websocket_listener(client))
    
    while True:
        try:
            tickers = exchange.fetch_tickers()
            all_symbols = [symbol for symbol in tickers.keys() if symbol.endswith('/USDT')]
            sorted_gainers = sorted(all_symbols, key=lambda x: tickers[x].get('percentage', 0) or 0, reverse=True)[:GAINERS_COUNT]
            sorted_losers = sorted(all_symbols, key=lambda x: tickers[x].get('percentage', 0) or 0)[:LOSERS_COUNT]
    
            # Monitor fixed symbols
            await monitor_fixed_symbols(sorted_gainers, sorted_losers)
    
            # Monitor sudden changes
            await monitor_sudden_changes(all_symbols, sorted_gainers, sorted_losers)
    
            # Reset alerts at a specific time
            current_time = datetime.utcnow()
            if current_time.minute == 59 and current_time.second >= 30:
                await reset_alerts_db()
    
        except Exception as e:
            logging.error(f"Error during monitoring: {e}")
    
        await asyncio.sleep(15)

    # Close the Binance client
    await client.close_connection()

# Start Monitoring
if __name__ == "__main__":
    asyncio.run(main())
