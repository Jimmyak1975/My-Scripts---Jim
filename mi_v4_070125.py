import asyncio
import sys

# Force the Windows event loop policy to SelectorEventLoop (fixes aiodns on Windows).
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd
import numpy as np
from binance import AsyncClient, BinanceSocketManager
from telegram import Bot
import logging
from ta import trend, momentum, volatility, volume
import datetime

# ======================= CONFIGURE LOGGING =======================
logging.basicConfig(
    level=logging.INFO,  # Shows INFO or higher in the console
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot_debug.log', mode='a'),
        logging.StreamHandler()
    ]
)

# ======================= GLOBAL CONFIGS ==========================
TELEGRAM_BOT_TOKEN = '7721789198:AAFQbDqk7Ln5O-O3eGwYh05MZMCdVunfMHI'   # <-- Replace with your actual token
TELEGRAM_CHAT_ID = '7052327528'       # <-- Replace with your actual chat ID

# HPP Criteria: Top 10 tokens with highest price difference in last 10 days
HPP_LOOKBACK_DAYS = 10

# Dormant Token Monitoring Timeframe
DORMANT_WINDOW_START = datetime.time(0, 59, 30)  # 12:59:30 AM UTC
DORMANT_WINDOW_END = datetime.time(1, 2, 30)     # 1:02:30 AM UTC

# ======================= DATA STRUCTURES =========================
TOKENS = []        # Dynamically updated list of top gainers
data_frames = {}   # Dictionary of DataFrames keyed by token symbol
hpp_tokens = set() # High Profit Potential tokens

# ======================= TELEGRAM BOT ============================
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# ======================= INDICATOR FUNCTIONS ======================
def calculate_indicators(df):
    """Calculate required technical indicators using ta library."""
    if len(df) < 100:
        return df

    # Moving Averages
    df['MA3'] = df['close'].rolling(window=3).mean()
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA25'] = df['close'].rolling(window=25).mean()

    # MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    bollinger = volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()

    # Stochastic Oscillator
    stochastic = momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'],
        window=14, smooth_window=3
    )
    df['%K'] = stochastic.stoch()
    df['%D'] = stochastic.stoch_signal()

    # ATR
    atr = volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    )
    df['ATR'] = atr.average_true_range()

    # OBV
    obv = volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    df['OBV'] = obv.on_balance_volume()

    return df

def identify_support_resistance(df, windows=[20, 50, 100], threshold=3):
    """Identify support and resistance levels based on multiple window sizes."""
    support_levels = {}
    resistance_levels = {}

    for window in windows:
        df[f'min_{window}'] = df['close'].rolling(window=window, center=True).min()
        df[f'max_{window}'] = df['close'].rolling(window=window, center=True).max()

        support = df[df['close'] == df[f'min_{window}']]['close'].dropna()
        resistance = df[df['close'] == df[f'max_{window}']]['close'].dropna()

        s_agg = support.round(decimals=2).value_counts()
        r_agg = resistance.round(decimals=2).value_counts()

        confirmed_support = s_agg[s_agg >= threshold].index.tolist()
        confirmed_resistance = r_agg[r_agg >= threshold].index.tolist()

        support_levels[window] = confirmed_support
        resistance_levels[window] = confirmed_resistance

    return support_levels, resistance_levels

# ======================= PATTERN DETECTION =======================
def detect_candlestick_patterns(symbol, df):
    """Detect various bullish and bearish candlestick patterns."""
    alerts = []
    if len(df) < 2:
        return alerts

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    body = latest['close'] - latest['open']
    lower_wick = latest['open'] - latest['low'] if body > 0 else latest['close'] - latest['low']
    upper_wick = latest['high'] - latest['close'] if body > 0 else latest['high'] - latest['open']

    # Example: Hammer
    if body > 0 and lower_wick > 2 * abs(body) and (latest['high'] - max(latest['close'], latest['open'])) < 0.1 * abs(body):
        alerts.append(f"{symbol}: Hammer detected!")

    # Example: Bearish Engulfing
    if body < 0 and previous['close'] > previous['open'] and latest['open'] > previous['close'] and latest['close'] < previous['open']:
        alerts.append(f"{symbol}: Bearish Engulfing detected!")

    return alerts

def detect_volume_patterns(symbol, df):
    """Detect various volume-based patterns."""
    alerts = []
    if len(df) < 20:
        return alerts

    latest = df.iloc[-1]
    if len(df) > 1:
        previous = df.iloc[-2]
    else:
        previous = None

    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    if latest['volume'] > 2 * avg_volume:
        alerts.append(f"{symbol}: Volume Spike on Breakout detected!")

    return alerts

# ======================= PRICE RANGE & CONFIDENCE ================
def calculate_price_range(df):
    """Calculate the price range change over the last minute."""
    if len(df) < 2:
        return 0
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    return ((latest['close'] - prev['close']) / prev['close']) * 100 if prev['close'] != 0 else 0

def calculate_confidence(df):
    """Very basic confidence measure counting bullish vs bearish signals."""
    if len(df) < 2:
        return 50

    bullish = 0
    bearish = 0

    # MAs
    if df['MA3'].iloc[-1] > df['MA7'].iloc[-1] > df['MA25'].iloc[-1]:
        bullish += 1
    elif df['MA3'].iloc[-1] < df['MA7'].iloc[-1] < df['MA25'].iloc[-1]:
        bearish += 1

    # MACD
    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
        bullish += 1
    else:
        bearish += 1

    # Bollinger
    if df['close'].iloc[-1] > df['Bollinger_High'].iloc[-1]:
        bullish += 1
    elif df['close'].iloc[-1] < df['Bollinger_Low'].iloc[-1]:
        bearish += 1

    # OBV
    if df['OBV'].iloc[-1] > df['OBV'].iloc[-2]:
        bullish += 1
    else:
        bearish += 1

    total = bullish + bearish
    if total == 0:
        return 50
    return (bullish / total) * 100

# ======================= HPP TOKENS ==============================
def detect_hpp_tokens(symbol, df):
    """
    Detect High Profit Potential (HPP) tokens based on
    a 10-day lookback period with a 10% threshold.
    """
    global hpp_tokens
    if len(df) < 14400:  # 10 days * 24h * 60min
        return None

    latest_close = df['close'].iloc[-1]
    price_10d_ago = df['close'].iloc[-14400]
    if price_10d_ago <= 0:
        return None

    price_diff = ((latest_close - price_10d_ago) / price_10d_ago) * 100
    if price_diff >= 10:
        if symbol not in hpp_tokens:
            hpp_tokens.add(symbol)
            return f"{symbol}: HPP Token detected! +{price_diff:.2f}% over last 10 days."
    else:
        if symbol in hpp_tokens:
            hpp_tokens.remove(symbol)
    return None

# ======================= ALERT FORMATTING ========================
def format_telegram_message(symbol, signal_type, confidence, price_range, price_now):
    buy_symbol = '🟢B'
    sell_symbol = '🔴S'
    sudden_activity_symbol = '🟡SA'
    hpp_symbol = '🔵HPP'

    symbols = []
    if signal_type == 'buy':
        symbols.append(buy_symbol)
    elif signal_type == 'sell':
        symbols.append(sell_symbol)
    if signal_type == 'sudden_activity':
        symbols.append(sudden_activity_symbol)
    if symbol in hpp_tokens:
        symbols.append(hpp_symbol)

    top_line = ' '.join(symbols)
    message = (
        f"{top_line}\n"
        f"{symbol}\n"
        f"Confidence Level: {confidence:.2f}%\n"
        f"Price Range Change: {price_range:.2f}%\n"
        f"Current Price: {price_now}\n"
    )
    return message

async def send_telegram_message(text):
    """Send message to Telegram and log it."""
    try:
        logging.info(f"Sending message to Telegram: {text}")
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

# ======================= TRADE PROCESSING ========================
async def process_trade(symbol, trade):
    """Process each trade from Binance WebSocket."""
    try:
        df = data_frames[symbol]

        event_time = pd.to_datetime(trade['E'], unit='ms')
        price = float(trade['c'])
        volume_trade = float(trade['v'])

        new_row = {
            'timestamp': event_time,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume_trade
        }
        df = df.append(new_row, ignore_index=True)
        df = df.tail(1200)  # keep last 1200 rows
        data_frames[symbol] = df

        # Indicators
        df = calculate_indicators(df)
        # S/R Levels
        identify_support_resistance(df)

        # Patterns
        candle_alerts = detect_candlestick_patterns(symbol, df)
        volume_alerts = detect_volume_patterns(symbol, df)

        # HPP
        hpp_alert = detect_hpp_tokens(symbol, df)
        if hpp_alert:
            msg = format_telegram_message(symbol, 'buy', 80, 0.0, price)
            await send_telegram_message(msg)

        # Combine alerts
        all_alerts = candle_alerts + volume_alerts
        for alert in all_alerts:
            if any(k in alert for k in ['Bullish', 'Hammer', 'Golden Cross']):
                signal_type = 'buy'
            elif any(k in alert for k in ['Bearish', 'Shooting Star', 'Death Cross']):
                signal_type = 'sell'
            else:
                signal_type = 'other'

            conf = calculate_confidence(df)
            pr_range = calculate_price_range(df)
            price_now = df['close'].iloc[-1]
            msg = format_telegram_message(symbol, signal_type, conf, pr_range, price_now)
            await send_telegram_message(msg)

    except Exception as e:
        logging.error(f"Error in process_trade for {symbol}: {e}")

# ======================= WEBSOCKET HANDLERS ======================
async def binance_websocket_handler():
    """Continuously receive trade messages from Binance for monitored tokens."""
    try:
        # Force public client mode with None keys
        client = await AsyncClient.create(api_key=None, api_secret=None, testnet=False)
        bm = BinanceSocketManager(client)

        streams = [f"{symbol.lower()}@trade" for symbol in TOKENS]
        logging.info(f"Starting WebSocket for: {streams}")
        async with bm.multiplex_socket(streams) as stream:
            while True:
                msg = await stream.recv()
                if 'e' not in msg:
                    continue
                if msg['e'] != 'trade':
                    continue
                s = msg['s']
                asyncio.create_task(process_trade(s, msg))

    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        logging.info("Closing WebSocket connection...")
        await client.close_connection()

# ======================= FETCH TOP GAINERS =======================
async def fetch_top_gainers(client, limit=20):
    """
    Fetch top USDT gainers from the last 24 hours (public data) using `get_ticker()`
    which returns 24-hour rolling statistics for all symbols.
    """
    try:
        logging.info("Fetching top gainers from Binance (24h).")
        # Instead of get_ticker_24hr, we use get_ticker()
        tickers = await client.get_ticker()  # Returns 24hr price stats for all symbols

        df_tickers = pd.DataFrame(tickers)

        # Filter for USDT pairs only
        df_tickers = df_tickers[df_tickers['symbol'].str.contains('USDT')]

        # Convert the priceChangePercent field to float
        df_tickers['priceChangePercent'] = pd.to_numeric(df_tickers['priceChangePercent'], errors='coerce')
        df_tickers = df_tickers.dropna(subset=['priceChangePercent'])

        # Sort by biggest gainers
        df_tickers = df_tickers.sort_values(by='priceChangePercent', ascending=False)
        top_n = df_tickers.head(limit)
        top_symbols = top_n['symbol'].tolist()
        logging.info(f"Top Gainers: {top_symbols}")
        return top_symbols
    except Exception as e:
        logging.error(f"Error fetching top gainers: {e}")
        return []

async def update_top_gainers():
    """Periodically update the top gainers list every few minutes."""
    global TOKENS, data_frames

    try:
        logging.info("Updating the top gainers list...")
        client = await AsyncClient.create(api_key=None, api_secret=None)
        new_list = await fetch_top_gainers(client, limit=20)
        await client.close_connection()

        # If there's no difference, do nothing
        if not new_list:
            logging.warning("No top gainer data returned, skipping update...")
            return

        old_set = set(TOKENS)
        new_set = set(new_list)

        # Tokens to add or remove
        to_add = new_set - old_set
        to_remove = old_set - new_set

        if to_add or to_remove:
            logging.info(f"Tokens to add: {to_add}, Tokens to remove: {to_remove}")

        TOKENS = new_list

        for sym in to_add:
            if sym not in data_frames:
                data_frames[sym] = pd.DataFrame(
                    columns=['timestamp','open','high','low','close','volume']
                )

        for sym in to_remove:
            if sym in data_frames:
                del data_frames[sym]

        if to_add or to_remove:
            logging.info(f"Updated list of tokens: {TOKENS}")
            asyncio.create_task(restart_websocket_streams())

    except Exception as e:
        logging.error(f"Error updating top gainers: {e}")

async def restart_websocket_streams():
    """Restart the WebSocket streams with the new tokens."""
    logging.info("Restarting WebSocket streams...")
    await asyncio.sleep(2)
    # Start new streams
    asyncio.create_task(binance_websocket_handler())

# ======================= MAIN LOOP ==============================
async def main():
    """Main function to handle periodic updates and WebSocket connections."""
    logging.info("Starting main() function...")

    # Create minimal DataFrames for tokens
    for sym in TOKENS:
        if sym not in data_frames:
            data_frames[sym] = pd.DataFrame(
                columns=['timestamp','open','high','low','close','volume']
            )

    # Start the WebSocket in the background
    asyncio.create_task(binance_websocket_handler())

    # Update top gainers every 5 minutes
    while True:
        await update_top_gainers()
        await asyncio.sleep(300)  # 5 minutes

# ======================= ENTRY POINT ============================
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        # Initial fetch
        logging.info("Fetching initial top gainers before starting the main loop...")
        init_client = loop.run_until_complete(AsyncClient.create(api_key=None, api_secret=None))
        initial_tokens = loop.run_until_complete(fetch_top_gainers(init_client, limit=20))
        loop.run_until_complete(init_client.close_connection())

        if not initial_tokens:
            logging.warning("No initial tokens found. The script will still run, but no tokens are monitored.")
        TOKENS = initial_tokens

        # Initialize data frames
        for s in TOKENS:
            data_frames[s] = pd.DataFrame(
                columns=['timestamp','open','high','low','close','volume']
            )

        logging.info(f"Initial token list: {TOKENS}")
        asyncio.run(main())

    except KeyboardInterrupt:
        logging.info("Script terminated by user.")
    except Exception as e:
        logging.error(f"Unexpected error in __main__: {e}")
