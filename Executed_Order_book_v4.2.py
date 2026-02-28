import time
import random
import threading
from dataclasses import dataclass
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------------- Toolbar theming (bottom bar) ----------------
def _apply_mpl_toolbar_theme(fig, bg="#1E1E1E", fg="#D0D0D0"):
    try:
        backend = plt.get_backend().lower()
        manager = getattr(fig.canvas, "manager", None)
        if manager is None:
            return

        tb = getattr(manager, "toolbar", None)
        if tb is None:
            return

        if "tk" in backend:
            try:
                if hasattr(tb, "config"):
                    tb.config(background=bg)

                if hasattr(tb, "winfo_children"):
                    for w in tb.winfo_children():
                        try:
                            if hasattr(w, "config"):
                                w.config(background=bg)
                                try:
                                    w.config(foreground=fg)
                                except Exception:
                                    pass
                                try:
                                    w.config(activebackground=bg, activeforeground=fg)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                msg = getattr(tb, "_message_label", None)
                if msg is not None and hasattr(msg, "config"):
                    msg.config(background=bg)
                    try:
                        msg.config(foreground=fg)
                    except Exception:
                        pass
            except Exception:
                pass

        elif "qt" in backend:
            try:
                tb.setStyleSheet(f"""
                    QToolBar {{
                        background: {bg};
                        color: {fg};
                        spacing: 6px;
                    }}
                    QToolButton {{
                        background: {bg};
                        color: {fg};
                        border: 0px;
                        padding: 2px;
                    }}
                    QToolButton:hover {{
                        background: {bg};
                    }}
                    QLabel {{
                        background: {bg};
                        color: {fg};
                    }}
                """)
            except Exception:
                pass
    except Exception:
        return


# ---------------- Configuration ----------------
SYMBOL = "SOLUSDT"
FETCH_INTERVAL = 2         # seconds (polling loop)
TABLE_SIZE = 50            # show top 50 per side
TOTAL_FETCH_LIMIT = 1000   # used where exchange supports it
VOLUME_THRESHOLD = 10      # filter trades >= 10 SOL

# Colors
BASE_COLOR_EVEN = "#2E2E2E"
BASE_COLOR_ODD = "#1E1E1E"

HIGHLIGHT_YELLOW = "yellow"   # > 50k
HIGHLIGHT_ORANGE = "orange"   # > 100k
HIGHLIGHT_BLUE = "blue"       # > 200k
HIGHLIGHT_RED = "purple"      # > 500k

EXCHANGE_SYMBOLS = {
    "Binance": "SOLUSDT",
    "Coinbase": "SOL-USD",
    "Kraken": "SOLUSD",
    "Bitfinex": "tSOLUSD",
    "Huobi Global": "solusdt",
    "OKX": "SOL-USDT",
    "KuCoin": "SOL-USDT",
    "Bitstamp": "solusd",
    "Gemini": "SOLUSD",
    "Gate.io": "SOL_USDT"
}

EXCHANGES = {
    "Binance": f"https://fapi.binance.com/fapi/v1/aggTrades",
    "Coinbase": f"https://api.exchange.coinbase.com/products/{EXCHANGE_SYMBOLS['Coinbase']}/trades",
    "Kraken": f"https://api.kraken.com/0/public/Trades",
    "Bitfinex": f"https://api-pub.bitfinex.com/v2/trades/{EXCHANGE_SYMBOLS['Bitfinex']}/hist",
    "Huobi Global": f"https://api.huobi.pro/market/history/trade",
    "OKX": f"https://www.okx.com/api/v5/market/trades",
    "KuCoin": f"https://api.kucoin.com/api/v1/market/histories",
    "Bitstamp": f"https://www.bitstamp.net/api/v2/transactions/{EXCHANGE_SYMBOLS['Bitstamp']}/",
    "Gemini": f"https://api.gemini.com/v1/trades/{EXCHANGE_SYMBOLS['Gemini'].lower()}",
    "Gate.io": f"https://api.gateio.ws/api/v4/spot/trades",
}


# ---------------- Networking (robust session) ----------------
def _build_retry():
    # Works across urllib3 versions (allowed_methods vs method_whitelist)
    kwargs = dict(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.2,  # 1.2s, 2.4s, 4.8s...
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )
    try:
        return Retry(**kwargs, allowed_methods=frozenset(["GET"]))
    except TypeError:
        return Retry(**kwargs, method_whitelist=frozenset(["GET"]))


def make_session() -> requests.Session:
    s = requests.Session()
    retry = _build_retry()
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=50,
        pool_maxsize=50,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    s.headers.update({
        "User-Agent": "Mozilla/5.0 (TradeTable/1.0)",
        "Accept": "application/json,text/plain,*/*"
    })
    return s


class RateLimitedLogger:
    """Prevents console spam: prints per exchange at most once per N seconds."""
    def __init__(self, min_seconds=30):
        self.min_seconds = min_seconds
        self._last = {}
        self._lock = threading.Lock()

    def log(self, key: str, msg: str):
        now = time.time()
        with self._lock:
            last = self._last.get(key, 0)
            if now - last >= self.min_seconds:
                self._last[key] = now
                print(msg)


LOGGER = RateLimitedLogger(min_seconds=25)


# ---------------- Trade parsing helpers ----------------
def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _clip_trades(trades, limit):
    return trades[:limit] if len(trades) > limit else trades


@dataclass(frozen=True)
class Trade:
    value: float
    is_sell: bool


def parse_binance(data):
    # aggTrades: p=price, q=qty, m=buyerIsMaker (True => taker is seller => SELL)
    out = []
    if not isinstance(data, list):
        return out
    for t in data:
        p = _safe_float(t.get("p"))
        q = _safe_float(t.get("q"))
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=p * q, is_sell=bool(t.get("m"))))
    return out


def parse_coinbase(data):
    out = []
    if not isinstance(data, list):
        return out
    for t in data:
        p = _safe_float(t.get("price"))
        q = _safe_float(t.get("size"))
        side = (t.get("side") or "").lower()
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=p * q, is_sell=(side == "sell")))
    return out


def parse_kraken(data):
    # Kraken Trades response:
    # result: { "<pair_key>": [[price, volume, time, side, ordertype, misc], ...], "last": "..." }
    out = []
    if not isinstance(data, dict):
        return out
    result = data.get("result")
    if not isinstance(result, dict):
        return out

    # Robust: find first key that isn't "last"
    pair_key = None
    for k in result.keys():
        if k != "last":
            pair_key = k
            break
    if not pair_key:
        return out

    trades = result.get(pair_key, [])
    if not isinstance(trades, list):
        return out

    for t in trades:
        if not isinstance(t, list) or len(t) < 4:
            continue
        p = _safe_float(t[0])
        q = _safe_float(t[1])
        side = str(t[3]).lower()  # <-- FIXED (side is index 3)
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=p * q, is_sell=(side == "s")))
    return out


def parse_bitfinex(data):
    # v2 trades hist: [ID, MTS, AMOUNT, PRICE]
    out = []
    if not isinstance(data, list):
        return out
    for t in data:
        if not isinstance(t, list) or len(t) < 4:
            continue
        amount = _safe_float(t[2])
        price = _safe_float(t[3])
        if amount is None or price is None:
            continue
        qty = abs(amount)
        if qty < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=price * qty, is_sell=(amount < 0)))
    return out


def parse_huobi(data):
    out = []
    if not isinstance(data, dict):
        return out
    groups = data.get("data")
    if not isinstance(groups, list) or not groups:
        return out
    trades = groups[0].get("data", [])
    if not isinstance(trades, list):
        return out

    for t in trades:
        p = _safe_float(t.get("price"))
        q = _safe_float(t.get("amount"))
        direction = (t.get("direction") or "").lower()
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=p * q, is_sell=(direction == "sell")))
    return out


def parse_okx(data):
    # OKX /api/v5/market/trades returns:
    # { "code":"0", "data":[{"px":"..","sz":"..","side":"buy/sell",...}, ...] }
    out = []
    if not isinstance(data, dict):
        return out
    trades = data.get("data")
    if not isinstance(trades, list):
        return out
    for t in trades:
        p = _safe_float(t.get("px") or t.get("price"))
        q = _safe_float(t.get("sz") or t.get("size"))
        side = (t.get("side") or "").lower()
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=p * q, is_sell=(side == "sell")))
    return out


def parse_kucoin(data):
    out = []
    if not isinstance(data, dict):
        return out
    if data.get("code") != "200000":
        return out
    trades = data.get("data")
    if not isinstance(trades, list):
        return out
    for t in trades:
        p = _safe_float(t.get("price"))
        q = _safe_float(t.get("size"))
        side = (t.get("side") or "").lower()
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=p * q, is_sell=(side == "sell")))
    return out


def parse_bitstamp(data):
    out = []
    if not isinstance(data, list):
        return out
    for t in data:
        p = _safe_float(t.get("price"))
        q = _safe_float(t.get("amount"))
        typ = t.get("type")
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        # Bitstamp: type=0 buy, type=1 sell
        is_sell = False
        try:
            is_sell = int(typ) == 1
        except Exception:
            pass
        out.append(Trade(value=p * q, is_sell=is_sell))
    return out


def parse_gemini(data):
    out = []
    if not isinstance(data, list):
        return out
    for t in data:
        p = _safe_float(t.get("price"))
        q = _safe_float(t.get("amount"))
        typ = (t.get("type") or "").lower()
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=p * q, is_sell=(typ == "sell")))
    return out


def parse_gateio(data):
    out = []
    if not isinstance(data, list):
        return out
    for t in data:
        p = _safe_float(t.get("price"))
        q = _safe_float(t.get("amount"))
        side = (t.get("side") or "").lower()
        if p is None or q is None:
            continue
        if q < VOLUME_THRESHOLD:
            continue
        out.append(Trade(value=p * q, is_sell=(side == "sell")))
    return out


PARSERS = {
    "Binance": parse_binance,
    "Coinbase": parse_coinbase,
    "Kraken": parse_kraken,
    "Bitfinex": parse_bitfinex,
    "Huobi Global": parse_huobi,
    "OKX": parse_okx,
    "KuCoin": parse_kucoin,
    "Bitstamp": parse_bitstamp,
    "Gemini": parse_gemini,
    "Gate.io": parse_gateio,
}


def build_params(exchange: str):
    # Keep calls lighter where possible (reduces timeouts + rate limits)
    if exchange == "Binance":
        return {"symbol": EXCHANGE_SYMBOLS["Binance"], "limit": min(TOTAL_FETCH_LIMIT, 1000)}
    if exchange == "Coinbase":
        return {"limit": min(TOTAL_FETCH_LIMIT, 1000)}
    if exchange == "Kraken":
        return {"pair": EXCHANGE_SYMBOLS["Kraken"]}
    if exchange == "Bitfinex":
        return {"limit": min(TOTAL_FETCH_LIMIT, 1000)}
    if exchange == "Huobi Global":
        # Huobi supports size (typically max 200)
        return {"symbol": EXCHANGE_SYMBOLS["Huobi Global"], "size": 200}
    if exchange == "OKX":
        return {"instId": EXCHANGE_SYMBOLS["OKX"], "limit": 100}
    if exchange == "KuCoin":
        return {"symbol": EXCHANGE_SYMBOLS["KuCoin"]}
    if exchange == "Gemini":
        return {"limit_trades": 200}
    if exchange == "Gate.io":
        return {"currency_pair": EXCHANGE_SYMBOLS["Gate.io"], "limit": 200}
    return {}


# ---------------- Aggregation / processing ----------------
def process_trades(trades):
    sell_vals = [t.value for t in trades if t.is_sell]
    buy_vals = [t.value for t in trades if not t.is_sell]

    total_sell = sum(sell_vals)
    total_buy = sum(buy_vals)

    sell_vals.sort(reverse=True)
    buy_vals.sort(reverse=True)

    display_sell = _clip_trades(sell_vals, TABLE_SIZE)
    display_buy = _clip_trades(buy_vals, TABLE_SIZE)

    return total_sell, total_buy, display_sell, display_buy


@dataclass
class Snapshot:
    total_sell: float = 0.0
    total_buy: float = 0.0
    sell_values: list = None
    buy_values: list = None
    updated_at: float = 0.0

    def __post_init__(self):
        if self.sell_values is None:
            self.sell_values = []
        if self.buy_values is None:
            self.buy_values = []


class TradePoller(threading.Thread):
    """
    New logic:
    - Background thread polls exchanges
    - UI reads last snapshot (no network in Matplotlib update)
    """
    def __init__(self, session: requests.Session, interval: float):
        super().__init__(daemon=True)
        self.session = session
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = Snapshot()

    def stop(self):
        self._stop.set()

    def get_snapshot(self) -> Snapshot:
        with self._lock:
            # return a shallow copy (safe for UI thread)
            return Snapshot(
                total_sell=self._snapshot.total_sell,
                total_buy=self._snapshot.total_buy,
                sell_values=list(self._snapshot.sell_values),
                buy_values=list(self._snapshot.buy_values),
                updated_at=self._snapshot.updated_at,
            )

    def _fetch_one(self, exchange: str, url: str):
        parser = PARSERS.get(exchange)
        if parser is None:
            return []

        params = build_params(exchange)
        # jitter prevents "thundering herd" every 2 seconds
        time.sleep(random.uniform(0.0, 0.12))

        try:
            r = self.session.get(url, params=params, timeout=(4, 25))
            # If server sends HTML or empty response, .json() may fail
            data = r.json()
            return parser(data)
        except requests.exceptions.RequestException as e:
            LOGGER.log(exchange, f"Error fetching from {exchange}: {e}")
            return []
        except ValueError as e:
            LOGGER.log(exchange, f"Error parsing JSON from {exchange}: {e}")
            return []
        except Exception as e:
            LOGGER.log(exchange, f"Unexpected error in {exchange}: {e}")
            return []

    def run(self):
        # Modest parallelism reduces timeouts vs max_workers=len(EXCHANGES)
        import concurrent.futures
        max_workers = min(6, max(2, len(EXCHANGES) // 2))

        while not self._stop.is_set():
            t0 = time.time()
            all_trades = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(self._fetch_one, name, url): name
                    for name, url in EXCHANGES.items()
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        all_trades.extend(fut.result())
                    except Exception as e:
                        name = futures.get(fut, "Unknown")
                        LOGGER.log(name, f"Worker error in {name}: {e}")

            total_sell, total_buy, sell_values, buy_values = process_trades(all_trades)

            with self._lock:
                self._snapshot.total_sell = total_sell
                self._snapshot.total_buy = total_buy
                self._snapshot.sell_values = sell_values
                self._snapshot.buy_values = buy_values
                self._snapshot.updated_at = time.time()

            # sleep remainder
            elapsed = time.time() - t0
            sleep_for = max(0.0, self.interval - elapsed)
            self._stop.wait(sleep_for)


# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(4, 10))
fig.canvas.manager.set_window_title(f"{SYMBOL} Live Trades")
ax.axis("off")

fig.patch.set_facecolor(BASE_COLOR_ODD)
ax.set_facecolor(BASE_COLOR_ODD)
_apply_mpl_toolbar_theme(fig, bg=BASE_COLOR_ODD, fg="#D0D0D0")

session = make_session()
poller = TradePoller(session=session, interval=FETCH_INTERVAL)
poller.start()


def on_close(event):
    try:
        poller.stop()
    except Exception:
        pass


fig.canvas.mpl_connect("close_event", on_close)


def update(_frame):
    # Blink logic (same behavior)
    now = datetime.now()
    blink = (now.minute == 57 and now.second < 15)
    blink_toggle = (now.second % 2 == 0) if blink else False

    snap = poller.get_snapshot()

    sell_values = list(snap.sell_values)
    buy_values = list(snap.buy_values)

    # pad to TABLE_SIZE
    sell_values += [0] * (TABLE_SIZE - len(sell_values))
    buy_values += [0] * (TABLE_SIZE - len(buy_values))
    sell_values = sell_values[:TABLE_SIZE]
    buy_values = buy_values[:TABLE_SIZE]

    totals_row = [
        f"{int(round(snap.total_buy)):,}" if snap.total_buy > 0 else "",
        f"{int(round(snap.total_sell)):,}" if snap.total_sell > 0 else "",
    ]

    trade_rows = [
        [
            f"{int(round(bv)):,}" if bv > 0 else "",
            f"{int(round(sv)):,}" if sv > 0 else "",
        ]
        for sv, bv in zip(sell_values, buy_values)
    ]

    table_data = [totals_row] + trade_rows

    ax.clear()
    ax.axis("off")

    fig.patch.set_facecolor(BASE_COLOR_ODD)
    ax.set_facecolor(BASE_COLOR_ODD)

    table = ax.table(
        cellText=table_data,
        colLabels=["BUY", "SELL"],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.85, 0.85)

    # Style
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.2)
        row, col = key
        if row == 0:  # header
            cell.set_facecolor("#404040")
            cell.set_text_props(weight="bold", color="white")
        elif row == 1:  # totals
            cell.set_facecolor("pink")
            cell.set_text_props(weight="bold")
        else:
            base_color = BASE_COLOR_EVEN if (row % 2 == 0) else BASE_COLOR_ODD
            cell.set_facecolor(base_color)
            cell.set_text_props(color=("green" if col == 0 else "red"), weight="bold")

    # Highlights (rows start at index 2 in table because 0 header, 1 totals)
    for i in range(2, len(table_data) + 1):
        idx = i - 2
        for col, val in [(0, buy_values[idx]), (1, sell_values[idx])]:
            cell = table[i, col]
            if val > 500000:
                cell.set_facecolor(HIGHLIGHT_RED)
            elif val > 200000:
                cell.set_facecolor(HIGHLIGHT_BLUE)
            elif val > 100000:
                cell.set_facecolor(HIGHLIGHT_ORANGE)
            elif val > 50000:
                cell.set_facecolor(HIGHLIGHT_YELLOW)

    if blink and blink_toggle:
        for i in range(2, len(table_data) + 1):
            for col in (0, 1):
                table[i, col].set_facecolor("white")

    # Optional: show last update time (tiny)
    # ax.text(0.5, 0.01, f"Updated: {datetime.fromtimestamp(snap.updated_at).strftime('%H:%M:%S')}",
    #         transform=ax.transAxes, ha="center", va="bottom", color="#D0D0D0", fontsize=7)

    return []


ani = animation.FuncAnimation(
    fig,
    update,
    interval=500,  # UI refresh faster; fetching still happens every FETCH_INTERVAL seconds
    cache_frame_data=False
)

plt.show()