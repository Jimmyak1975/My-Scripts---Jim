import json
import threading
import websocket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timezone

# ---------------- Configuration ----------------
SYMBOL = "ltcusdt"  # Use lowercase for the websocket endpoint.
WS_URL = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"
FETCH_INTERVAL = 1000   # Update the table every 1000 ms (1 second)
TABLE_SIZE = 100        # Display the last 100 CVD snapshots

# Colors for alternating rows (base dark theme)
BASE_COLOR_EVEN = "#2E2E2E"  # Dark gray
BASE_COLOR_ODD  = "#1E1E1E"  # Almost black

# ---------------- Global Variables ----------------
data_lock = threading.Lock()
cumulative_cvd = 0.0      # Running cumulative CVD, updated in real time
cvd_history = []          # History of cumulative CVD snapshots (one per table update)

# ---------------- WebSocket Callback Functions ----------------
def on_message(ws, message):
    global cumulative_cvd
    try:
        data = json.loads(message)
        # Data fields: "T" (trade time in ms), "p" (price, str), "q" (quantity, str), "m" (bool)
        price = float(data["p"])
        qty = float(data["q"])
        trade_value = price * qty
        # If "m" is False, buyer is aggressive (add volume); if True, seller is aggressive (subtract volume).
        side = 1 if not data.get("m", False) else -1
        delta = side * trade_value
        with data_lock:
            cumulative_cvd += delta
    except Exception as e:
        print("Error processing message:", e)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed:", close_status_code, close_msg)

def on_open(ws):
    print("WebSocket connection opened.")

def run_ws():
    ws = websocket.WebSocketApp(WS_URL,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# Start the WebSocket in a separate thread.
ws_thread = threading.Thread(target=run_ws)
ws_thread.daemon = True
ws_thread.start()

# ---------------- Plot Setup ----------------
fig, ax = plt.subplots(figsize=(3, 12))
try:
    fig.canvas.manager.set_window_title("LTCUSDT - CVD (WebSocket)")
except Exception:
    pass
ax.axis("off")

def update(frame):
    global cumulative_cvd, cvd_history
    with data_lock:
        current_value = cumulative_cvd
    # Append the current cumulative CVD to history.
    cvd_history.append(current_value)
    # Keep only the last TABLE_SIZE snapshots.
    if len(cvd_history) > TABLE_SIZE:
        cvd_history = cvd_history[-TABLE_SIZE:]
    # Build table data: one column, formatted with 2 decimal places.
    table_data = [[f"{val:,.2f}"] for val in cvd_history]
    if len(table_data) < TABLE_SIZE:
        table_data = [[""]] * (TABLE_SIZE - len(table_data)) + table_data

    # Draw the table.
    ax.clear()
    ax.axis("off")
    col_labels = ["CVD"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.85, 0.85)

    # Set borders and alternating row colors.
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.2)
        if row == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(weight="bold", color="white")
        else:
            base_color = BASE_COLOR_EVEN if row % 2 == 0 else BASE_COLOR_ODD
            cell.set_facecolor(base_color)

    # Color-code values by comparing each row with the previous one.
    first_data_idx = None
    for i in range(TABLE_SIZE):
        if table_data[i][0] != "":
            first_data_idx = i
            break
    if first_data_idx is not None:
        table[first_data_idx + 1, 0].set_text_props(color="white")
        for i in range(first_data_idx + 1, TABLE_SIZE):
            if table_data[i][0] == "":
                continue
            try:
                current_val = float(table_data[i][0].replace(",", ""))
                previous_val = float(table_data[i-1][0].replace(",", ""))
            except Exception:
                current_val = previous_val = 0
            if current_val > previous_val:
                table[i+1, 0].set_text_props(color="green")
            elif current_val < previous_val:
                table[i+1, 0].set_text_props(color="red")
            else:
                table[i+1, 0].set_text_props(color="white")

ani = animation.FuncAnimation(fig, update, interval=FETCH_INTERVAL, cache_frame_data=False)
plt.show()
