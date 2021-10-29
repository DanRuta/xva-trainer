# Import WebSocket client library (and others)
import websocket
import _thread
import time

# Define WebSocket callback functions
def ws_message(ws, message):
    print("WebSocket thread: %s" % message)

def ws_open(ws):
    ws.send('{"event":"subscribe", "subscription":{"name":"trade"}, "pair":["XBT/USD","XRP/USD"]}')

def ws_thread(*args):
    ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_open = ws_open, on_message = ws_message)
    ws.run_forever()

# Start a new thread for the WebSocket interface
_thread.start_new_thread(ws_thread, ())

# Continue other (non WebSocket) tasks in the main thread
while True:
    time.sleep(5)
    print("Main thread: %d" % time.time())