import websocket

def on_message(ws, message):
    print(f"Received")

def on_error(ws, error):
    print(f"Error")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connected to the WebSocket server")
    ws.send("Hello, WebSocket!")  # Send a test message

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:9001",
                                 on_open=on_open,
                                 on_message=on_message,
                                 on_error=on_error,
                                 on_close=on_close)
    ws.run_forever()
