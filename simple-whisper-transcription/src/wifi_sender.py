import socket
import threading

HOST = "0.0.0.0"      # Listen on all interfaces
PORT = 5000

_client = None
_server_started = False


def _server_thread():
    global _client

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((HOST, PORT))

    server.listen(1)

    print(f"[WiFi] Waiting for Arduino on port {PORT}...")

    while True:
        client, address = server.accept()

        print(f"[WiFi] Arduino connected from {address}")

        _client = client


def start_server():
    global _server_started

    if _server_started:
        return

    thread = threading.Thread(
        target=_server_thread,
        daemon=True
    )

    thread.start()

    _server_started = True


def send_csv(intent, subject, action, concept):

    global _client

    if _client is None:
        return

    message = f"{intent},{subject},{action},{concept}\n"

    try:
        _client.sendall(message.encode("utf-8"))

        print(f"[WiFi SENT] {message.strip()}")

    except Exception as e:

        print("[WiFi] Client disconnected.")

        _client.close()

        _client = None
