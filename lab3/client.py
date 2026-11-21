import socket
import pickle
import struct
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SERVER_IP = "10.182.145.117"
PORT = 2040

KX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

KY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])


def send_all(sock, data):
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def receive_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        packet = sock.recv(min(65536, size - len(data)))

        if not packet:
            raise ConnectionError("Połączenie zostało przerwane.")

        data.extend(packet)

    return bytes(data)


def receive_all(sock):
    header = receive_exact(sock, 8)
    length = struct.unpack("!Q", header)[0]

    if length > 500_000_000:
        raise ValueError("Otrzymano zbyt duży komunikat.")

    payload = receive_exact(sock, length)
    return pickle.loads(payload)


def sobel_filter(image_array):
    image_array = image_array.astype(np.float32)

    padded = np.pad(image_array, 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))

    gx = np.sum(windows * KX, axis=(-2, -1))
    gy = np.sum(windows * KY, axis=(-2, -1))

    magnitude = np.sqrt(gx**2 + gy**2)

    return np.clip(magnitude, 0, 255).astype(np.uint8)


def client_main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        print(f"Łączenie z serwerem {SERVER_IP}:{PORT}...")
        client_socket.connect((SERVER_IP, PORT))
        print("Połączono z serwerem.")

        task = receive_all(client_socket)

        fragment = task["fragment"]
        y_start = task["y_start"]
        crop_top = task["crop_top"]
        core_height = task["core_height"]

        print(f"Otrzymano fragment od wiersza {y_start}.")

        processed = sobel_filter(fragment)

        processed = processed[
            crop_top:crop_top + core_height, :
        ]

        send_all(client_socket, {
            "y_start": y_start,
            "fragment": processed
        })

        print("Wynik wysłany do serwera.")


if __name__ == "__main__":
    client_main()
