import socket
import pickle
import struct
import time
import numpy as np
from PIL import Image

HOST = "0.0.0.0"
PORT = 2040


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


def split_image(image_array, n_clients):
    height = image_array.shape[0]
    tasks = []

    for i in range(n_clients):
        y_start = i * height // n_clients
        y_end = (i + 1) * height // n_clients

        if y_start == y_end:
            raise ValueError("Liczba klientów nie może przekraczać wysokości obrazu.")

        ext_start = max(0, y_start - 1)
        ext_end = min(height, y_end + 1)

        tasks.append({
            "fragment": image_array[ext_start:ext_end, :],
            "y_start": y_start,
            "crop_top": y_start - ext_start,
            "core_height": y_end - y_start
        })

    return tasks


def merge_image(processed_fragments, height, width):
    result = np.zeros((height, width), dtype=np.uint8)

    for item in processed_fragments:
        fragment = item["fragment"]
        y_start = item["y_start"]

        result[y_start:y_start + fragment.shape[0], :] = fragment

    return Image.fromarray(result)


def server_main(image_path, n_clients):
    image = Image.open(image_path).convert("L")
    image_array = np.array(image)

    height, width = image_array.shape
    print(f"Wczytano obraz {width} x {height}.")

    tasks = split_image(image_array, n_clients)

    # 3. Uruchamiamy serwer.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(n_clients)

        print(f"Serwer nasłuchuje na porcie {PORT}.")

        clients = []

        for i in range(n_clients):
            client_socket, address = server_socket.accept()
            client_socket.settimeout(120)
            clients.append(client_socket)
            print(f"Klient {i + 1} połączony: {address}")

        try:
            start = time.perf_counter()

            for i, client_socket in enumerate(clients):
                send_all(client_socket, tasks[i])
                print(f"Wysłano fragment {i + 1}.")

            processed_fragments = []

            for i, client_socket in enumerate(clients):
                result = receive_all(client_socket)
                processed_fragments.append(result)
                print(f"Odebrano wynik od klienta {i + 1}.")

            elapsed = time.perf_counter() - start

            result_image = merge_image(
                processed_fragments, height, width
            )

            result_image.save("processed_image.png")

            print("Zapisano processed_image.png")
            print(f"Czas komunikacji i przetwarzania: {elapsed:.4f} s")

        finally:
            for client_socket in clients:
                client_socket.close()


if __name__ == "__main__":
    server_main("obraz2.jpg", n_clients=1)
