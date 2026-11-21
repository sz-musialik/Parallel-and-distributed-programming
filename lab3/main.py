from multiprocessing import Pool, cpu_count
from PIL import Image
import numpy as np
import time


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

KX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

KY = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

def sobel_filter(image_array):
    image_array = image_array.astype(np.float32)
    p = np.pad(image_array, 1, mode="edge")

    windows = sliding_window_view(p, (3, 3))

    gx = np.sum(windows * KX, axis=(-2, -1))
    gy = np.sum(windows * KY, axis=(-2, -1))

    magnitude = np.sqrt(gx**2 + gy**2)

    return np.clip(magnitude, 0, 255).astype(np.uint8)

def process_chunk(task):
    tile, y_start, y_end, crop_start = task

    filtered = sobel_filter(tile)

    # Usuwamy dodatkowe wiersze, które służyły jako margines.
    filtered = filtered[
        crop_start:crop_start + (y_end - y_start), :
    ]

    return y_start, filtered

def create_tasks(image_array, number_of_chunks):
    """Dzieli obraz na poziome fragmenty z marginesem 1 piksela."""
    height = image_array.shape[0]
    tasks = []

    chunk_height = max(1, (height + number_of_chunks - 1)
                       // number_of_chunks)

    for y_start in range(0, height, chunk_height):
        y_end = min(y_start + chunk_height, height)

        # Dodatkowy wiersz nad i pod fragmentem.
        ext_start = max(0, y_start - 1)
        ext_end = min(height, y_end + 1)

        tile = image_array[ext_start:ext_end, :]

        tasks.append((
            tile,
            y_start,
            y_end,
            y_start - ext_start
        ))

    return tasks


def main():
    image_path = "obraz3.jpg"

    # 1. Wczytanie obrazu i konwersja do skali szarości.
    image = Image.open(image_path).convert("L")
    image_array = np.array(image)

    height, width = image_array.shape

    print(f"Wczytano obraz: {width} x {height} pikseli")

    # Ustalamy liczbę procesów i fragmentów.
    number_of_processes = min(4, cpu_count(), height)
    tasks = create_tasks(image_array, number_of_processes)

    print(f"Liczba procesów: {number_of_processes}")
    print(f"Liczba fragmentów: {len(tasks)}")

    # 2. Przetwarzanie sekwencyjne - jeden po drugim.
    start = time.perf_counter()

    sequential_results = [
        process_chunk(task) for task in tasks
    ]

    sequential_image = np.zeros_like(image_array)

    for y_start, fragment in sequential_results:
        sequential_image[
            y_start:y_start + fragment.shape[0], :
        ] = fragment

    sequential_time = time.perf_counter() - start

    # 3. Przetwarzanie równoległe w osobnych procesach.
    start = time.perf_counter()

    with Pool(processes=number_of_processes) as pool:
        parallel_results = pool.map(process_chunk, tasks)

    parallel_image = np.zeros_like(image_array)

    for y_start, fragment in parallel_results:
        parallel_image[
            y_start:y_start + fragment.shape[0], :
        ] = fragment

    parallel_time = time.perf_counter() - start

    # 4. Zapis obrazu wynikowego.
    Image.fromarray(parallel_image).save("wynik.png")

    print("Zapisano wynik.png")
    print(f"Czas sekwencyjny: {sequential_time:.4f} s")
    print(f"Czas równoległy:  {parallel_time:.4f} s")

    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"Przyspieszenie: {speedup:.2f}x")


if __name__ == "__main__":
    main()
