import random
import math
import time
import numpy as np
from multiprocessing import Pool, RawArray, cpu_count

N_CITIES = 200
POP_SIZE = 400
GENERATIONS = 400
TOURNAMENT_K = 5
MUTATION_RATE = 0.1
SEED = 42
BLOCK = 10

random.seed(SEED)
CITIES = [(random.random() * 100, random.random() * 100) for _ in range(N_CITIES)]
DIST = [[math.dist(a, b) for b in CITIES] for a in CITIES]
N_BLOCKS = math.ceil((POP_SIZE - 1) / BLOCK)

_pop = None
_fit = None


def _init(pop_buf, fit_buf):
    global _pop, _fit
    _pop = np.frombuffer(pop_buf, dtype=np.int32).reshape(2, POP_SIZE, N_CITIES)
    _fit = np.frombuffer(fit_buf, dtype=np.float64).reshape(2, POP_SIZE)



def route_length(route):
    return sum(DIST[route[i]][route[i - 1]] for i in range(len(route)))


def tournament(fit):
    return min(random.sample(range(POP_SIZE), TOURNAMENT_K), key=fit.__getitem__)


def ox(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    child = [None] * n
    child[a:b] = p1[a:b]
    used = set(p1[a:b])
    fill = [g for g in p2[b:] + p2[:b] if g not in used]

    for i, pos in enumerate(list(range(b, n)) + list(range(0, a))):
        child[pos] = fill[i]

    return child


def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route


def make_children(task):
    cur, gen, b0, b1 = task
    pop, fit = _pop[cur], _fit[cur].tolist()
    out_pop, out_fit = _pop[1 - cur], _fit[1 - cur]

    for b in range(b0, b1):
        random.seed(SEED * 1_000_003 + gen * 1000 + b)

        for i in range(1 + b * BLOCK, min(1 + (b + 1) * BLOCK, POP_SIZE)):
            p1 = pop[tournament(fit)].tolist()
            p2 = pop[tournament(fit)].tolist()
            child = mutate(ox(p1, p2))
            out_pop[i] = child
            out_fit[i] = route_length(child)


def ga(parallel=False, workers=None):
    workers = (workers or cpu_count()) if parallel else 1
    pop_buf = RawArray('i', 2 * POP_SIZE * N_CITIES)
    fit_buf = RawArray('d', 2 * POP_SIZE)
    _init(pop_buf, fit_buf)

    random.seed(SEED)
    for i in range(POP_SIZE):
        route = random.sample(range(N_CITIES), N_CITIES)
        _pop[0][i] = route
        _fit[0][i] = route_length(route)

    pool = Pool(workers, initializer=_init, initargs=(pop_buf, fit_buf)) if parallel else None
    per = math.ceil(N_BLOCKS / workers)
    try:
        cur = 0
        for gen in range(GENERATIONS):
            best = int(_fit[cur].argmin())
            _pop[1 - cur][0] = _pop[cur][best]
            _fit[1 - cur][0] = _fit[cur][best]

            tasks = [(cur, gen, b, min(b + per, N_BLOCKS)) for b in range(0, N_BLOCKS, per)]
            if pool:
                pool.map(make_children, tasks, chunksize=1)
            else:
                make_children(tasks[0])
            cur = 1 - cur
        return float(_fit[cur].min())
    finally:
        if pool:
            pool.close()
            pool.join()


if __name__ == "__main__":
    t = time.perf_counter()
    best_seq = ga(parallel=False)
    t_seq = time.perf_counter() - t
    print(f"Jednowatkowo:   najlepsza trasa = {best_seq:.2f}, czas = {t_seq:.2f} s")

    t = time.perf_counter()
    best_par = ga(parallel=True)
    t_par = time.perf_counter() - t
    print(f"Rownolegle ({cpu_count()}): najlepsza trasa = {best_par:.2f}, czas = {t_par:.2f} s")

    print(f"Przyspieszenie: {t_seq / t_par:.2f}x")