"""Sample plans on the unit hypercube [0, 1)^d (standard library only).

halton(): scrambled Halton sequence - a low-discrepancy ("space-filling") sequence like
Sobol. It is deterministic for a given seed and extensible: points skip..skip+n-1 can be
added later without changing the earlier ones, so a dataset can grow in stages.
random_points(): plain uniform random points, for hold-out (test) sets.
"""
import random

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def halton(n, dim, seed=0, skip=0):
    if dim > len(PRIMES):
        raise ValueError(f"at most {len(PRIMES)} dimensions")
    rng = random.Random(seed)
    perms = []
    for b in PRIMES[:dim]:
        rest = list(range(1, b))
        rng.shuffle(rest)
        perms.append([0] + rest)            # digit permutation per base (0 kept fixed)
    pts = []
    for i in range(skip + 1, skip + n + 1):  # index 0 would be the all-zero corner
        row = []
        for d, b in enumerate(PRIMES[:dim]):
            x, f, k = 0.0, 1.0 / b, i
            while k:
                x += perms[d][k % b] * f
                k //= b
                f /= b
            row.append(x)
        pts.append(row)
    return pts


def random_points(n, dim, seed=0):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(dim)] for _ in range(n)]
