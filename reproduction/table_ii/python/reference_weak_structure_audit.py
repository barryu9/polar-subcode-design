"""Build binary check rows in codeword coordinates for a fixed profile."""

from functools import lru_cache

def binary_rank(rows):
    """Rank of the binary check rows, encoded as integers."""
    pivots = {}
    for value in rows:
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)

@lru_cache(maxsize=8)
def polar_columns(n):
    return tuple(sum(int(t & j == j) << t for t in range(n)) for j in range(n))

def check_rows(profile, columns):
    n = len(columns)
    inverse_columns = polar_columns(n)
    checks = []
    for j in range(n):
        if j in profile:
            continue
        row = inverse_columns[j]
        for i in profile:
            if columns[j] >> i & 1:
                row ^= inverse_columns[i]
        checks.append(row)
    return checks
