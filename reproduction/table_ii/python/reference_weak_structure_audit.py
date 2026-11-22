"""Compare successful references with profile-matched controls, without search."""

from functools import lru_cache
from audit_success_support_structure import binary_rank

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
