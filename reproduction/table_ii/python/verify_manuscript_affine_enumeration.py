"""Small exact checks of manuscript Algorithms 1 and 2, not a speed benchmark.

Uses no training data. Gaussian solves and syndrome tables implement the
stated quotient/reduction/lifting construction; direct input enumeration is
used only as a separate oracle. Lengths 1, 2, 4 are exhaustive over canonical
causal supports, while length 8 uses reproducible synthetic checks.
"""

def parity(value):
    return value.bit_count() & 1

def combine(columns, value):
    result = 0
    for j, column in enumerate(columns):
        if value >> j & 1:
            result ^= column
    return result

def project(value, positions):
    return sum(((value >> j) & 1) << t for t, j in enumerate(positions))

def solve(equations, width):
    rows = {}
    for row, rhs in equations:
        for pivot in sorted(rows):
            if row >> pivot & 1:
                other, bit = rows[pivot]
                row ^= other
                rhs ^= bit
        if not row:
            if rhs:
                return None
            continue
        pivot = (row & -row).bit_length() - 1
        for old, (other, bit) in tuple(rows.items()):
            if other >> pivot & 1:
                rows[old] = (other ^ row, bit ^ rhs)
        rows[pivot] = (row, rhs)
    origin = sum(rhs << pivot for pivot, (_, rhs) in rows.items())
    basis = tuple((1 << free) | sum(((row >> free) & 1) << pivot
                                  for pivot, (row, _) in rows.items())
                  for free in range(width) if free not in rows)
    return origin, basis
