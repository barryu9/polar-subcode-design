"""Binary linear algebra and the row-vector polar transform."""
from functools import lru_cache

def leading(value):
    if not value:
        raise ValueError('Zero has no leading position')
    return (value & -value).bit_length() - 1


def polar_transform(value, n):
    bits = [(value >> i) & 1 for i in range(n)]
    step = 1
    while step < n:
        for start in range(0, n, 2 * step):
            for j in range(start, start + step):
                bits[j] ^= bits[j + step]
        step *= 2
    return sum(bit << i for i, bit in enumerate(bits))


def rref(vectors):
    rows = {}
    for value in vectors:
        for pivot in sorted(rows):
            if (value >> pivot) & 1:
                value ^= rows[pivot]
        if value:
            pivot = leading(value)
            for old in rows:
                if (rows[old] >> pivot) & 1:
                    rows[old] ^= value
            rows[pivot] = value
    return tuple(rows[pivot] for pivot in sorted(rows))


def span(basis):
    points = {0}
    for row in basis:
        points |= {point ^ row for point in tuple(points)}
    return frozenset(points)


def residual(value, basis):
    for row in basis:
        if (value >> leading(row)) & 1:
            value ^= row
    return value


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


def generator(profile, columns):
    n = len(columns)
    return [polar_transform((1 << i) | sum(((value >> i) & 1) << j
                                         for j, value in enumerate(columns)), n)
            for i in profile]


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


def all_words(profile, columns):
    result = []
    for message in range(1 << len(profile)):
        u = sum(((message >> q) & 1) << i for q, i in enumerate(profile))
        for j, support in enumerate(columns):
            if (u & support).bit_count() & 1:
                u |= 1 << j
        result.append((u, polar_transform(u, len(columns)).bit_count()))
    return result



def all_subspaces(n):
    known = {()}
    frontier = {()}
    while frontier:
        following = {rref(basis + (value,)) for basis in frontier for value in range(1, 2**n)} - known
        known |= following
        frontier = following
    return sorted(known, key=lambda basis: (len(basis), basis))


def gf_mul(a, b, n, polynomial):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & n:
            a ^= polynomial
    return result
