"""Binary linear algebra; bit j always denotes coordinate j (zero based)."""
from functools import lru_cache


@lru_cache(maxsize=20)
def butterfly_masks(n):
    return tuple((s, sum(((1 << s) - 1) << j for j in range(0, n, 2 * s)))
                 for s in (1 << k for k in range(n.bit_length() - 1)))


def polar(value, n, transpose=False):
    for shift, mask in butterfly_masks(n):
        value ^= (value & mask) << shift if transpose else (value >> shift) & mask
    return value


def insert(basis, row):
    while row:
        j = row.bit_length() - 1
        if j not in basis:
            basis[j] = row
            return True
        row ^= basis[j]
    return False


def reduce_vector(basis, row):
    for j in sorted(basis, reverse=True):
        if row >> j & 1:
            row ^= basis[j]
    return row


def canonical(rows):
    basis = {}
    for row in rows:
        insert(basis, row)
    return basis


def generators(n, checks):
    for i in range(n):
        if i in checks:
            continue
        u = 1 << i
        for j in sorted(checks):
            if (u & checks[j]).bit_count() & 1:
                u ^= 1 << j
        yield i, polar(u, n)


def prepare(n, rows, punctured, shortened):
    checks = canonical(rows)
    original = n - len(checks)
    checks = canonical(list(checks.values()) + [polar(1 << j, n, True) for j in shortened])
    shortened_dimension = n - len(checks)
    kept = ((1 << n) - 1) ^ sum(1 << j for j in punctured + shortened)
    projected, selected = {}, set()
    words = list(generators(n, checks))
    for i, word in reversed(words):
        if insert(projected, word & kept):
            selected.add(i)
    for i, _ in words:
        if i not in selected:
            checks[i] = 1 << i
    return checks, dict(original_dimension=original,
                        shortened_dimension=shortened_dimension,
                        transmitted_dimension=len(selected),
                        kernel_dimension=shortened_dimension - len(selected))


def kernel_directions(free, pivots):
    independent = [j for j in range(free.bit_length()) if free >> j & 1 and j not in pivots]
    directions = []
    for j in independent:
        v = 1 << j
        for pivot in sorted(pivots):
            row = pivots[pivot][0] & free
            if (row & v).bit_count() & 1:
                v ^= 1 << pivot
        directions.append(v)
    return directions
