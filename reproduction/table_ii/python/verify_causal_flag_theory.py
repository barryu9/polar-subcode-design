"""Exact checks of causal-flag polar-subcode theory, without training data.

Enumerates every profile/support and every input subspace at lengths 1, 2, 4.
Larger lengths only check transform/coset identities, not search performance.
"""

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
