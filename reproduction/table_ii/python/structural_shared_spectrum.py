"""Low-weight fibers from quadratic syndromes and one cubic check.

The formulas apply to a codimension-one subcode of RM(m-3,m).
The design family uses fixed causal coordinates, without reference coefficients.
"""

from itertools import combinations

def alternating_rows(edge_bits, m):
    """Convert polar input coefficients of degree-two checks into a matrix."""
    rows = [0] * m
    for a, b in combinations(range(m), 2):
        index = (1 << a) | (1 << b)
        if edge_bits >> index & 1:
            rows[a] |= 1 << b
            rows[b] |= 1 << a
    return tuple(rows)
