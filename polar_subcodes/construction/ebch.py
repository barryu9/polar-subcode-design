"""Construct Section VI.A's eBCH polar subcode in natural coordinate order.

An eBCH word satisfies moment checks over GF(2^m). Expressing each check
in the polar input u gives a binary equation. Eliminate from right to left
so that each frozen position depends only on earlier information positions.
"""
import numpy as np

from ..code import PolarSubcode
from .profiles import reliability_order


def ebch_subcode(n, k, distance, polynomial, ebn0_db=2.0):
    """Choose the K best GA positions among the free positions of the parent."""
    if type(n) is not int or not 2 <= n <= 16384 or n & (n-1):
        raise ValueError("Use a power-of-two length 2 through 16384")
    if not 1 <= k <= n or not 2 <= distance <= n or distance % 2:
        raise ValueError("Expected a valid dimension and an even designed distance")
    if polynomial.bit_length() != n.bit_length() or not polynomial & 1:
        raise ValueError("Invalid field polynomial")
    exponents = []
    log = {}
    x = 1
    for power in range(n-1):
        if x in log:
            raise ValueError("The field polynomial must have primitive element x")
        log[x] = power
        exponents.append(x)
        x <<= 1
        if x & n:
            x ^= polynomial
    if x != 1:
        raise ValueError("The multiplicative group did not close")

    # Even parity is u[0] = 0. Conjugate moments yield the same binary checks.
    rows = [1]
    seen = set()
    for degree in range(1, distance-1):
        if degree in seen:
            continue
        power = degree
        while power not in seen:
            seen.add(power)
            power = (2*power) % (n-1)
        values = np.array([0] + [exponents[(degree*log[j]) % (n-1)] for j in range(1,n)],
                          dtype=np.uint16)
        # Transform the output-coordinate check through x = u G_N.
        step = 1
        while step < n:
            blocks = values.reshape(-1, 2*step)
            blocks[:, step:] ^= blocks[:, :step]
            step *= 2
        for bit in range(n.bit_length()-1):
            packed = np.packbits(((values >> bit) & 1).astype(np.uint8), bitorder="little")
            rows.append(int.from_bytes(packed.tobytes(), "little"))

    checks = {}
    for row in rows:
        for pivot in sorted(checks, reverse=True):
            if row >> pivot & 1:
                row ^= checks[pivot]
        if row:
            pivot = row.bit_length()-1
            for earlier in checks:
                if checks[earlier] >> pivot & 1:
                    checks[earlier] ^= row
            checks[pivot] = row
    order = reliability_order(n, "ga", rate=k/n, ebn0_db=ebn0_db)
    free = [i for i in order if i not in checks]
    if len(free) < k:
        raise ValueError("eBCH parent has insufficient dimension")
    profile = tuple(sorted(free[:k]))
    mask = sum(1 << i for i in profile)
    return PolarSubcode(n, profile, tuple(checks.get(j, 0) & mask for j in range(n)))
