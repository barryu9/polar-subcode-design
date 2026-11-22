"""Algorithm 1. List polar *inputs* u whose encoded words have weight <= D.

Run from the repository root: python -m algorithms.algorithm1_directory
An integer stores a binary vector with physical coordinate i in bit i.
"""
from itertools import combinations

from polar_subcodes.algebra import polar_transform
from polar_subcodes.code import Design, PolarSubcode
from polar_subcodes.spectrum.quotient import ImageFactor, embed, gray_span


def xor_selected(columns, mask):
    """Multiply a binary matrix (stored by columns) by a packed vector."""
    result = 0
    while mask:
        bit = mask & -mask
        result ^= columns[bit.bit_length() - 1]
        mask ^= bit
    return result


def from_causal_checks(n, rows):
    """Convert homogeneous causal check rows on u to canonical expressions.

    Supply each row as an integer mask; its highest set bit is the frozen
    position. Substitute earlier frozen expressions in increasing pivot order,
    leaving only earlier information positions in each right-hand side.
    """
    Design(n, ())
    rows = tuple(rows)
    if any(type(row) is not int or row <= 0 or row >> n for row in rows):
        raise ValueError("Each check must be a nonzero N-bit integer.")
    checks = {row.bit_length() - 1: row for row in rows}
    if len(checks) != len(rows):
        raise ValueError("A causal check presentation needs distinct frozen pivots.")
    columns = [0] * n
    for j in sorted(checks):
        rhs = checks[j] ^ (1 << j)
        for i in range(j):
            if i in checks and rhs >> i & 1:
                rhs ^= (1 << i) ^ columns[i]
        columns[j] = rhs
    return PolarSubcode(n, tuple(i for i in range(n) if i not in checks), tuple(columns))


def split_code(code):
    """Return the two fixed half codes and the outer columns in source order."""
    h = code.n // 2
    left = PolarSubcode(h, tuple(i for i in code.profile if i < h), code.columns[:h])
    right = PolarSubcode(h, tuple(i - h for i in code.profile if i >= h),
                         tuple(column >> h for column in code.columns[h:]))
    frozen = [j for j in range(h) if j not in right.profile]
    outer = tuple(sum(((code.columns[h+j] >> i) & 1) << q for q, j in enumerate(frozen))
                  for i in left.profile)
    return left, right, outer


def right_checks(right):
    """Columns of P = H_R G_h^T, indexed by physical output coordinate."""
    rows = right.checks
    columns = []
    for j in range(right.n):
        input_word = polar_transform(1 << j, right.n)
        columns.append(sum(((check & input_word).bit_count() % 2) << r
                           for r, check in enumerate(rows)))
    return tuple(columns)


def outside_masks(positions, radius):
    for weight in range(min(radius, len(positions)) + 1):
        for subset in combinations(positions, weight):
            yield sum(1 << j for j in subset)


def list_inputs(code, cutoff, tick=lambda: None):
    """Yield each input exactly once. tick() lets the caller interrupt work."""
    tick()
    if cutoff < 0:
        return
    if code.n == 1:
        yield 0
        if code.k and cutoff >= 1:
            yield 1
        return
    left, right, outer = split_code(code)
    h = right.n
    # u_L=0 produces a repeated right codeword: each occupied position costs two.
    for u_right in list_inputs(right, cutoff // 2, tick):
        yield u_right << h
    if not left.profile or left.d_P > cutoff:
        return

    columns = right_checks(right)
    for u_left in list_inputs(left, cutoff, tick):
        if u_left == 0:
            continue
        message = sum(((u_left >> i) & 1) << k for k, i in enumerate(left.profile))
        target = xor_selected(outer, message)
        a = polar_transform(u_left, h)
        inside = [j for j in range(h) if a >> j & 1]
        outside = [j for j in range(h) if not (a >> j & 1)]
        factor = ImageFactor([columns[j] for j in inside])
        # Elimination supplies a quotient test, an inside solution and its kernel.
        for z in outside_masks(outside, (cutoff - len(inside)) // 2):
            tick()
            remainder, origin = factor.reduce(target ^ xor_selected(columns, z))
            if remainder:
                continue
            for delta in gray_span(factor.kernel):
                tick()
                x_right = z | embed(origin ^ delta, inside)
                # The transform is its own inverse over F_2.
                yield u_left | (polar_transform(x_right, h) << h)


if __name__ == "__main__":
    demo = PolarSubcode(4, (1, 3), (0, 0, 2, 0))
    for u in list_inputs(demo, 2):
        x = polar_transform(u, demo.n)
        print(f"u={u:04b}, x={x:04b}, weight={x.bit_count()} (coordinate zero on the right)")
