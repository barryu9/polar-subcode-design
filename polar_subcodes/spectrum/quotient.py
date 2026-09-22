"""Complete word directories with direct or weight-free quotient extensions."""
from itertools import combinations
from collections import Counter
import hashlib
import math
from polar_subcodes.algebra import polar_transform

def gray_span(basis):
    value = 0
    yield value
    for index in range(1, 1 << len(basis)):
        value ^= basis[(index & -index).bit_length()-1]
        yield value


def embed(value, positions):
    return sum(((value >> q) & 1) << j for q, j in enumerate(positions))


class ImageFactor:
    """One column elimination, reused for every right-hand side."""
    def __init__(self, columns):
        self.pivots = {}
        self.kernel = []
        for index, column in enumerate(columns):
            residual, preimage = self.reduce(column)
            label = preimage ^ (1 << index)
            if residual:
                self.pivots[residual.bit_length()-1] = residual, label
            else:
                self.kernel.append(label)
        self.rank = len(self.pivots)

    def reduce(self, value):
        label = 0
        for pivot in sorted(self.pivots, reverse=True):
            if value >> pivot & 1:
                other, preimage = self.pivots[pivot]
                value ^= other
                label ^= preimage
        return value, label


def right_data(half, coupled):
    mask = (1 << half)-1
    physical_rows = [polar_transform(1 << j, half) for j in range(half)]
    columns = tuple(sum((((row >> half) & word).bit_count() & 1) << q
                        for q, row in enumerate(coupled))
                    for word in physical_rows)
    return tuple(row & mask for row in coupled), columns


def extend(half, left, left_checks, columns, cutoff, method, stats):
    s = polar_transform(left, half)
    inside = tuple(j for j in range(half) if s >> j & 1)
    outside = tuple(j for j in range(half) if not s >> j & 1)
    radius = (cutoff-s.bit_count())//2
    if radius < 0:
        return []
    rhs = sum(((row & left).bit_count() & 1) << q
              for q, row in enumerate(left_checks))
    outside_count = sum(math.comb(len(outside), t)
                        for t in range(min(radius, len(outside))+1))
    predicted = outside_count * (1 << len(inside))
    stats["direct_candidates_equivalent"] += predicted
    if method == "direct" and stats["direct_candidates_equivalent"] > stats["limit"]:
        raise RuntimeError("Declared direct-candidate limit exceeded")
    stats["extensions"] += 1
    stats["outside_vectors"] += outside_count
    if stats["outside_vectors"] > stats["limit"]:
        raise RuntimeError("Declared outside-vector limit exceeded")
    outputs = []
    inside_columns = tuple(columns[j] for j in inside)
    if method == "quotient":
        factor = ImageFactor(inside_columns)
        kernel = tuple(embed(v, inside) for v in factor.kernel)
        stats["factorizations"] += 1
    elif method != "direct":
        raise ValueError("Expected direct or quotient")

    for weight in range(min(radius, len(outside))+1):
        for positions in combinations(outside, weight):
            e, syndrome = 0, rhs
            for j in positions:
                e ^= 1 << j
                syndrome ^= columns[j]
            if method == "quotient":
                residual, z = factor.reduce(syndrome)
                stats["quotient_tests"] += 1
                if residual:
                    continue
                stats["consistent_outside"] += 1
                origin = embed(z, inside) ^ e
                for direction in gray_span(kernel):
                    if stats["emitted_at_all_levels"]+len(outputs) >= stats["limit"]:
                        raise RuntimeError("Declared word-materialization limit exceeded")
                    b = polar_transform(origin ^ direction, half)
                    outputs.append(left | (b << half))
            else:
                z, value, found = 0, 0, False
                for index in range(1 << len(inside)):
                    if index:
                        bit = (index & -index).bit_length()-1
                        z ^= 1 << inside[bit]
                        value ^= inside_columns[bit]
                    if value == syndrome:
                        if stats["emitted_at_all_levels"]+len(outputs) >= stats["limit"]:
                            raise RuntimeError("Declared word-materialization limit exceeded")
                        b = polar_transform(z ^ e, half)
                        outputs.append(left | (b << half))
                        found = True
                stats["consistent_outside"] += int(found)
    stats["emitted_at_all_levels"] += len(outputs)
    return outputs


def directory(n, checks, cutoff, method, stats):
    if cutoff < 0:
        return []
    if n == 1:
        return [u for u in (0, 1) if u <= cutoff
                and all(not ((row & u).bit_count() & 1) for row in checks)]
    half = n//2
    left_checks = [row for row in checks if row.bit_length() <= half]
    coupled = [row for row in checks if row.bit_length() > half]
    left = directory(half, left_checks, cutoff, method, stats)
    offsets, columns = right_data(half, coupled)
    outputs = []
    for a in left:
        outputs.extend(extend(half, a, offsets, columns, cutoff, method, stats))
    return outputs


def evaluate(case, method, limit=100_000_000):
    stats = Counter({name: 0 for name in (
        "direct_candidates_equivalent", "extensions", "outside_vectors",
        "factorizations", "quotient_tests", "consistent_outside",
        "emitted_at_all_levels")})
    stats["limit"] = limit
    if case["kind"] == "directory":
        words = directory(case["n"], case["checks"], case["cutoff"], method, stats)
    else:
        half = case["n"]//2
        coupled = [row for row in case["checks"] if row.bit_length() > half]
        offsets, columns = right_data(half, coupled)
        words = extend(half, case["left_input"], offsets, columns,
                       case["cutoff"], method, stats)
    return words, {k: v for k, v in stats.items() if k != "limit"}


def checksum(words, n):
    digest = hashlib.sha256()
    for word in sorted(words):
        digest.update(word.to_bytes((n+7)//8, "little"))
    return digest.hexdigest()
