"""Algorithm 2. Project the critical messages and complete the outer matrix.

Run: python -m algorithms.algorithm2_outer
The f entries of an outer column are packed in right-frozen-position order.
"""
from dataclasses import dataclass
from fractions import Fraction

from polar_subcodes.algebra import polar_transform, rref
from polar_subcodes.code import PolarSubcode
from polar_subcodes.spectrum.quotient import ImageFactor

from .algorithm1_directory import list_inputs, outside_masks, right_checks, xor_selected


@dataclass
class Condition:
    message: int
    quotient: ImageFactor
    forbidden: set

    def project(self, syndrome):
        # A remainder represents the coset modulo im(P_inside).
        return self.quotient.reduce(syndrome)[0]


def critical_conditions(left, right, cutoff, tick=lambda: None):
    if left.n != right.n:
        raise ValueError("Both constituent codes must have the same length.")
    conditions = []
    columns = right_checks(right)
    for u in list_inputs(left, cutoff, tick):
        if not u:
            continue
        a = polar_transform(u, left.n)
        message = sum(((u >> i) & 1) << q for q, i in enumerate(left.profile))
        inside = [j for j in range(left.n) if a >> j & 1]
        outside = [j for j in range(left.n) if not a >> j & 1]
        factor = ImageFactor([columns[j] for j in inside])
        bad = set()
        for z in outside_masks(outside, (cutoff - len(inside)) // 2):
            tick()
            bad.add(factor.reduce(xor_selected(columns, z))[0])
        conditions.append(Condition(message, factor, bad))
    return conditions


def reverse_source_basis(domains, conditions, width):
    """Choose basis directions of L_D R in reverse source order (Prop. 3)."""
    retained = [[] for _ in domains]
    basis = ()
    for i in reversed(range(len(domains))):
        for direction in domains[i][1]:
            # Pack all message blocks, keeping even the blocks of later groups.
            joint = sum(c.project(direction) << (j * width)
                        for j, c in enumerate(conditions) if c.message >> i & 1)
            enlarged = rref((*basis, joint))
            if len(enlarged) > len(basis):
                retained[i].append(direction)
                basis = enlarged
    return retained


def affine_mean(checks):
    """Return (some condition forced to fail, Phi) for affine observations.

    Each tuple contains a base syndrome, remaining observation directions, and
    forbidden set. Intersection and span sizes suffice; there is no need to
    enumerate the admissible values or assume independence across conditions.
    """
    mean = Fraction()
    for base, directions, forbidden in checks:
        factor = ImageFactor(directions)
        count = sum(factor.reduce(value ^ base)[0] == 0 for value in forbidden)
        size = 1 << factor.rank
        if count == size:
            return True, None
        mean += Fraction(count, size)
    return False, mean


def complete_outer(left, right, cutoff, domains=None, tick=lambda: None):
    """Return a tuple of outer columns, or None after a proved failure.

    domains[i]=(t_i^0, independent columns of B_i). Use None for the full
    space. Returns one representative; Proposition 3 recovers the full fiber.
    """
    if left.n != right.n:
        raise ValueError("Constituents must have equal lengths.")
    f = right.n - right.k
    if domains is None:
        domains = [(0, tuple(1 << j for j in range(f))) for _ in left.profile]
    domains = [(origin, tuple(basis)) for origin, basis in domains]
    if len(domains) != left.k or any(
        len(rref(basis)) != len(basis) or any(v < 0 or v >> f for v in (origin, *basis))
        for origin, basis in domains
    ):
        raise ValueError("Expected one valid independent affine column domain per source.")
    if any(list_inputs(right, cutoff // 2, tick)):
        return None
    conditions = critical_conditions(left, right, cutoff, tick)
    retained = reverse_source_basis(domains, conditions, f)
    groups = [[c for c in conditions if c.message.bit_length() == i + 1]
              for i in range(left.k)]

    def visit(earlier, column, depth):
        tick()
        i = len(earlier)
        checks = []
        for condition in groups[i]:
            offset = xor_selected(earlier, condition.message & ((1 << i) - 1))
            checks.append((condition.project(offset ^ column),
                           [condition.project(b) for b in retained[i][depth:]],
                           condition.forbidden))
        return affine_mean(checks)

    def search(earlier):
        i = len(earlier)
        if i == left.k:
            return earlier

        def assign(column, depth):
            if depth == len(retained[i]):
                return search((*earlier, column))
            children = []
            for bit in (0, 1):
                trial = column ^ (retained[i][depth] if bit else 0)
                forced, mean = visit(earlier, trial, depth + 1)
                if not forced:
                    children.append((mean, bit, trial))
            # Lower conditional mean first; ties go to zero. Both children
            # remain available when the sufficient mean bound is inconclusive.
            for _, _, trial in sorted(children):
                result = assign(trial, depth + 1)
                if result is not None:
                    return result
            return None

        origin = domains[i][0]
        if visit(earlier, origin, 0)[0]:
            return None
        return assign(origin, 0)

    return search(())


if __name__ == "__main__":
    left = PolarSubcode(4, (1,), (0,) * 4)
    right = PolarSubcode(4, (3,), (0,) * 4)
    print("Outer columns for d_star=4:", complete_outer(left, right, 3))
