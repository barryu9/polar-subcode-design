"""Algorithm 3. Construct a code on a fixed information set with domain feedback.

Run: python -m algorithms.algorithm3_fixed_profile
The implementation favors a visible proof path for small cases. For production
lengths see the repository's simulation-construction commands.
"""
from dataclasses import dataclass
from itertools import product

from polar_subcodes.algebra import leading
from polar_subcodes.code import Design, PolarSubcode
from .algorithm1_directory import list_inputs
from .algorithm2_outer import affine_mean, complete_outer


@dataclass(frozen=True)
class Suffix:
    length: int
    profile: tuple       # local indices in this suffix
    status: str          # verified, unresolved, infeasible (full suffix space)


@dataclass(frozen=True)
class Outcome:
    status: str
    code: object = None
    suffixes: tuple = ()
    released: tuple = ()


class Deferred(Exception):
    """Raise from the optional work callback to return a pending construction."""


def causal_pairs(design):
    return tuple((j, i) for j in design.rows for i in design.profile if i < j)


def make_code(design, pairs, values):
    columns = [0] * design.n
    for (j, i), bit in zip(pairs, values):
        if bit:
            columns[j] |= 1 << i
    return PolarSubcode(design.n, design.profile, tuple(columns))


def common_conditions(design, free, cutoff, tick):
    """Release the touched checks, then group directory inputs by source pattern.

    A residual is stored at its physical bit position. Leaving zero gaps has no
    effect on equality, affine images, or the forbidden-set count.
    """
    released = {j for j, _ in free}
    source_mask = sum(1 << i for i in {i for _, i in free})
    residual_mask = sum(1 << j for j in released)
    common_profile = tuple(sorted(set(design.profile) | released))
    common = PolarSubcode(design.n, common_profile, (0,) * design.n)
    forbidden = {}
    for u in list_inputs(common, cutoff, tick):
        if u and leading(u) in design.profile:
            forbidden.setdefault(u & source_mask, set()).add(u & residual_mask)
    return [(tuple((1 << j) if q >> i & 1 else 0 for j, i in free), bad)
            for q, bad in forbidden.items()]


def solve_direct(design, free, cutoff, tick):
    checks = common_conditions(design, free, cutoff, tick)

    def evaluate(depth, current):
        tick()
        return affine_mean((base, cols[depth:], bad)
                           for base, (cols, bad) in zip(current, checks))

    def visit(bits, current):
        if len(bits) == len(free):
            return make_code(design, free, bits)
        children = []
        for bit in (0, 1):
            updated = tuple(v ^ (cols[len(bits)] if bit else 0)
                            for v, (cols, _) in zip(current, checks))
            forced, mean = evaluate(len(bits) + 1, updated)
            if not forced:
                children.append((mean, bit, updated))
        for _, bit, updated in sorted(children):
            solution = visit((*bits, bit), updated)
            if solution is not None:
                return solution
        return None

    base = (0,) * len(checks)
    return None if evaluate(0, base)[0] else visit((), base)


def solve_factored(design, free, cutoff, tick):
    h = design.n // 2
    left = Design(h, tuple(i for i in design.profile if i < h))
    right = Design(h, tuple(i - h for i in design.profile if i >= h))
    left_pairs = [(j, i) for j, i in free if j < h]
    right_pairs = [(j - h, i - h) for j, i in free if i >= h]
    outer_domains = [(0, tuple(1 << r for r, j in enumerate(right.rows)
                               if (h + j, i) in free)) for i in left.profile]
    for left_bits in product((0, 1), repeat=len(left_pairs)):
        left_code = make_code(left, left_pairs, left_bits)
        for right_bits in product((0, 1), repeat=len(right_pairs)):
            tick()
            right_code = make_code(right, right_pairs, right_bits)
            # Algorithm 2 also tests the repeated right branch.
            outer = complete_outer(left_code, right_code, cutoff, outer_domains, tick)
            if outer is None:
                continue
            columns = list(left_code.columns) + [value << h for value in right_code.columns]
            for r, j in enumerate(right.rows):
                for i, value in zip(left.profile, outer):
                    columns[h + j] |= ((value >> r) & 1) << i
            return PolarSubcode(design.n, design.profile, tuple(columns))
    return None


def construct(n, profile, target, *, method="direct", suffix_lengths=(), tick=lambda: None):
    """Return an Outcome, including available suffix reports on deferral.

    suffix_lengths selects optional unrestricted dyadic suffix trials in the
    supplied order. At each verified suffix, first try its extension with the
    remaining coefficients zero. Only an exhaustive failure proves infeasibility.
    """
    design = Design(n, tuple(profile))
    if type(target) is not int or target < 1 or method not in ("direct", "factored"):
        raise ValueError("Expected a positive integer target and direct/factored method.")
    suffix_lengths = tuple(suffix_lengths)
    if any(h <= 0 or h >= n or h & (h - 1) for h in suffix_lengths):
        raise ValueError("Each requested suffix must be a proper positive dyadic length.")
    suffixes, released = [], []

    def outcome(status, code=None):
        return Outcome(status, code, tuple(suffixes), tuple(released))

    try:
        tick()
        zero = PolarSubcode(n, design.profile, (0,) * n)
        if not design.profile:
            return outcome("verified", zero)
        if target > n:
            return outcome("infeasible")
        if design.d_P >= target:
            return outcome("verified", zero)

        for h in suffix_lengths:
            local = tuple(i - n + h for i in design.profile if i >= n - h)
            suffix = construct(h, local, (target*h+n-1)//n, method=method, tick=tick)
            suffixes.append(Suffix(h, local, suffix.status))
            if suffix.status != "verified":
                return outcome(suffix.status)
            trial = PolarSubcode(n, design.profile, (0,) * (n-h) +
                                 tuple(c << (n-h) for c in suffix.code.columns))
            if not any(list_inputs(trial, target - 1, tick)):
                return outcome("verified", trial)

        all_pairs = causal_pairs(design)
        free = {(j, i) for j, i in all_pairs if (1 << j.bit_count()) >= design.d_P
                and (1 << i.bit_count()) < target}
        solver = solve_direct if method == "direct" else solve_factored
        while True:
            candidate = solver(design, tuple(sorted(free)), target - 1, tick)
            if candidate is not None:
                if any(list_inputs(candidate, target - 1, tick)):
                    raise ArithmeticError("Solver returned a word below the distance target.")
                return outcome("verified", candidate)
            # Exhaustive failure proves that at least one still-fixed coefficient
            # must change. The full complement is the fallback core in Algorithm 3.
            fixed = set(all_pairs) - free
            if not fixed:
                return outcome("infeasible")
            edge = min(fixed)
            released.append(edge)
            free.add(edge)
    except Deferred:
        return outcome("unresolved")


if __name__ == "__main__":
    for solver in ("direct", "factored"):
        answer = construct(8, (1, 7), 4, method=solver, suffix_lengths=(2, 4))
        print(solver, answer)
