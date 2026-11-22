"""Algorithm 4. Reliability-prioritized information sets and expression design.

Run: python -m algorithms.algorithm4_profiles
The small reference implementation accepts a finite family and a prescribed
fixed-profile solver, optional design observations, and known exclusions.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite

from polar_subcodes.algebra import polar_transform
from polar_subcodes.code import Design, PolarSubcode
from .algorithm1_directory import list_inputs
from .algorithm3_fixed_profile import causal_pairs, construct


def target_multiplicity(code, target):
    """Return A_target, or None when the candidate misses the distance target."""
    count = 0
    for u in list_inputs(code, target):
        if not u:
            continue
        if polar_transform(u, code.n).bit_count() < target:
            return None
        count += 1
    return count


def refine(code, target, cutoff, path_loss=None):
    """Return the verified spectral and protected-prefix endpoints (§V.D).

    path_loss(code) measures whole-frame transmitted-path loss on fixed design
    observations. The large Proposed construction in §VI.C uses a different,
    position-local score; its construction recipe is shipped separately.
    """
    if cutoff < target:
        raise ValueError("The protection cutoff must include the target weight.")
    threshold = min((i for i in code.profile if 1 << i.bit_count() <= cutoff),
                    default=code.n)
    edges = causal_pairs(Design(code.n, code.profile))
    prefix = [(j, i) for j, i in edges if j < threshold]
    spectral = [edge for edge in edges if edge not in prefix]

    def descend(current, candidates, objective):
        best = objective(current)
        while True:
            for j, i in candidates:
                columns = list(current.columns)
                columns[j] ^= 1 << i
                trial = PolarSubcode(code.n, code.profile, tuple(columns))
                score = objective(trial)
                if score is not None and score < best:
                    current, best = trial, score
                    break
            else:
                return current

    after_spectrum = descend(code, spectral, lambda c: target_multiplicity(c, target))
    # Proposition 1 guarantees the same protected words on every allowed edge.
    after_prefix = descend(after_spectrum, prefix, path_loss) if path_loss else after_spectrum
    return after_spectrum, after_prefix


def frame_loss_measure(llrs, list_size):
    """Construct the §V.D whole-frame rule for LLRs from all-zero transmissions.

    Reuse precisely the same frames for competing expressions. The count of
    frames in which the transmitted path disappears orders candidates like its
    empirical fraction because the denominator is fixed.
    """
    import numpy as np
    from polar_subcodes.decoding.full import decode_full_path_batch
    from polar_subcodes.decoding.scl import parameters

    llrs = np.asarray(llrs, dtype=float)
    if llrs.ndim != 2 or not len(llrs) or not np.isfinite(llrs).all() or list_size < 1:
        raise ValueError("Supply finite nonempty design observations and a positive list size.")

    def measure(code):
        if llrs.shape[1] != code.n:
            raise ValueError("The observations must have N columns.")
        if code.n == 1:
            return int((llrs[:, 0] < 0).sum()) if code.k and list_size == 1 else 0
        actions = [0 if j in code.profile else 2 if code.columns[j] else 1
                   for j in range(code.n)]
        _, first_loss, _ = decode_full_path_batch(
            parameters(code.n), actions, code.supports, llrs, list_size,
            early_stop=False, backend="numpy", max_support_size=None)
        return int((first_loss >= 0).sum())
    return measure


@dataclass
class SearchResult:
    candidates: dict  # verified profile -> (construction, spectral, protected)
    selected: list    # at most Q distinct profiles
    outcomes: dict    # constructed profiles, including unresolved returns


def design_profiles(n, k, target, order, costs, family, count, *, cutoff=None,
                    excluded=lambda profile: False, solver=construct, path_loss=None):
    """Process the connected proposals reached from the first admissible seed."""
    Design(n, ())
    order, costs = tuple(order), tuple(costs)
    if (set(order) != set(range(n)) or len(order) != n or len(costs) != n or
        any(not isfinite(c) or c < 0 for c in costs) or
        not 0 <= k <= n or count < 0 or target < 1):
        raise ValueError("Invalid reliability order, costs, dimensions or requested count.")
    cutoff = target if cutoff is None else cutoff
    family = {Design(n, tuple(p)).profile for p in family}
    if any(len(p) != k for p in family):
        raise ValueError("Each member of the family must have dimension K.")
    rank = {i: r for r, i in enumerate(order)}

    def rank_tuple(profile):
        return tuple(sorted(rank[i] for i in profile))

    def cost(profile):
        return sum(costs[i] for i in profile)

    proofs = set()
    def excluded_now(profile):
        # Only coefficient-independent full-suffix proofs may remove another profile.
        return excluded(profile) or any(
            tuple(i - n + h for i in profile if i >= n - h) == local
            for h, local in proofs)

    seed = next((p for p in sorted(family, key=rank_tuple) if not excluded_now(p)), None)
    queue = deque([] if seed is None else [seed])
    proposed = set(queue)
    candidates, outcomes = {}, {}
    while queue:
        current = queue.popleft()
        if excluded_now(current):
            continue
        outcome = solver(n, current, target)
        outcomes[current] = outcome
        if outcome.status == "verified":
            endpoints = refine(outcome.code, target, cutoff, path_loss)
            candidates[current] = (outcome.code, *endpoints)
        for suffix in outcome.suffixes:
            if suffix.status == "infeasible":
                proofs.add((suffix.length, suffix.profile))

        unresolved = [s.length for s in outcome.suffixes if s.status == "unresolved"]
        nu = min(unresolved, default=0)
        mu = max((s.length for s in outcome.suffixes if s.status == "verified"
                  and s.length < nu), default=0)
        neighbours = []
        for a in current:
            for b in range(n):
                if b in current:
                    continue
                other = tuple(sorted((set(current) - {a}) | {b}))
                if other not in family or other in proposed or excluded_now(other):
                    continue
                touches_ring = any(n - nu <= i < n - mu for i in (a, b))
                preserves_shorter = max(a, b) < n - mu
                if nu and touches_ring and preserves_shorter:
                    group = 0
                elif costs[b] < costs[a]:
                    group = 1
                else:
                    group = 2
                neighbours.append((group, cost(other), rank_tuple(other), other))
        for _, _, _, other in sorted(neighbours):
            queue.append(other)
            proposed.add(other)

    # Round-robin over dyadic-suffix classes. Among ties use the reliability
    # order of the profiles. This is a diversity rule, not an exclusion test.
    classes = defaultdict(deque)
    for profile in sorted(candidates, key=lambda p: (cost(p), rank_tuple(p))):
        key = tuple(sum(i >= n - (1 << b) for i in profile) for b in range(n.bit_length()-1))
        classes[key].append(profile)
    selected = []
    while len(selected) < min(count, len(candidates)):
        for group in classes.values():
            if group and len(selected) < count:
                selected.append(group.popleft())
    return SearchResult(candidates, selected, outcomes)


if __name__ == "__main__":
    from itertools import combinations
    example = design_profiles(4, 2, 2, (3, 2, 1, 0), (3, 2, 1, 0),
                              combinations(range(4), 2), 3)
    print("Selected profiles:", example.selected)
