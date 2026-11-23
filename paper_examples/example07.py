"""Example 7. Exhaustive N=32 counts, bounds, and both Fig. 3 traces.

Every one of the 2^18 common-supercode inputs contributes to the direct
spectrum check. The quotient is computed independently from the check matrix.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product

from polar_subcodes.algebra import polar_transform, rref
from polar_subcodes.code import PolarSubcode
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum

LEFT = (3, 5)
RIGHT = (3, 5, 9, 10, 11, 12, 13, 14, 15)
FROZEN = tuple(j for j in range(16) if j not in RIGHT)
PROFILE = LEFT + tuple(16 + i for i in RIGHT)


def all_labels(columns):
    """Entry m is the XOR of the columns selected by the little-endian mask m."""
    labels = [0]
    for column in columns:
        labels += [x ^ column for x in labels]
    return labels


def trace(groups, checks):
    """Choose the bit giving the smaller conditional mean in the current group."""
    final = [0, 0]
    traces = []
    for source, positions in enumerate(groups):
        alternatives = all_labels([1 << pos for pos in positions])
        penalties = {}
        for value in alternatives:
            if source == 0:
                penalties[value] = int(checks[0][value] == 0)
            else:
                penalties[value] = int(checks[1][value] == 0) + int(checks[2][final[0] ^ value] == 0)
        trajectory = [str(Fraction(sum(penalties.values()), len(alternatives)))]
        for pos in positions:
            parts = [[v for v in alternatives if (v >> pos) & 1 == bit] for bit in (0, 1)]
            means = [Fraction(sum(penalties[v] for v in part), len(part)) for part in parts]
            selected = int(means[1] < means[0])  # The paper breaks ties in favor of zero.
            alternatives = parts[selected]
            trajectory.append(str(means[selected]))
        final[source], = alternatives
        assert penalties[final[source]] == 0
        traces.append(trajectory)
    return final, traces


def run():
    model = RelativePlotkinSpectrum(PROFILE, [0] * 32)
    left_inputs = (1 << 3, 1 << 5, (1 << 3) ^ (1 << 5))
    left_words = [polar_transform(u, 16) for u in left_inputs]
    events = [model.constraint(u, 7) for u in left_inputs]
    assert all(e["rank"] == 4 and e["kernel_dimension"] == 1 for e in events)
    assert all(len(e["forbidden"]) == 8 for e in events)
    projections = [all_labels(e["syndrome_map"]) for e in events]

    # Use all right inputs, sorted by their actual seven-bit check residual.
    # For each left word a, the full codeword is (a+x_R, x_R).
    right_words = all_labels([polar_transform(1 << i, 16) for i in range(16)])
    cosets = [[Counter() for _ in range(128)] for _ in range(4)]
    for u, x in enumerate(right_words):
        syndrome = sum(((u >> j) & 1) << bit for bit, j in enumerate(FROZEN))
        for index, a in enumerate([0, *left_words]):
            weight = (a ^ x).bit_count() + x.bit_count()
            if 0 < weight <= 7:
                cosets[index][syndrome][weight] += 1
    assert not cosets[0][0]
    # The smallest nonzero zero-left output is twice the right minimum.
    assert min(right_words[u].bit_count() for u in range(1, 1 << 16)
               if all(not (u >> j) & 1 for j in FROZEN)) == 4

    observed = {}
    sizes = Counter()
    feasible = Counter()
    spectra = {}
    for first, second in product(range(128), repeat=2):
        # T has two seven-bit source columns. The third nonzero message selects
        # both columns, so the three residuals depend on the same two choices.
        labels = (first, second, first ^ second)
        obs = tuple(projection[value] for projection, value in zip(projections, labels))
        hist = Counter()
        for block, residual in enumerate(labels, start=1):
            hist.update(cosets[block][residual])
        signature = tuple(hist[w] for w in range(1, 8))
        assert observed.setdefault(obs, signature) == signature
        sizes[obs] += 1
        spectra[first, second] = signature
        for target in (6, 8):
            # At D=5 only projected residual zero is forbidden. At D=7
            # single-bit outside assignments supply the other seven labels.
            bad = [{0}] * 3 if target == 6 else [e["forbidden"] for e in events]
            allowed = all(y not in B for y, B in zip(obs, bad))
            assert allowed == (sum(signature[:target - 1]) == 0)
            feasible[target] += allowed
    assert len(sizes) == 512 and set(sizes.values()) == {32}
    assert feasible == {6: 13472, 8: 1536}

    # A bit of T contributes one column of L_D. Stack its three message blocks,
    # then use reverse-source basis selection to retain independent columns,
    # starting with the later source. Rank 9 leaves 2^(14-9) lifts per class.
    columns = []
    for source in range(2):
        for j in range(7):
            blocks = [projections[i][1 << j] if (i + 1) & (1 << source) else 0
                      for i in range(3)]
            columns.append(sum(block << (4 * i) for i, block in enumerate(blocks)))
    first_two_block_rank = len(rref(c & 255 for c in columns))
    assert first_two_block_rank == 8 and len(rref(columns)) == 9
    basis = []
    groups = [[], []]
    for pos in reversed(range(14)):
        trial = rref([*basis, columns[pos]])
        if len(trial) > len(basis):
            basis = trial
            groups[pos // 7].append(pos % 7)
    groups = [sorted(g) for g in groups]
    assert list(map(len, groups)) == [4, 5]
    choices = [all_labels([1 << j for j in group]) for group in groups]
    child_counts = []
    for first in choices[0]:
        if projections[0][first]:
            child_counts.append(sum(not any(spectra[first, second][:5]) for second in choices[1]))
    assert Counter(child_counts) == {28: 14, 29: 1}
    phi = sum(Fraction(1, 2**e["rank"]) for e in events)
    # At target 6 each message forbids only the zero observation. The joint
    # lower bound counts violations with multiplicity. The source-wise bound
    # uses the minimum number of extensions over all accepted first columns.
    joint_bound = (1 << len(basis)) * (1 - phi)
    product_bound = len(child_counts) * min(child_counts)
    assert (joint_bound, product_bound, sum(child_counts)) == (416, 420, 421)
    full, full_trace = trace([range(7), range(7)], projections)
    reduced, reduced_trace = trace(groups, projections)
    assert full == reduced
    columns = [0] * 32
    for source, column in zip(LEFT, full):
        for j, destination in enumerate(FROZEN):
            if (column >> j) & 1:
                columns[16 + destination] |= 1 << source
    code = PolarSubcode(32, PROFILE, tuple(columns))
    completed = Counter(code.encode(m).bit_count() for m in range(1, 1 << code.k))
    assert code.supports[18] == (5,) and code.supports[20] == (3, 5)
    assert min(completed) == 6 and completed[6] == 12
    return {"joint_ranks": [first_two_block_rank, len(basis)], "kernel_dimension": 14 - len(basis),
            "classes": len(observed), "lifts_per_class": 32, "counts_full": dict(feasible),
            "counts_reduced": {k: v // 32 for k, v in feasible.items()},
            "bounds_and_exact_at_target_6": [int(joint_bound), product_bound, sum(child_counts)],
            "children_count_histogram": dict(Counter(child_counts)), "full_trace": full_trace,
            "reduced_trace": reduced_trace, "dmin": min(completed), "A6": completed[6]}


if __name__ == "__main__":
    print(run())
