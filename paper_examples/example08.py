"""Example 8. Complete an N=256 fixed profile and verify the distance witness.

The default walkthrough reads the critical left inputs and three source-group
means from the supplied trace. --recompute-means counts the forbidden values
in the attainable set at each prefix. --verify-spectrum re-enumerates with the C++
backend (build first via python tools/build_native.py).
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from polar_subcodes.algebra import polar_transform, solve
from polar_subcodes.code import PolarSubcode, load_code
from polar_subcodes.construction.conditional import xor_mask
from polar_subcodes.construction.domains import initial_structure
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum

DATA = Path(__file__).with_name("data")


def add_independent_row(pivots, row):
    # A large row of L_D is cheap to XOR as a single Python integer.
    while row:
        index = row.bit_length() - 1
        if index not in pivots:
            pivots[index] = row
            return
        row ^= pivots[index]


def construct():
    """Rebuild the published Example 8 from its RM–GA profile and zero support."""
    from functools import lru_cache
    from itertools import chain, combinations

    import numpy as np
    from paper_examples.initial_domain import complete_initial_domain
    from polar_subcodes.construction.conditional import suffix_coordinates
    from polar_subcodes.construction.profiles import make_profile
    from polar_subcodes.spectrum.backend import words

    base = make_profile(256, 128, "rm_ga", ebn0_db=2)
    removed = {31, 195, 225, 226, 228, 232, 240}
    added = {90, 92, 101, 102, 105, 142, 149}
    profile = tuple(sorted((set(base) - removed) | added))
    local = tuple(i - 128 for i in profile if i >= 128)
    # At target 24, the zero-left branch repeats each right word twice and
    # therefore requires right distance at least 12. Fix this right code and
    # the unprecoded left code before choosing any of the 39 x 39 outer entries.
    right = complete_initial_domain(128, local, 12, "lookahead")
    left = PolarSubcode(128, tuple(i for i in profile if i < 128), (0,) * 128)
    inputs, _ = words(left, 23, max_work=500_000_000)
    full_columns = [0] * 128 + [c << 128 for c in right.columns]
    split = RelativePlotkinSpectrum(profile, full_columns)
    destinations = [128 + j for j in range(128) if j not in local]
    previous = {}

    @lru_cache(maxsize=8)
    def outside_subsets(length, weight):
        indices = chain.from_iterable(combinations(range(length), weight))
        return np.fromiter(indices, dtype=np.int32).reshape(-1, weight)

    for source in left.profile:
        state = []
        for u, _ in inputs:
            # Assign a critical message to the group of its last nonzero
            # source. Once this column is fixed, later columns cannot change
            # that message's observation Q_a T m.
            if u.bit_length() - 1 != source:
                continue
            event = split.constraint(u, 23, materialize=False)
            # Express each projected column in a basis chosen from the right.
            # Fixing source entries left-to-right then settles at most one
            # syndrome bit per step. Each reachable syndrome is equally likely.
            coordinates, pivots, encode = suffix_coordinates(event["syndrome_map"])
            errors = np.array([encode(y) for y in event["error_columns"]], dtype=np.uint32)
            labels = [np.array([0], dtype=np.uint32)]
            for w in range(1, event["radius"] + 1):
                indices = outside_subsets(len(errors), w)
                labels.append(np.bitwise_xor.reduce(errors[indices], axis=1))
            bad = np.unique(np.concatenate(labels))
            shift = 0
            for earlier_source, selected in previous.items():
                if u >> earlier_source & 1:
                    shift ^= selected
            state.append((coordinates, pivots, bad, xor_mask(coordinates, shift), event["rank"]))

        chosen = 0
        for bit in range(len(destinations)):
            # Each total is the conditional mean for the current source group
            # after assigning this coefficient. Count distinct forbidden
            # observations in each child's image and divide by its image size.
            totals = [Fraction(), Fraction()]
            masks = []
            for coords, pivots, bad, offset, rank in state:
                if bit in pivots:
                    parity_bit = pivots[bit]
                    zero = ((bad >> parity_bit) & 1) == ((offset >> parity_bit) & 1)
                    size = np.count_nonzero(zero)
                    totals[0] += Fraction(int(size), 1 << (rank - 1))
                    totals[1] += Fraction(len(bad) - int(size), 1 << (rank - 1))
                    masks.append(zero)
                else:
                    for b in (0, 1):
                        totals[b] += Fraction(len(bad), 1 << rank)
                    masks.append(None)
            value = int(totals[1] < totals[0])
            chosen |= value << bit
            next_state = []
            for (coords, pivots, bad, offset, rank), zero in zip(state, masks):
                if zero is not None:
                    bad = bad[~zero if value else zero]
                    rank -= 1
                if value:
                    offset ^= coords[bit]
                next_state.append((coords, pivots, bad, offset, rank))
            state = next_state
        # For this execution every group reaches zero violations on its first
        # trajectory, including the three groups with an initial mean >= 1.
        assert all(len(bad) == 0 for _, _, bad, _, _ in state)
        previous[source] = chosen
        for k, dest in enumerate(destinations):
            if chosen >> k & 1:
                full_columns[dest] |= 1 << source
    return PolarSubcode(256, profile, tuple(full_columns))


def run(recompute_means=False, verify_spectrum=False, rebuild=False):
    final = load_code(DATA / "example08_code.json")
    if rebuild:
        reconstructed = construct()
        if reconstructed != final:
            raise AssertionError("Reconstructed expressions differ from the archived Example 8 code")
        final = reconstructed
    saved = json.loads((DATA / "example08_trace.json").read_text(encoding="utf-8"))
    left_inputs = [int(v, 16) for v in saved["left_inputs_hex"]]
    model = RelativePlotkinSpectrum(final.profile, final.columns)
    k_left, k_right, f = len(model.left_profile), len(model.right_profile), model.redundancy
    assert (k_left, k_right, f) == (39, 89, 39)
    right = PolarSubcode(128, model.right_profile, model.right_columns)
    setup = initial_structure(128, right.profile, 12)
    mother_profile = tuple(sorted((*right.profile, *setup["rows"])))
    assert len(setup["core"]) == 64 and len(mother_profile) == 96
    assert min(1 << i.bit_count() for i in setup["core"]) == 16
    assert len(set(left_inputs)) == 920 and all(polar_transform(u, 128).bit_count() == 16
                                                for u in left_inputs)
    left_mask = sum(1 << i for i in model.left_profile)
    assert all(u & ~left_mask == 0 for u in left_inputs)
    # Project each critical message and stack its rows on all selected sources.
    # The resulting row-space rank is independent of the quotient basis choice.
    pivots = {}
    right_destinations = [128 + j for j in range(128) if j not in right.profile]
    means = dict.fromkeys((125, 126, 127), Fraction())
    for u in left_inputs:
        last = u.bit_length() - 1
        check_mean = recompute_means and last in means
        event = model.constraint(u, 23, materialize=check_mean)
        columns = event["syndrome_map"]  # Columns of Q_a, one per packed check.
        sources = [k for k, i in enumerate(model.left_profile) if u >> i & 1]
        for bit in range(event["rank"]):
            q_row = sum(((column >> bit) & 1) << j for j, column in enumerate(columns))
            row = sum(q_row << (k * f) for k in sources)
            add_independent_row(pivots, row)

        if check_mean:
            fixed = {125: 4, 126: 5, 127: 6}[last]
            # Earlier selected source columns give an affine offset. The next
            # column has a zero prefix and its remaining entries may vary.
            earlier = u ^ (1 << last)
            rhs = sum(((final.columns[j] & earlier).bit_count() % 2) << k
                      for k, j in enumerate(right_destinations))
            offset = xor_mask(columns, rhs)
            _, dual = solve([(col, 0) for col in columns[fixed:]], event["rank"])
            # Every attainable observation has equal probability. The free
            # image's dual tests membership in the shifted image.
            bad = event["forbidden"]
            if dual:
                num = sum(all(((y ^ offset) & q).bit_count() % 2 == 0 for q in dual) for y in bad)
            else:
                num = len(bad)
            means[last] += Fraction(num, 1 << (event["rank"] - len(dual)))
    assert len(pivots) == 1082
    recorded = {int(src): Fraction(path[-1]) for src, path in saved["means_after_leading_zero_bits"].items()}
    if recompute_means:
        assert means == recorded
    means_used = means if recompute_means else recorded
    # At source 127 the preceding 38 columns and six entries of this column
    # are fixed. Theorem 3 counts completions among the remaining 33 bits.
    guaranteed = 2**(f - 6) * (1 - means_used[127])
    assert guaranteed == 354383872
    source = final.profile.index(142)
    # This right-only message gives the same weight-24 witness for every
    # outer assignment. Excluding weights 1–23 then establishes equality.
    witness_input = final.input_word(1 << source)
    assert witness_input == (1 << 142) | (1 << 147)
    witness_weight = final.encode(1 << source).bit_count()
    assert witness_weight == 24 and final.columns[131] & (1 << 102)
    if verify_spectrum:
        from polar_subcodes.spectrum.backend import count, words
        # Rebuild the complete critical-message input directory and the
        # archived standalone spectrum checks (including the zero-left limit).
        left = PolarSubcode(128, model.left_profile, model.left_columns)
        exact_left, _ = words(left, 23, max_work=500_000_000)
        assert {u for u, _ in exact_left} == set(left_inputs)
        mother = PolarSubcode(128, mother_profile, (0,) * 128)
        assert count(mother, 11, 500_000_000)["spectrum"] == {0: 1, 8: 74288}
        assert count(right, 11, 500_000_000)["spectrum"] == {0: 1}
        # Odd weights are absent since position 0 is frozen. With the right
        # distance verified, check the 920 nonzero-left cosets through D=23.
        assert all(not model.inspect(u, 23)["spectrum"] for u in left_inputs)
    return {"split": [k_left, k_right], "right_core_and_mother_dimensions": [64, 96],
            "weights_to_exclude": [16 + 2 * t for t in range(4)],
            "left_inputs": len(left_inputs), "rank": len(pivots), "kernel_dimension": f * k_left - len(pivots),
            "last_three_group_means": {i: str(v) for i, v in means_used.items()},
            "means_input": "recomputed" if recompute_means else "archived completed run",
            "last_33_bits_guaranteed_completions": int(guaranteed),
            "witness_input_support": [142, 147], "witness_weight": witness_weight,
            "saved_exact_spectrum_through_24": saved["completed_spectrum_through_24"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construct", action="store_true", help="Regenerate every frozen expression from the profile")
    parser.add_argument("--recompute-means", action="store_true")
    parser.add_argument("--verify-spectrum", action="store_true")
    args = parser.parse_args()
    print(run(args.recompute_means, args.verify_spectrum, args.construct))
