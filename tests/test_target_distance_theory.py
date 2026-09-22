"""Small exact checks of the general target-distance and outer-completion claims.

No experimental N256 profile or checkpoint is used. Projections are built
directly from canonical checks, with arbitrary radius and Python integer labels.
"""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import ceil
import unittest

from polar_subcodes.construction.conditional import xor_mask
from polar_subcodes.algebra import all_words
from polar_subcodes.algebra import polar_transform
from polar_subcodes.algebra import solve


def projected_records(profile, columns):
    half = len(columns) // 2
    left_profile = tuple(i for i in profile if i < half)
    right_profile = tuple(i - half for i in profile if i >= half)
    right_columns = tuple(c >> half for c in columns[half:])
    frozen = tuple(j for j in range(half) if j not in right_profile)
    checks = tuple((1 << j) ^ right_columns[j] for j in frozen)
    physical_columns = [
        sum(((check & polar_transform(1 << t, half)).bit_count() & 1) << q
            for q, check in enumerate(checks))
        for t in range(half)]
    records = []
    for u, weight in all_words(left_profile, columns[:half]):
        if not weight:
            continue
        a = polar_transform(u, half)
        inside = tuple(t for t in range(half) if (a >> t) & 1)
        outside = tuple(t for t in range(half) if not (a >> t) & 1)
        _, quotient = solve([(physical_columns[t], 0) for t in inside], len(frozen))

        def project(value):
            return sum(((value & row).bit_count() & 1) << q
                       for q, row in enumerate(quotient))

        errors = tuple(project(physical_columns[t]) for t in outside)
        distribution = Counter(
            (z.bit_count(), xor_mask(errors, z)) for z in range(1 << len(outside)))
        records.append(dict(
            input=u, weight=weight, quotient=quotient,
            syndrome_map=tuple(project(1 << q) for q in range(len(frozen))),
            kernel_dimension=len(inside) - len(frozen) + len(quotient),
            distribution=distribution, outside=len(outside)))
    return records, frozen, right_profile, right_columns


def cross_label(record, frozen, columns):
    half = len(columns) // 2
    cross = sum(((columns[half+j] & record['input']).bit_count() & 1) << q
                for q, j in enumerate(frozen))
    return xor_mask(record['syndrome_map'], cross)


def spectrum_from_records(records, frozen, right_profile, right_columns, columns):
    result = Counter(2*w for _, w in all_words(right_profile, right_columns))
    for record in records:
        label = cross_label(record, frozen, columns)
        for (radius, syndrome), count in record['distribution'].items():
            if syndrome == label:
                result[record['weight'] + 2*radius] += count << record['kernel_dimension']
    return result


def support_arrays(n, profile, exhaustive=True):
    edges = [(j, i) for j in range(n) if j not in profile for i in profile if i < j]
    masks = range(1 << len(edges)) if exhaustive else (
        (value * 0x9E3779B1) & ((1 << len(edges))-1) for value in range(32))
    for mask in masks:
        columns = [0]*n
        for q, (j, i) in enumerate(edges):
            columns[j] |= ((mask >> q) & 1) << i
        yield columns


class TargetDistanceTheoryTests(unittest.TestCase):
    def test_projection_and_every_cutoff_across_lengths_and_profiles(self):
        cases = [(4, profile, True)
                 for size in range(1, 4) for profile in combinations(range(4), size)]
        cases += [(8, (1, 3, 5, 7), True), (8, (0, 3, 5, 7), True),
                  (16, (7, 11, 13, 15), True), (16, (1, 7, 13, 15), False)]
        large_radius_seen = False
        for n, profile, exhaustive in cases:
            for columns in support_arrays(n, profile, exhaustive):
                records, frozen, right_profile, right_columns = projected_records(profile, columns)
                expected = Counter(w for _, w in all_words(profile, columns))
                self.assertEqual(
                    spectrum_from_records(records, frozen, right_profile, right_columns, columns),
                    expected)
                right_weights = [w for _, w in all_words(right_profile, right_columns) if w]
                for target in range(2, n+1):
                    cutoff = target-1
                    condition = not any(2*w <= cutoff for w in right_weights)
                    for record in records:
                        if record['weight'] > cutoff:
                            continue
                        radius = (cutoff-record['weight'])//2
                        large_radius_seen |= radius > 3
                        forbidden = {label for (r, label) in record['distribution'] if r <= radius}
                        condition &= cross_label(record, frozen, columns) not in forbidden
                    self.assertEqual(condition, not any(0 < w < target for w in expected))
        self.assertTrue(large_radius_seen)

    def test_exact_target_requires_a_retained_word(self):
        n, profile = 8, (1, 3, 5, 7)
        weight_inputs = {}
        for u in range(1, 1 << n):
            weight_inputs.setdefault(polar_transform(u, n).bit_count(), []).append(u)
        strict_lower_bounds = 0
        for columns in support_arrays(n, profile):
            minimum = min(w for _, w in all_words(profile, columns) if w)
            for target in range(2, n+1):
                anchor_exists = any(
                    all(not ((u & ((1 << j) ^ columns[j])).bit_count() & 1)
                        for j in range(n) if j not in profile)
                    for u in weight_inputs.get(target, []))
                self.assertEqual(minimum >= target and anchor_exists, minimum == target)
                strict_lower_bounds += minimum > target
        self.assertGreater(strict_lower_bounds, 0)

    def test_coupled_affine_prefix_means_counts_and_greedy_completion(self):
        n, profile = 8, (1, 3, 5, 7)
        base = [0]*n
        base[2] = 1 << 1
        base[6] = 1 << 5
        edges = ((4, 1), (6, 1), (4, 3), (6, 3))
        # Each direction can couple columns; these three directions are independent.
        directions, origin = (0b0101, 0b1010, 0b1100), 0b0010
        records, frozen, right_profile, right_columns = projected_records(profile, base)
        right_min = min(w for _, w in all_words(right_profile, right_columns) if w)

        def candidate(theta):
            value = origin ^ xor_mask(directions, theta)
            columns = base.copy()
            for q, (j, i) in enumerate(edges):
                columns[j] ^= ((value >> q) & 1) << i
            return columns

        candidates = [candidate(theta) for theta in range(8)]
        self.assertEqual(len(set(map(tuple, candidates))), 8)
        verified_strict_prefixes = 0
        for target in range(2, 9):
            if 2*right_min < target:
                continue
            critical = [record for record in records if record['weight'] < target]

            def events(theta):
                bad = 0
                for record in critical:
                    radius = (target-1-record['weight'])//2
                    forbidden = {s for (r, s) in record['distribution'] if r <= radius}
                    bad += cross_label(record, frozen, candidates[theta]) in forbidden
                return bad

            def prefix_mean(depth, prefix):
                allowed = [t for t in range(8) if t & ((1 << depth)-1) == prefix]
                mean = Fraction(0)
                for record in critical:
                    image = {cross_label(record, frozen, candidates[t]) for t in allowed}
                    radius = (target-1-record['weight'])//2
                    forbidden = {s for (r, s) in record['distribution'] if r <= radius}
                    mean += Fraction(len(forbidden & image), len(image))
                self.assertEqual(mean, Fraction(sum(events(t) for t in allowed), len(allowed)))
                good = sum(events(t) == 0 for t in allowed)
                self.assertLessEqual(ceil(len(allowed)*max(Fraction(0), 1-mean)), good)
                return mean

            for depth in range(4):
                for prefix in range(1 << depth):
                    mean = prefix_mean(depth, prefix)
                    if mean >= 1:
                        continue
                    verified_strict_prefixes += 1
                    chosen = prefix
                    for bit_index in range(depth, 3):
                        means = [prefix_mean(bit_index+1, chosen | (bit << bit_index))
                                 for bit in (0, 1)]
                        bit = int(means[1] < means[0])
                        chosen |= bit << bit_index
                    self.assertEqual(events(chosen), 0)
                    self.assertTrue(all(w == 0 or w >= target
                                        for _, w in all_words(profile, candidates[chosen])))
        self.assertGreater(verified_strict_prefixes, 0)


if __name__ == '__main__':
    unittest.main()
