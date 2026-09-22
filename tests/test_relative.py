from collections import Counter
from itertools import combinations
import random
import unittest

from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum, small_error_counts, causal_containment_domain
from polar_subcodes.algebra import polar_transform


def all_words(profile, columns):
    result = []
    for message in range(1 << len(profile)):
        u = sum(((message >> q) & 1) << i for q, i in enumerate(profile))
        for j, support in enumerate(columns):
            if (u & support).bit_count() & 1:
                u |= 1 << j
        result.append((u, polar_transform(u, len(columns)).bit_count()))
    return result


class RelativeCosetTests(unittest.TestCase):
    def test_small_error_counts_with_repeated_and_zero_columns(self):
        for columns in ([0, 0, 1, 1, 2], [1, 2, 3], []):
            for syndrome in range(4):
                expected = [sum(
                    syndrome == self.xor(columns[i] for i in positions)
                    for positions in combinations(range(len(columns)), size))
                    for size in range(4)]
                self.assertEqual(small_error_counts(columns, syndrome, 3), expected)

    def test_radius_three_relative_spectrum_against_all_messages(self):
        for profile in ((1, 3, 5, 7), (0, 3, 5, 7)):
            edges = [(j, i) for j in range(8) if j not in profile for i in profile if i < j]
            for mask in range(1 << len(edges)):
                columns = [0]*8
                for q, (j, i) in enumerate(edges):
                    columns[j] |= ((mask >> q) & 1) << i
                model = RelativePlotkinSpectrum(profile, columns)
                left = [(u, w) for u, w in all_words(model.left_profile, model.left_columns) if w]
                right = Counter(w for _, w in all_words(model.right_profile, model.right_columns))
                expected = Counter(w for _, w in all_words(profile, columns))
                self.assertEqual(model.spectrum(left, right, 8)['spectrum'], expected)

    def test_radius_four_counts_against_distinct_coordinate_subsets(self):
        rng = random.Random(240917)
        for n in range(13):
            for _ in range(5):
                columns = [rng.randrange(8) for _ in range(n)]
                for syndrome in range(8):
                    expected = [sum(
                        syndrome == self.xor(columns[i] for i in positions)
                        for positions in combinations(range(n), size))
                        for size in range(5)]
                    self.assertEqual(small_error_counts(columns, syndrome, 4), expected)

    def test_radius_four_relative_spectrum_against_all_messages(self):
        rng = random.Random(2416)
        profile = (1, 3, 7, 9, 11, 15)
        for _ in range(20):
            columns = [sum(rng.randrange(2) << i for i in profile if i < j)
                       if j not in profile else 0 for j in range(16)]
            model = RelativePlotkinSpectrum(profile, columns)
            left = [(u, w) for u, w in all_words(model.left_profile, model.left_columns)
                    if 0 < w <= 10]
            right = Counter(w for _, w in all_words(model.right_profile, model.right_columns)
                            if 2*w <= 10)
            expected = Counter(w for _, w in all_words(profile, columns) if w <= 10)
            self.assertEqual(model.spectrum(left, right, 10)['spectrum'], expected)

    @staticmethod
    def xor(values):
        result = 0
        for value in values:
            result ^= value
        return result

    def test_all_causal_n8_supports(self):
        n, profile, cutoff = 8, (1, 3, 5, 7), 4
        edges = [(j, i) for j in range(n) if j not in profile for i in profile if i < j]
        for mask in range(1 << len(edges)):
            columns = [0]*n
            for q, (j, i) in enumerate(edges):
                columns[j] |= ((mask >> q) & 1) << i
            model = RelativePlotkinSpectrum(profile, columns)
            left = [(u, w) for u, w in all_words(model.left_profile, model.left_columns)
                    if 0 < w <= cutoff]
            right = Counter(w for _, w in all_words(model.right_profile, model.right_columns)
                            if 2*w <= cutoff)
            expected = Counter(w for _, w in all_words(profile, columns) if w <= cutoff)
            self.assertEqual(model.spectrum(left, right, cutoff)['spectrum'], expected)

    def test_bad_left_input_rejected(self):
        model = RelativePlotkinSpectrum((1, 3, 5, 7), [0]*8)
        with self.assertRaises(ValueError):
            model.inspect(1 << 2, 4)

    def test_containment_domain_counts_every_legal_support(self):
        profile, mandatory = (1, 3, 5, 7), [2, 8]
        domain = causal_containment_domain(profile, mandatory, 8)
        self.assertTrue(domain['consistent'])
        edges = [(j, i) for j in range(8) if j not in profile for i in profile if i < j]
        count = 0
        for mask in range(1 << len(edges)):
            columns = [0]*8
            for q, (j, i) in enumerate(edges):
                columns[j] |= ((mask >> q) & 1) << i
            count += all(not (u & ((1 << j) ^ columns[j])).bit_count() & 1
                         for u in mandatory for j in range(8) if j not in profile)
        self.assertEqual(count, 1 << domain['dimension'])
        self.assertFalse(causal_containment_domain(profile, [1 << 2], 8)['consistent'])


if __name__ == '__main__':
    unittest.main()
