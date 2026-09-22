from collections import Counter
import json
from pathlib import Path
import unittest

from polar_subcodes.theory.retained_core import extension_bound, first_order_profile_family, grouped_left_spectrum
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.algebra import all_words
from polar_subcodes.algebra import polar_transform


class RetainedCoreExtensionTests(unittest.TestCase):
    def test_shortening_rank_on_retained_words(self):
        for h, profile in ((4, (1, 3)), (8, (3, 5, 6, 7)), (8, (3, 7))):
            words = all_words(profile, [0]*h)
            code = [polar_transform(u, h) for u, _ in words]
            delta = min(w for _, w in words if w)
            model = RelativePlotkinSpectrum(profile+tuple(h+i for i in profile), [0]*(2*h))
            for u, weight in words:
                if 0 < weight < 2*delta:
                    a = polar_transform(u, h)
                    shortened = [x for x in code if not x & ~a]
                    self.assertEqual(set(shortened), {0, a})
                    item = model.constraint(u, 2*delta-1, materialize=False, max_radius=6)
                    self.assertEqual(item['kernel_dimension'], 1)
                    self.assertEqual(item['rank'], h-len(profile)-weight+1)

    def test_small_complete_outer_counts(self):
        for h, left, right, target in ((4, (1,), (1, 3), 3),
                                       (4, (1, 3), (1, 3), 3),
                                       (8, (3,), (3, 7), 6),
                                       (8, (3,), (3, 5, 6, 7), 6)):
            delta = min(w for _, w in all_words(right, [0]*h) if w)
            groups = grouped_left_spectrum(left, all_words(left, [0]*h))
            bound = extension_bound(h, len(right), delta, target, groups)
            rows = [j for j in range(h) if j not in right]
            edges = [(j, i) for j in rows for i in left]
            profile = left+tuple(h+i for i in right)
            successful = 0
            for value in range(1 << len(edges)):
                columns = [0]*(2*h)
                for q, (j, i) in enumerate(edges):
                    columns[h+j] |= ((value >> q) & 1) << i
                successful += all(not w or w >= target for _, w in all_words(profile, columns))
            if bound['guaranteed']:
                self.assertGreater(successful, 0)
                self.assertGreaterEqual(successful, bound['completion_count_lower_bound'])
            if h == 8 and len(left) == 1:
                self.assertEqual(successful, bound['completion_count_lower_bound'])

    def test_closed_form_family_histograms(self):
        for exponent in (4, 5):
            family = first_order_profile_family(exponent)
            profile = family['constituent_profile']
            h = 1 << exponent
            groups = grouped_left_spectrum(profile, all_words(profile, [0]*h))
            length = h//2
            self.assertEqual([groups[i][length] for i in profile], family['critical_group_sizes'])
            expected = {0: 1, length: 2*h-2, h: 1}
            self.assertEqual(dict(Counter(w for _, w in all_words(profile, [0]*h))), expected)
            bound = extension_bound(h, len(profile), length, family['guaranteed_distance'], groups)
            self.assertTrue(bound['guaranteed'])
            self.assertEqual(bound['maximum_group_mean_upper_bound'], family['maximum_group_mean_upper_bound'])
            next_bound = extension_bound(h, len(profile), length, family['guaranteed_distance']+2, groups)
            self.assertFalse(next_bound['guaranteed'])

    def test_uniform_bound_over_nonzero_retained_right_supports(self):
        h, left, right, target = 8, (3,), (3, 5), 8
        groups = grouped_left_spectrum(left, all_words(left, [0]*h))
        bound = extension_bound(h, len(right), 4, target, groups)
        self.assertTrue(bound['guaranteed'])
        self.assertEqual(bound['completion_count_lower_bound'], 24)
        rows = tuple(j for j in range(h) if j not in right)
        full_profile = left+tuple(h+i for i in right)
        for right_bits in range(4):
            local = [0]*h
            for bit, j in enumerate((6, 7)):
                local[j] = ((right_bits >> bit) & 1) << 5
            right_words = all_words(right, local)
            self.assertGreaterEqual(min(w for _, w in right_words if w), 4)
            physical_words = {polar_transform(u, h) for u, _ in right_words}
            self.assertIn(polar_transform(1 << 3, h), physical_words)
            successful = 0
            for outer in range(1 << len(rows)):
                columns = [0]*h+[c << h for c in local]
                for bit, j in enumerate(rows):
                    columns[h+j] |= ((outer >> bit) & 1) << 3
                successful += all(not w or w >= target
                                  for _, w in all_words(full_profile, columns))
            self.assertGreaterEqual(successful, bound['completion_count_lower_bound'])


    def test_inconclusive_is_not_an_impossibility_claim(self):
        groups = {3: {4: 1}, 5: {4: 2}}
        bound = extension_bound(8, 4, 4, 6, groups)
        self.assertFalse(bound['guaranteed'])
        self.assertIn('inconclusive', bound['failure_means'])


if __name__ == '__main__':
    unittest.main()
