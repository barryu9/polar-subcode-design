from itertools import combinations
from math import comb, gcd
import unittest

from polar_subcodes.construction.conditional import xor_mask
from polar_subcodes.theory.structured_existence import (canonical_moore_code, causal_check_domain,
    echelon, first_irreducible, gf_power, is_irreducible, moore_layer,
    moore_value, positive_rate_bound)
from polar_subcodes.algebra import all_words
from polar_subcodes.algebra import polar_transform


class StructuredDomainExistenceTests(unittest.TestCase):
    def test_fields_and_moore_independence(self):
        for degree in (4, 5, 6, 7, 9):
            polynomial = first_irreducible(degree)
            self.assertTrue(is_irreducible(polynomial, degree))
            for value in range(1, 1 << degree):
                self.assertEqual(gf_power(value, (1 << degree)-1, degree, polynomial), 1)
        self.assertFalse(is_irreducible(0b10101, 4))
        for vectors in combinations(range(1, 32), 3):
            value = moore_value(vectors, 5, first_irreducible(5))
            self.assertEqual(bool(value), len(echelon(vectors)) == 3)

    def test_moore_check_on_all_small_affine_minimum_words(self):
        for degree in (4, 5):
            model = moore_layer(degree, 2)
            n = 1 << degree
            for a, b in combinations(range(1, n), 2):
                expected = moore_value((a, b), degree, model['polynomial'])
                for offset in range(4):
                    word = sum(1 << x for x in range(n)
                               if ((a & x).bit_count() & 1) == (offset & 1)
                               and ((b & x).bit_count() & 1) == ((offset >> 1) & 1))
                    self.assertEqual(word.bit_count(), n//4)
                    u = polar_transform(word, n)
                    self.assertEqual(xor_mask(model['check_columns'], u), expected)
                    self.assertNotEqual(expected, 0)

    def test_causal_criterion_and_count_against_all_small_supports(self):
        n = 4
        for mask in range(1, 16):
            profile = tuple(i for i in range(n) if mask >> i & 1)
            destinations = tuple(j for j in range(n) if j not in profile)
            full_edges = tuple((j, i) for i in profile for j in destinations if i < j)
            for edges in (full_edges, full_edges[::2]):
                for check in range(32):
                    checks = tuple(((check >> i) & 1) |
                                   ((((check >> ((i+1) % n)) & 1) << 1) if check >= 16 else 0)
                                   for i in range(n))
                    result = causal_check_domain(n, profile, destinations, edges, checks)
                    successes = 0
                    for support in range(1 << len(edges)):
                        valid = True
                        for i in profile:
                            value = checks[i]
                            for q, (j, source) in enumerate(edges):
                                if source == i and support >> q & 1:
                                    value ^= checks[j]
                            valid &= not value
                        successes += valid
                    self.assertEqual(result['count'], successes)
                    self.assertEqual(result['compatible'], bool(successes))
                    if successes:
                        for i in profile:
                            u = (1 << i) | sum(1 << j for j, column in enumerate(result['columns'])
                                              if column >> i & 1)
                            self.assertEqual(xor_mask(checks, u), 0)

    def test_canonical_kernel_profile_and_exact_small_distances(self):
        for degree, order, expected_k, expected_d in ((4, 2, 7, 6), (5, 2, 11, 12)):
            code = canonical_moore_code(degree, order)
            self.assertEqual(code['k'], expected_k)
            self.assertEqual(code['count'], 1)
            self.assertEqual(min(w for _, w in all_words(code['profile'], code['columns']) if w), expected_d)
            core = moore_layer(degree, order)['core']
            self.assertTrue(all(not (column & sum(1 << i for i in core)) for column in code['columns']))

    def test_rank_and_retained_core_without_spectrum_search(self):
        for degree, order in ((5, 3), (6, 3), (7, 4), (9, 5)):
            code = canonical_moore_code(degree, order)
            model = moore_layer(degree, order)
            if gcd(degree, order) == 1:
                self.assertEqual(code['auxiliary_rank'], degree)
            self.assertEqual(code['k'], sum(comb(degree, j) for j in range(order+1))-model['check_rank'])
            for i in code['profile']:
                u = (1 << i) | sum(1 << j for j, column in enumerate(code['columns']) if column >> i & 1)
                self.assertEqual(xor_mask(model['check_columns'], u), 0)

    def test_positive_rate_full_code_family(self):
        expected = ((4, 1024, 629, 48), (5, 4096, 2499, 96), (6, 16384, 9895, 192))
        for t, n, k, distance in expected:
            result = positive_rate_bound(t)
            self.assertEqual((result['n'], result['k'], result['guaranteed_distance']), (n, k, distance))
            self.assertTrue(result['sufficient'])
            self.assertLess(result['total_column_bound'], 1 << result['coarse_log_bound'])
            self.assertLess(result['coarse_log_bound'], result['coarse_redundancy_lower_bound'])
            self.assertGreater(k/n, 0.5)
        for t in (2, 3):
            self.assertFalse(positive_rate_bound(t)['sufficient'])
            self.assertIsNone(positive_rate_bound(t)['guaranteed_distance'])
        for t in range(4, 31):
            bound = t*t+t-1+(1 << (t-1))*(2*t+5)
            next_bound = (t+1)**2+(t+1)-1+(1 << t)*(2*t+7)
            self.assertLess(next_bound, 4*bound)
            self.assertLess(bound, 1 << (2*t-1))
            self.assertLessEqual(comb(2*t+1, t+1), 1 << (2*t-1))

    def test_invalid_coordinate_domains_are_rejected(self):
        with self.assertRaises(ValueError):
            causal_check_domain(4, (1,), (0,), ((0, 1),), (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            moore_layer(4, 3)


if __name__ == '__main__':
    unittest.main()
