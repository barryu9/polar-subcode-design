from collections import Counter
from fractions import Fraction
from itertools import combinations, product
import unittest

from polar_subcodes.construction.completion import SourceCompletion, prepare_exclusion, small_error_labels
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.algebra import all_words
from polar_subcodes.construction.conditional import xor_mask


def xor(values):
    result = 0
    for value in values:
        result ^= value
    return result


class RadiusThreeCompletionTests(unittest.TestCase):
    def test_labels_include_exactly_distinct_coordinate_error_supports(self):
        for columns in ((), (0, 0, 1, 1, 2), (1, 2, 3, 4, 7), (5,)):
            for radius in range(4):
                expected = {xor(columns[i] for i in positions)
                            for size in range(radius+1)
                            for positions in combinations(range(len(columns)), size)}
                self.assertEqual(set(map(int, small_error_labels(columns, radius))), expected)

    def test_materialized_constraints_match_projected_spectrum_through_radius_three(self):
        profile = (1, 3, 5, 7)
        for c4, c6 in product(range(4), repeat=2):
            columns = [0]*8
            for q, source in enumerate((1, 3)):
                columns[4] |= ((c4 >> q) & 1) << source
                columns[6] |= ((c6 >> q) & 1) << source
            model = RelativePlotkinSpectrum(profile, columns)
            for u, weight in all_words(model.left_profile, model.left_columns):
                if not weight:
                    continue
                for cutoff in (2, 4, 6, 8):
                    if weight > cutoff:
                        continue
                    item = model.constraint(u, cutoff)
                    cross = sum(((columns[j] & u).bit_count() & 1) << q
                                for q, j in enumerate((4, 6)))
                    label = xor_mask(item['syndrome_map'], cross)
                    self.assertEqual(label not in item['forbidden'], model.inspect(u, cutoff)['separated'])

    def test_bit_frontier_enumerates_all_and_only_solutions(self):
        profile, columns = (1, 3, 5, 7), [0]*8
        model = RelativePlotkinSpectrum(profile, columns)
        info_mask = (1 << 1) | (1 << 3)
        left = [(u, w) for u, w in all_words((1, 3), [0]*4) if w]
        for source, cutoff in product((1, 3), (2, 4, 6, 8)):
            prepared = [prepare_exclusion(model, u, cutoff, info_mask)
                        for u, w in left if (u & info_mask).bit_length()-1 == source and w <= cutoff]
            for previous in ({}, {1: 0}, {1: 1}, {1: 2}, {1: 3}):
                if source == 1 and previous:
                    continue
                expected = []
                for value in range(4):
                    good = True
                    for item in prepared:
                        shift = value ^ xor(v for i, v in previous.items() if (item.message >> i) & 1)
                        good &= xor_mask(item.coordinates, shift) not in set(map(int, item.labels))
                    if good:
                        expected.append(value)
                solver = SourceCompletion(prepared, previous, 2)
                found = []
                for _ in range(30):
                    answer = solver.step()
                    if answer['status'] == 'sat':
                        found.append(answer['value'])
                    if answer['status'] == 'unsat':
                        break
                    solver = SourceCompletion(prepared, previous, 2, solver.snapshot())
                self.assertEqual(sorted(found), expected)
                self.assertEqual(len(found), len(set(found)))

    def test_empty_constraints_keep_all_future_repair_options(self):
        solver = SourceCompletion([], {}, 3)
        values = []
        for _ in range(20):
            answer = solver.step()
            if answer['status'] == 'sat':
                values.append(answer['value'])
            if answer['status'] == 'unsat':
                break
        self.assertEqual(sorted(values), list(range(8)))
        self.assertEqual(solver.expectation(), Fraction(0))

    def test_conditional_potential_equals_exact_average_bad_events(self):
        profile, columns = (1, 3, 5, 7), [0]*8
        model = RelativePlotkinSpectrum(profile, columns)
        info_mask = (1 << 1) | (1 << 3)
        prepared = [prepare_exclusion(model, u, 4, info_mask)
                    for u, w in all_words((1, 3), [0]*4)
                    if w and (u & info_mask).bit_length()-1 == 3]
        solver = SourceCompletion(prepared, {1: 1}, 2)
        for value in range(4):
            for point in solver.expectation_path(value):
                bits = point['fixed_bits']
                allowed = [v for v in range(4) if ((v ^ value) & ((1 << bits)-1)) == 0]
                total = 0
                successful = 0
                for v in allowed:
                    bad = 0
                    for item in prepared:
                        shift = v ^ int(bool(item.message & (1 << 1)))
                        bad += xor_mask(item.coordinates, shift) in set(map(int, item.labels))
                    total += bad
                    successful += bad == 0
                self.assertEqual(Fraction(point['expectation']), Fraction(total, len(allowed)))
                bound = solver.prefix_completion_bound(bits, value & ((1 << bits)-1))
                self.assertLessEqual(bound['successful_patterns_lower_bound'], successful)


if __name__ == '__main__':
    unittest.main()
