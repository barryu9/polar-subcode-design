from collections import Counter
from itertools import combinations, product
import unittest

from polar_subcodes.theory.circuits import analyze_circuit, project
from polar_subcodes.theory.interfaces import (coordinate_solver, count_interfaces, echelon,
    intersection, lift_outer_parameter, normalize_event, polar_outer_events, prepare_interfaces)
from polar_subcodes.algebra import all_words
from polar_subcodes.algebra import all_subspaces, span


def exhaustive_count(events, dimension):
    return sum(all((project(x, event['rows']) ^ event.get('offset', 0))
                   not in event['forbidden'] for event in events)
               for x in range(1 << dimension))


class OuterInterfaceTests(unittest.TestCase):
    def test_intersection_for_every_pair_of_small_subspaces(self):
        spaces = all_subspaces(3)
        for a, b in product(spaces, repeat=2):
            self.assertEqual(span(intersection(a, b)), span(a) & span(b))

    def test_all_three_scalar_maps_and_affine_offsets(self):
        for rows in product(range(8), repeat=3):
            for offset in (0, 1, 3, 7):
                events = [dict(rows=(row,), forbidden={0}, offset=(offset >> i) & 1)
                          for i, row in enumerate(rows)]
                actual = count_interfaces(events, 3)
                self.assertEqual(actual['successful_parameters'], exhaustive_count(events, 3))

    def test_affine_rank_deficient_outputs_and_deduplication(self):
        events = [dict(rows=(1, 1, 0), forbidden={0, 1, 2, 5}, offset=4),
                  dict(rows=(2, 3), forbidden={0, 2}, offset=1)]
        self.assertEqual(normalize_event(events[0], 2), dict(rows=(1,), forbidden=set()))
        self.assertEqual(count_interfaces(events, 2)['successful_parameters'], exhaustive_count(events, 2))

    def test_circuit_special_cases_and_recovered_code(self):
        for h, local, target, expected in ((4, (1, 2), 4, 0), (8, (3, 5), 8, 384)):
            profile = local+tuple(h+i for i in local)
            inputs = [u for u, w in all_words(local, [0]*h) if 0 < w < target]
            events, dimension = polar_outer_events(profile, [0]*(2*h), inputs, target-1)
            result = count_interfaces(events, dimension)
            circuit = analyze_circuit(profile, [0]*(2*h), inputs, target-1)
            self.assertEqual(result['successful_parameters'], expected)
            self.assertEqual(result['max_interface_rank'], circuit['common_rank'])
            self.assertEqual(expected, circuit['successful_outer_matrices'])
            if expected:
                columns = lift_outer_parameter(profile, [0]*(2*h), result['parameter'])
                self.assertEqual(min(w for _, w in all_words(profile, columns) if w), target)

    def test_genuine_polar_overlap_obstruction_not_seen_by_any_circuit(self):
        left, right = (3, 5, 7), (3, 5)
        profile = left+tuple(8+i for i in right)
        inputs = [u for u, w in all_words(left, [0]*8) if 0 < w < 8]
        events, dimension = polar_outer_events(profile, [0]*16, inputs, 7)
        self.assertEqual(inputs, [8, 32, 40, 136, 160, 168])
        circuits = []
        for size in (3, 4):
            for indices in combinations(range(6), size):
                messages = [inputs[i] for i in indices]
                total = 0
                for message in messages:
                    total ^= message
                if total or len(echelon(messages)) != size-1:
                    continue
                result = analyze_circuit(profile, [0]*16, messages, 7, lift=False)
                circuits.append(result['successful_outer_matrices'])
                self.assertGreater(circuits[-1], 0)
        self.assertEqual(sorted(circuits), [4096]*6+[24576])
        self.assertEqual(count_interfaces(events, dimension)['successful_parameters'], 0)
        self.assertEqual(exhaustive_count(events, dimension), 0)
        for i, expected in enumerate((256, 256, 256, 512, 512, 512)):
            subset = events[:i]+events[i+1:]
            self.assertEqual(count_interfaces(subset, dimension)['successful_parameters'], expected)

    def test_overlapping_feasible_groups_and_tree_invariance(self):
        local = (3, 5, 7)
        profile = local+tuple(8+i for i in local)
        inputs = [u for u, w in all_words(local, [0]*8) if 0 < w < 6]
        events, dimension = polar_outer_events(profile, [0]*16, inputs, 5)
        trees = (None, (((((0, 1), 2), 3), 4), 5), ((0, (2, 4)), (1, (3, 5))))
        for tree in trees:
            result = count_interfaces(events, dimension, tree=tree)
            self.assertEqual(result['successful_parameters'], 4864)
            columns = lift_outer_parameter(profile, [0]*16, result['parameter'])
            self.assertGreaterEqual(min(w for _, w in all_words(profile, columns) if w), 6)
        self.assertEqual(exhaustive_count(events, dimension), 4864)

    def test_independent_rows_need_no_interface(self):
        events = [dict(rows=(1, 2), forbidden={0}), dict(rows=(4, 8), forbidden={0, 1})]
        result = count_interfaces(events, 5)
        self.assertEqual(result['successful_parameters'], 12)
        self.assertEqual(result['max_interface_rank'], 0)

    def test_uniform_good_fiber_lower_bound(self):
        maps = ((1, 2), (4, 8), (1, 4), (2, 8), (3, 5))
        for rows in combinations(maps, 3):
            for bad_sets in product(((), (0,), (1, 2)), repeat=3):
                events = [dict(rows=row, forbidden=set(bad)) for row, bad in zip(rows, bad_sets)]
                normalized, _, nodes = prepare_interfaces(events, 4)
                private, product_minimum = 0, 1
                for node in nodes:
                    if node.children:
                        continue
                    event = normalized[node.indices[0]]
                    coordinates = coordinate_solver(event['rows'])
                    projection = [coordinates(row) for row in node.separator]
                    histogram = Counter(project(value, projection) for value in event['forbidden'])
                    private_rank = len(event['rows'])-len(node.separator)
                    private += private_rank
                    product_minimum *= (1 << private_rank)-max(histogram.values(), default=0)
                bound = (1 << (4-private))*product_minimum
                self.assertLessEqual(bound, exhaustive_count(events, 4))

    def test_validation_and_width_limit_are_not_infeasibility(self):
        self.assertEqual(count_interfaces([], 3)['successful_parameters'], 8)
        events = [dict(rows=(1,), forbidden={0})]*2
        with self.assertRaisesRegex(ValueError, 'Interface rank'):
            count_interfaces(events, 1, max_width=0)
        for tree in ((0, 0), (0, 2)):
            with self.assertRaises(ValueError):
                count_interfaces(events, 1, tree=tree)
        with self.assertRaises(ValueError):
            count_interfaces([dict(rows=(4,), forbidden={0})], 2)


if __name__ == '__main__':
    unittest.main()
