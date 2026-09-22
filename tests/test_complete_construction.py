"""Exhaustive verification of complete word-based construction."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
import math
import random
import unittest
from polar_subcodes.construction.conditional import xor_mask
from polar_subcodes.spectrum.quotient import evaluate
from polar_subcodes.theory.exhaustive import (codewords,profile_rule,span,word_list,
    source_candidates,outer_completion,support_stream,trial_construct)
from polar_subcodes.algebra import polar_transform

class ShortCodewordAlgorithmTests(unittest.TestCase):
    def test_actual_quotient_lifting_matches_exhaustive_inputs(self):
        rng = random.Random(2319)
        for n in (1, 2, 4, 8):
            checks_list = [[], [1 << j for j in range(n)]]
            checks_list += [[(1 << j) | rng.randrange(1 << j)
                             for j in range(n) if rng.randrange(2)] for _ in range(20)]
            for checks in checks_list:
                for cutoff in range(-1, n+2):
                    case = dict(kind="directory", n=n, checks=checks, cutoff=cutoff)
                    result, _ = evaluate(case, "quotient")
                    expected = {u for u in range(1 << n)
                                if polar_transform(u, n).bit_count() <= cutoff
                                and all(not ((u & row).bit_count() & 1) for row in checks)}
                    self.assertEqual(set(result), expected)
                    self.assertEqual(len(result), len(expected))

    def test_complete_trial_pipeline_matches_all_supports(self):
        cases = [(n, tuple(i for i in range(n) if mask >> i & 1))
                 for n in (1, 2, 4) for mask in range(1 << n)]
        cases += [(8, p) for p in ((), (0, 7), (3, 7), (1, 3, 5, 7))]
        for n, profile in cases:
            edges = tuple((j, i) for j in range(n) if j not in profile
                          for i in profile if i < j)
            distances = {s: min((w.bit_count() for w in codewords(n, profile, edges, s)
                                 if w), default=math.inf)
                         for s in range(1 << len(edges))}
            for target in range(1, n+2):
                result = trial_construct(n, profile, target)
                self.assertEqual(result is not None, max(distances.values()) >= target)
                if result is not None:
                    self.assertGreaterEqual(distances[result], target)

    def test_recursive_directory_matches_direct_inputs(self):
        rng = random.Random(731)
        for n in (1, 2, 4, 8):
            for _ in range(12):
                checks = [(1 << j) | rng.randrange(1 << j)
                          for j in range(n) if rng.randrange(2)]
                for cutoff in range(-1, n+1):
                    result = word_list(n, checks, cutoff)
                    expected = {u for u in range(1 << n)
                                if polar_transform(u, n).bit_count() <= cutoff
                                and all(not ((u & h).bit_count() & 1) for h in checks)}
                    self.assertEqual(set(result), expected)
                    self.assertEqual(len(result), len(set(result)))

    def test_domains_and_source_iterator_cover_exactly_all_solutions(self):
        profiles = [(4, tuple(i for i in range(4) if mask >> i & 1))
                    for mask in range(1, 16)] + [(8, (1, 3, 5, 7))]
        for n, profile in profiles:
            for target in (1, 2, 3, 4, n):
                _, _, _, edges, first, order = profile_rule(n, profile, target)
                domains = [(0, tuple(first))]
                free = list(first)
                for edge in order:
                    domains.append((1 << edge, tuple(free)))
                    free.append(edge)
                found = []
                for origin, free in domains:
                    rows = sorted({edges[e][0] for e in free})
                    row_id = {j: b for b, j in enumerate(rows)}
                    columns = [sum(((origin >> e) & 1) << i
                                   for e, (j, i) in enumerate(edges) if j == row)
                               for row in range(n)]
                    checks = [(1 << j) | columns[j] for j in range(n)
                              if j not in profile and j not in rows]
                    patterns = defaultdict(set)
                    sources = sorted({edges[e][1] for e in free})
                    for u in word_list(n, checks, target-1):
                        if not u or (u & -u).bit_length()-1 not in profile:
                            continue
                        q = sum((u & (1 << i)) for i in sources)
                        beta = sum((((u & ((1 << j) | columns[j])).bit_count() & 1) << b)
                                   for b, j in enumerate(rows))
                        patterns[q].add(beta)
                    source_domains = {i: tuple(sorted(span([1 << row_id[j]
                                      for e, (j, k) in enumerate(edges) if e in free and k == i])))
                                      for i in sources}
                    for assignment in source_candidates(patterns, source_domains):
                        support = origin
                        for edge in free:
                            j, i = edges[edge]
                            support ^= ((assignment[i] >> row_id[j]) & 1) << edge
                        found.append(support)
                expected = {s for s in range(1 << len(edges))
                            if all(not x or x.bit_count() >= target
                                   for x in codewords(n, profile, edges, s))}
                self.assertEqual(set(found), expected)
                self.assertEqual(len(found), len(set(found)))

    def test_outer_returns_to_earlier_source_after_safe_column(self):
        groups = {1: [(2, (1, 2), {1, 2, 3}), (3, (1, 2), {0})]}
        answer, stats = outer_completion(groups, 2, 2)
        self.assertIsNotNone(answer)
        self.assertNotEqual(answer[0], 0)
        self.assertEqual(answer[1], 0)
        self.assertGreater(stats['failed_columns'], 0)

    def test_outer_existence_matches_exhaustive_cross_matrices(self):
        rng = random.Random(937)
        for _ in range(120):
            groups = defaultdict(list)
            for message in (1, 2, 3):
                columns = tuple(rng.randrange(4) for _ in range(2))
                forbidden = {v for v in range(4) if rng.randrange(2)}
                groups[message.bit_length()-1].append((message, columns, forbidden))
            solutions = [values for values in product(range(4), repeat=2)
                         if all(xor_mask(columns, xor_mask(values, message)) not in bad
                                for records in groups.values() for message, columns, bad in records)]
            answer, _ = outer_completion(groups, 2, 2)
            self.assertEqual(answer is not None, bool(solutions))
            if answer is not None:
                self.assertIn(answer, solutions)
