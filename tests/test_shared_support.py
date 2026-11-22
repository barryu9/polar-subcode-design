"""Checks for the shared-support mainline and its exact outer specialization."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import unittest

from polar_subcodes.construction.conditional import xor_mask
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.algebra import all_words
from polar_subcodes.algebra import polar_transform, rref


def support_columns(n, edges, mask):
    columns = [0]*n
    for bit, (j, i) in enumerate(edges):
        columns[j] |= ((mask >> bit) & 1) << i
    return columns


def residual_increment(edges, mask, sources, destinations, pattern):
    message = sum(((pattern >> q) & 1) << i for q, i in enumerate(sources))
    return sum((sum(((mask >> b) & 1) * ((message >> i) & 1)
                    for b, (j, i) in enumerate(edges) if j == row) & 1) << t
               for t, row in enumerate(destinations))


def directory(n, profile, edges, baseline, cutoff):
    sources = sorted({i for _, i in edges})
    destinations = sorted({j for j, _ in edges})
    columns = support_columns(n, edges, baseline)
    fixed = [(1 << j) ^ columns[j] for j in range(n)
             if j not in profile and j not in destinations]
    groups, multiplicities = defaultdict(set), Counter()
    for u in range(1, 1 << n):
        if any((u & row).bit_count() & 1 for row in fixed):
            continue
        if (u & -u).bit_length()-1 not in profile:
            continue
        weight = polar_transform(u, n).bit_count()
        if weight > cutoff:
            continue
        pattern = sum(((u >> i) & 1) << q for q, i in enumerate(sources))
        beta = sum((((u & ((1 << j) ^ columns[j])).bit_count() & 1) << q)
                   for q, j in enumerate(destinations))
        groups[pattern].add(beta)
        multiplicities[weight, pattern, beta] += 1
    return sources, destinations, groups, multiplicities


class SupportCentricManuscriptTests(unittest.TestCase):
    def test_replacement_example_rejects_old_but_admits_new_words(self):
        records = []
        for alpha in (0, 1):
            columns = [0, 0, alpha << 1, 0]
            records.append({polar_transform(u, 4) for u, w in all_words((1, 3), columns) if w == 2})
        self.assertEqual(records, [{0b0011, 0b1100}, {0b0110, 0b1001}])
        self.assertFalse(records[0] & records[1])
        _, _, groups, counts = directory(4, (1, 3), [(2, 1)], 0, 2)
        self.assertEqual(dict(groups), {1: {0, 1}})
        self.assertEqual(counts, {(2, 1, 0): 2, (2, 1, 1): 2})

    def test_affine_support_means_counts_and_greedy_completion(self):
        strict_nonleaf = 0
        cases = [
            (4, (1, 3), 0, 0, (1,)),
            (8, (1, 4), 23, 0, (8, 16, 32)),
            (8, (1, 4), 23, 1, (8 ^ 16, 16 ^ 32, 64)),
            (8, (3, 5, 7), 0, 0, (1, 2)),
        ]
        for n, profile, baseline, origin, directions in cases:
            edges = [(j, i) for j in range(n) if j not in profile
                     for i in profile if i < j]
            self.assertEqual(len(rref(directions)), len(directions))
            self.assertLess(baseline | origin | sum(directions), 1 << len(edges))
            p = len(directions)
            for cutoff in (2, 3, 4):
                sources, destinations, groups, multiplicities = directory(
                    n, profile, edges, baseline, cutoff)
                patterns = sorted(groups)
                b = len(destinations)

                def outputs(theta):
                    delta = origin ^ xor_mask(directions, theta)
                    return tuple(residual_increment(edges, delta, sources, destinations, q)
                                 for q in patterns)

                def failures(theta):
                    return sum(value in groups[q] for q, value in zip(patterns, outputs(theta)))

                def completions(bits):
                    prefix = sum(bit << j for j, bit in enumerate(bits))
                    return [prefix | (rest << len(bits)) for rest in range(1 << (p-len(bits)))]

                def mean(bits):
                    candidates = completions(bits)
                    total = Fraction(0)
                    for pos, q in enumerate(patterns):
                        image = {outputs(theta)[pos] for theta in candidates}
                        total += Fraction(len(image & groups[q]), len(image))
                    return total

                joint = {outputs(theta) for theta in range(1 << p)}
                linear_columns = []
                offset = outputs(0)
                for j in range(p):
                    linear_columns.append(sum((value ^ old) << (b*k)
                                              for k, (value, old) in enumerate(zip(outputs(1 << j), offset))))
                rank = len(rref(linear_columns))
                good_joint = [ys for ys in joint if all(y not in groups[q] for q, y in zip(patterns, ys))]
                good = [theta for theta in range(1 << p) if not failures(theta)]
                self.assertEqual(len(good), (1 << (p-rank))*len(good_joint))
                for theta in range(1 << p):
                    columns = support_columns(n, edges, baseline ^ origin ^ xor_mask(directions, theta))
                    actual = Counter(w for _, w in all_words(profile, columns) if 0 < w <= cutoff)
                    predicted = Counter()
                    for q, value in zip(patterns, outputs(theta)):
                        for weight in range(1, cutoff+1):
                            predicted[weight] += multiplicities[weight, q, value]
                    self.assertEqual(+predicted, actual)
                    self.assertEqual(bool(failures(theta)), bool(actual))
                for fixed in range(p+1):
                    for bits in product((0, 1), repeat=fixed):
                        candidates = completions(bits)
                        phi = mean(bits)
                        self.assertEqual(phi, Fraction(sum(failures(v) for v in candidates), len(candidates)))
                        exact_good = sum(not failures(v) for v in candidates)
                        lower = max(Fraction(0), len(candidates)*(1-phi))
                        self.assertGreaterEqual(exact_good, -(-lower.numerator // lower.denominator))
                        if fixed < p:
                            self.assertEqual(phi, (mean(bits+(0,))+mean(bits+(1,)))/2)
                        if phi < 1:
                            strict_nonleaf += int(fixed < p and bool(patterns))
                            path = bits
                            while len(path) < p:
                                bit = min((0, 1), key=lambda value: (mean(path+(value,)), value))
                                path += (bit,)
                            self.assertFalse(failures(completions(path)[0]))
        self.assertGreater(strict_nonleaf, 0)

    def test_outer_quotient_is_exact_preimage_of_word_forbidden_set(self):
        for profile in ((1, 3, 5, 7), (1, 4), (3, 5, 7, 11, 13)):
            n = 16 if max(profile) >= 8 else 8
            for baseline in (0, 1):
                columns = [sum(baseline << i for i in profile if i < j)
                           if j not in profile else 0 for j in range(n)]
                model = RelativePlotkinSpectrum(profile, columns)
                for u, weight in all_words(model.left_profile, model.left_columns):
                    if not weight:
                        continue
                    a = polar_transform(u, model.half)
                    for radius in range(3):
                        cutoff = weight+2*radius
                        condition = model.constraint(u, cutoff)
                        raw = {xor_mask(model.check_columns, v) for v in range(1 << model.half)
                               if (a ^ v).bit_count()+v.bit_count() <= cutoff}
                        projected = {syndrome for syndrome in range(1 << model.redundancy)
                                     if xor_mask(condition["syndrome_map"], syndrome) in condition["forbidden"]}
                        self.assertEqual(raw, projected)



if __name__ == "__main__":
    unittest.main()
