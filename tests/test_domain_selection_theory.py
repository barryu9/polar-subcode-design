"""Exhaustive small-code checks of the manuscript's domain-selection claims."""
from itertools import combinations
import unittest

from polar_subcodes.algebra import polar_transform, rref


def span(basis):
    values = {0}
    for v in basis:
        values |= {u ^ v for u in values}
    return values


def profile_rule(n, profile, target):
    baseline = min(1 << i.bit_count() for i in profile)
    rows = tuple(j for j in range(n) if j not in profile and j > min(profile)
                 and (1 << j.bit_count()) >= baseline)
    core = tuple(i for i in profile if (1 << i.bit_count()) >= target)
    edges = tuple((j, i) for j in range(n) if j not in profile for i in profile if i < j)
    first = tuple(q for q, (j, i) in enumerate(edges) if j in rows and i not in core)
    missing_core = tuple(q for q, (j, i) in enumerate(edges)
                         if j in rows and i in core)
    missing_rows = tuple(sorted(
        (q for q, (j, _) in enumerate(edges) if j not in rows),
        key=lambda q: (-(1 << edges[q][0].bit_count()), *edges[q])))
    return baseline, rows, core, edges, first, missing_core+missing_rows


def codewords(n, profile, edges, support):
    basis = []
    for source in profile:
        u = 1 << source
        for q, (j, i) in enumerate(edges):
            if i == source and support >> q & 1:
                u ^= 1 << j
        basis.append(polar_transform(u, n))
    return span(basis)


class DomainSelectionTheoryTests(unittest.TestCase):
    def test_profile_inclusions_and_protection_across_profiles_and_rates(self):
        cases = [(4, tuple(i for i in range(4) if mask >> i & 1))
                 for mask in range(1, 16)]
        cases += [(8, (0,)), (8, (3, 7)), (8, (0, 2, 4, 6)),
                  (8, tuple(range(1, 8)))]
        for n, profile in cases:
            for target in (1, 2, 3, n):
                baseline, rows, core, edges, first, _ = profile_rule(n, profile, target)
                mother = span([polar_transform(1 << i, n) for i in (*profile, *rows)])
                pure = span([polar_transform(1 << i, n) for i in core])
                self.assertEqual(len(mother), 1 << (len(profile)+len(rows)))
                self.assertEqual(min(w.bit_count() for w in mother if w), baseline)
                self.assertTrue(all(not w or w.bit_count() >= target for w in pure))
                missing = set(range(1, target)) - {w.bit_count() for w in mother}
                for support in span([1 << q for q in first]):
                    words = codewords(n, profile, edges, support)
                    self.assertTrue(pure <= words <= mother)
                    self.assertFalse(missing & {w.bit_count() for w in words})

    def test_relaxation_shells_are_disjoint_and_cover_all_supports(self):
        for mask in range(1, 16):
            profile = tuple(i for i in range(4) if mask >> i & 1)
            for target in (1, 2, 3, 4):
                _, _, _, edges, first, order = profile_rule(4, profile, target)
                covered = span([1 << q for q in first])
                free = list(first)
                for q in order:
                    shell = {s | (1 << q) for s in span([1 << r for r in free])}
                    self.assertFalse(covered & shell)
                    covered |= shell
                    free.append(q)
                self.assertEqual(covered, set(range(1 << len(edges))))

    def test_affine_image_protection_iff_every_member_is_safe(self):
        n, profile, rows = 4, (0, 2), (1, 3)
        edges = ((1, 0), (3, 0), (3, 2))
        codes = {s: codewords(n, profile, edges, s) for s in range(8)}
        subspaces = {tuple(rref(vectors)) for count in range(4)
                     for vectors in combinations(range(1, 8), count)}

        def image(u, parameter):
            result = 0
            for q, (j, i) in enumerate(edges):
                result ^= (((parameter >> q) & (u >> i) & 1) << rows.index(j))
            return result

        for origin in range(8):
            for basis in subspaces:
                directions = span(basis)
                domain = {origin ^ delta for delta in directions}
                for protected in ({1}, {2}, {1, 2, 3}):
                    predicted = True
                    for u in range(1, 1 << n):
                        if polar_transform(u, n).bit_count() not in protected:
                            continue
                        rhs = sum(((u >> j) & 1) << q for q, j in enumerate(rows))
                        residual = rhs ^ image(u, origin)
                        predicted &= residual not in {image(u, d) for d in directions}
                    observed = all(not ({w.bit_count() for w in codes[s]} & protected)
                                   for s in domain)
                    self.assertEqual(predicted, observed)


if __name__ == "__main__":
    unittest.main()
