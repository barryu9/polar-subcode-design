"""Regression checks for the numbered algorithms and finite proof checks."""
import unittest
from itertools import combinations, product

from polar_subcodes.code import PolarSubcode
from algorithms.algorithm1_directory import from_causal_checks, list_inputs
from algorithms.algorithm2_outer import complete_outer
from algorithms.algorithm3_fixed_profile import construct, Deferred, Outcome, Suffix
from algorithms.algorithm4_profiles import design_profiles, frame_loss_measure, refine
from theory.check_results import CHECKS, configurations, independent_words, join


class PaperAlgorithms(unittest.TestCase):
    def test_check_substitution(self):
        # u_1=u_0, u_2=u_1+u_0=0, u_3=u_2+u_1=u_0
        code = from_causal_checks(4, (0b1110, 0b0011, 0b0111))
        self.assertEqual(code.columns, (0, 1, 0, 1))
        self.assertEqual(sorted(list_inputs(code, 4)), [0, 0b1011])

    def test_every_numbered_result(self):
        for check in CHECKS:
            with self.subTest(result=check.__name__):
                self.assertTrue(check())

    def test_outer_independent_affine_domains(self):
        left = PolarSubcode(4, (0, 1), (0, 0, 0, 2))
        right = PolarSubcode(4, (0, 1), (0, 0, 1, 3))
        # Nonzero base columns, restricted bases, and zero-parameter columns.
        for a, b in product(range(4), repeat=2):
            domains = ((a, (3,)), (b, ()))
            for cutoff in range(6):
                admissible = {(a, b), (a ^ 3, b)}
                good = {t for t in admissible if all(
                    not x or x.bit_count() > cutoff for _, x in independent_words(join(left, right, t)))}
                result = complete_outer(left, right, cutoff, domains)
                self.assertIn(result, good or {None})
        with self.assertRaises(ValueError):
            complete_outer(left, right, 3, [(0, (1, 1)), (0, ())])

    def test_algorithm3_covers_all_length_four_profiles(self):
        for k in range(5):
            for profile in combinations(range(4), k):
                possible = [min((x.bit_count() for _, x in independent_words(c) if x),
                                default=float("inf")) for c in configurations(4, profile)]
                for target in range(1, 6):
                    for method in ("direct", "factored"):
                        outcome = construct(4, profile, target, method=method)
                        self.assertEqual(outcome.status == "verified", max(possible) >= target)
                        if outcome.code:
                            self.assertFalse(any(list_inputs(outcome.code, target-1)))
        self.assertTrue(construct(8, (3, 5), 5).released)

    def test_deferral_returns_available_suffix_outcomes(self):
        # Inject a deterministic work interruption: no stopping condition applied
        # to an unfinished branch can be promoted into a proof.
        found = False
        for limit in range(1, 80):
            counter = iter(range(limit))
            def tick():
                if next(counter, None) is None:
                    raise Deferred
            result = construct(8, (1,7), 4, suffix_lengths=(2,4), tick=tick)
            if tuple(s.status for s in result.suffixes) == ("verified", "unresolved"):
                self.assertEqual(result.status, "unresolved")
                self.assertFalse(result.released)
                found = True
                break
        self.assertTrue(found)

    def test_profile_search_handles_empty_seed_and_verified_stages(self):
        args = (4, 2, 2, (3, 2, 1, 0), (3, 2, 1, 0))
        family = list(combinations(range(4), 2))
        self.assertFalse(design_profiles(*args, family, 3, excluded=lambda p: True).outcomes)
        result = design_profiles(*args, family, 3)
        self.assertEqual(set(result.outcomes), set(family))
        self.assertEqual(len(set(result.selected)), 3)
        for stages in result.candidates.values():
            self.assertEqual(len(stages), 3)
            self.assertTrue(all(not any(list_inputs(c, 1)) for c in stages))

    def test_unresolved_suffix_changes_order_but_excludes_no_profile(self):
        visited = []
        def solver(n, profile, target):
            visited.append(profile)
            if len(visited) == 1:
                return Outcome("unresolved", suffixes=(Suffix(2, (1,), "verified"),
                                                       Suffix(4, (1, 3), "unresolved")))
            return Outcome("unresolved")
        # Seed {5,7} has both a ring-changing exchange (5->0) and a cheaper
        # exchange affecting the verified suffix (7->6).
        family = [(5, 7), (0, 7), (5, 6)]
        args = (8, 2, 4, (5,7,6,4,3,2,1,0), (7,6,5,4,3,0,2,1))
        result = design_profiles(*args, family, 1, solver=solver)
        self.assertEqual(visited, [(5, 7), (0, 7), (5, 6)])
        self.assertFalse(result.candidates)
        # A full-suffix proof *does* exclude each profile with that suffix.
        first = family[0]
        def proved(n, profile, target):
            previous = (Suffix(2, (1,), "infeasible"),) if profile == first else ()
            return Outcome("infeasible", suffixes=previous)
        result = design_profiles(*args, family, 1, solver=proved)
        self.assertNotIn((0, 7), result.outcomes)

    def test_fixed_observation_refinement(self):
        code = PolarSubcode(8, (3, 4, 7), (0,) * 8)
        loss = frame_loss_measure([[3.0]*8, [-2.0]*8], 2)
        spectral, protected = refine(code, 2, 2, loss)
        low = lambda c: {x for _, x in independent_words(c) if x.bit_count() <= 2}
        self.assertEqual(low(spectral), low(protected))
        self.assertLessEqual(loss(protected), loss(spectral))


if __name__ == "__main__":
    unittest.main()
