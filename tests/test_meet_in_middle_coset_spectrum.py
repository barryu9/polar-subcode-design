from collections import Counter
from itertools import combinations
import random
import unittest

from polar_subcodes.spectrum.meet_in_middle import small_error_counts, relative_spectrum
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.algebra import all_words


class MeetInMiddleCosetTests(unittest.TestCase):
    def test_support_multiplicity(self):
        for columns in ([], [0]*7, [0, 1, 1, 2, 2, 3, 0], list(range(8))):
            for syndrome in range(8):
                expected = [sum(self.xor(columns[i] for i in support) == syndrome
                                for support in combinations(range(len(columns)), size))
                            for size in range(7)]
                self.assertEqual(small_error_counts(columns, syndrome, 6), expected)

    @staticmethod
    def xor(values):
        answer = 0
        for value in values:
            answer ^= value
        return answer

    def test_relative_against_full_enumeration(self):
        rng = random.Random(1932)
        profile = (1, 3, 6, 7, 9, 11, 13, 15)
        for _ in range(8):
            columns = [sum(rng.randrange(2) << i for i in profile if i < j)
                       if j not in profile else 0 for j in range(16)]
            model = RelativePlotkinSpectrum(profile, columns)
            left = all_words(model.left_profile, model.left_columns)
            right = Counter(w for _, w in all_words(model.right_profile, model.right_columns))
            expected = Counter(w for _, w in all_words(profile, columns) if w <= 12)
            self.assertEqual(relative_spectrum(model, left, right, 12)['spectrum'], expected)


if __name__ == '__main__':
    unittest.main()
