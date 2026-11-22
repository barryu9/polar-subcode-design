"""Necessary distance tests and exact minimum-layer bounds from polar suffixes.

The Hamming obstruction applies to all causal supports with the supplied
profile. The weight-six moment bound additionally assumes distance at least six,
evenness, and the all-one word. All proof arithmetic uses integers or fractions.
"""

from collections import Counter
from math import comb
from reference_weak_structure_audit import check_rows, binary_rank

def krawtchouk(n, degree, weight):
    return sum(
        (-1) ** j * comb(weight, j) * comb(n - weight, degree - j)
        for j in range(max(0, degree - (n - weight)), min(degree, weight) + 1))

def dual_macwilliams_spectrum(profile, columns, max_dual_dimension=20):
    checks = check_rows(profile, columns)
    if len(checks) > max_dual_dimension:
        raise ValueError("Dual enumeration exceeds the configured dimension")
    if binary_rank(checks) != len(checks):
        raise AssertionError("Physical checks are not independent")
    words = [0]
    for check in checks:
        words.extend([word ^ check for word in words])
    dual = Counter(word.bit_count() for word in words)
    n, divisor = len(columns), len(words)
    spectrum = {}
    for weight in range(n + 1):
        numerator = sum(count * krawtchouk(n, weight, j) for j, count in dual.items())
        count, remainder = divmod(numerator, divisor)
        if remainder or count < 0:
            raise AssertionError("Invalid exact MacWilliams coefficient")
        if count:
            spectrum[weight] = count
    if spectrum.get(0) != 1 or sum(spectrum.values()) != 1 << len(profile):
        raise AssertionError("MacWilliams dimension or zero-word check failed")
    return dict(dual_dimension=len(checks), enumerated_dual_words=divisor,
                dual_spectrum=dict(sorted(dual.items())), spectrum=spectrum)
