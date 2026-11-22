"""Exact spectral-feedback exchanges with bounded cold recursive completion.

The starting code supplies feedback only. Each different profile is constructed
from zero coefficients using shared constraints and the recursive exact solver.
When the full minimum layer cannot be catalogued, a fixed repeated-suffix
component supplies feedback. Component improvement never implies full-spectrum
improvement. Every accepted refinement verifies the full distance separately.
"""

from collections import Counter
from obstruction_support_search import enumerate_words
from refine_recursive_suffix_spectrum import suffix_design

def directory(n, profile, columns, target, length, work):
    factor = n//length
    if target % factor:
        raise ValueError('The suffix objective must have an integral local target')
    lp, lc = suffix_design(n, profile, columns, length)
    words, metadata = enumerate_words(length, lp, lc, target//factor, work)
    if any(w < target//factor for _, w in words):
        raise AssertionError('A repeated suffix contradicts the full distance')
    histogram = Counter((u & -u).bit_length()-1+n-length
                        for u, w in words if w == target//factor)
    return dict(length=length, local_target=target//factor, count=sum(histogram.values()),
                source_A_target={str(i): a for i, a in sorted(histogram.items())},
                exact=True, full_layer=length==n, oracle=metadata)
