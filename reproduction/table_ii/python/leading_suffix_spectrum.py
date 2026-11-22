"""Exact shortening using the leading polar-coset weight bound.

If every information row in the left half has weight above the cutoff, any
low-weight word has all left inputs zero. This holds for arbitrary canonical
causal expressions, not only for a static left component.
"""

from extended_exact_spectrum import measure
from feasible_design_bank import validate_design
from refine_recursive_suffix_spectrum import suffix_design

def leading_suffix_measure(n, profile, columns, cutoff, max_work=100_000_000,
                           policy='enumerate'):
    profile = validate_design(n, profile, columns)
    if cutoff < 0:
        raise ValueError('The cutoff must be nonnegative')
    original_n, original_k, original_cutoff = n, len(profile), cutoff
    factor, stages = 1, []
    while n > 1:
        half = n//2
        left = [i for i in profile if i < half]
        minimum = min((1 << i.bit_count() for i in left), default=n+1)
        if minimum <= cutoff:
            break
        stages.append(dict(N=n, cutoff=cutoff, left_information=left,
                           minimum_left_information_coset_weight=minimum,
                           argument='A low-weight codeword cannot have its leading information input in the left half. Causality forces all other left inputs to zero.'))
        profile, columns = suffix_design(n, profile, columns, half)
        n, cutoff, factor = half, cutoff//2, factor*2
    answer = measure(n, profile, columns, cutoff, max_work, policy=policy)
    return dict(answer, N=original_n, K=original_k, cutoff=original_cutoff,
                spectrum={factor*w: count for w, count in answer['spectrum'].items()},
                repetition_factor=factor*answer['repetition_factor'],
                leading_coset_reduction=stages)
