"""A priori outer-extension bounds for a retained polar constituent.

The certificate is conditional on a right code of dimension k_R, distance at
least delta, and containing the zero-support left polar code. No outer support,
forbidden-syndrome set, or successful search prefix is used to compute it.
"""
from collections import Counter
from fractions import Fraction
from math import comb, prod


def ball_volume(length, radius):
    if length < 0:
        raise ValueError('Negative length')
    return sum(comb(length, t) for t in range(min(radius, length)+1))


def grouped_left_spectrum(profile, words):
    mask = sum(1 << i for i in profile)
    groups = {i: Counter() for i in profile}
    for u, weight in words:
        if not weight:
            continue
        message = u & mask
        if not message:
            raise ValueError('A nonzero word has no information source')
        groups[message.bit_length()-1][weight] += 1
    return groups


def extension_bound(half, right_dimension, right_distance_lower_bound, target, groups):
    """Integer bad-column upper bounds, conditional on the retained-code hypothesis."""
    if not (0 <= right_dimension <= half and 1 <= target <= 2*right_distance_lower_bound):
        raise ValueError('Invalid parameters or target above twice the right distance guarantee')
    f = half-right_dimension
    size = 1 << f
    loads = {}
    for source, histogram in groups.items():
        load = 0
        for weight, count in histogram.items():
            if not 0 < weight < target:
                continue
            if weight < right_distance_lower_bound:
                raise ValueError('The left code contradicts the retained-code distance hypothesis')
            radius = (target-1-weight)//2
            load += count*(1 << (weight-1))*ball_volume(half-weight, radius)
        loads[source] = load
    guaranteed = all(load < size for load in loads.values())
    return dict(target=target, redundancy=f, guaranteed=guaranteed,
                bad_column_upper_bounds=loads,
                group_mean_upper_bounds={i: str(Fraction(v, size)) for i, v in loads.items()},
                maximum_group_mean_upper_bound=str(Fraction(max(loads.values(), default=0), size)),
                completion_count_lower_bound=prod(size-load for load in loads.values()) if guaranteed else 0,
                right_code_hypothesis_required=True,
                failure_means='sufficient condition inconclusive, not infeasible')


def first_order_profile_family(exponent):
    """Closed-form nonvacuous polar family; no code search enters the bound."""
    if exponent < 4:
        raise ValueError('The stated family starts at exponent four')
    h = 1 << exponent
    length = h//2
    profile = tuple(sorted([h-1-(1 << q) for q in range(exponent)]+[h-1]))
    counts = [1 << j for j in range(exponent)]+[(1 << exponent)-1]
    best = None
    for radius in range(length//2):
        volume = ball_volume(length, radius)
        ratio = Fraction(((1 << exponent)-1)*volume, 1 << (length-exponent))
        if ratio >= 1:
            break
        best = dict(radius=radius, guaranteed_distance=length+2*radius+2,
                    maximum_group_mean_upper_bound=str(ratio), error_ball_volume=volume)
    if best is None:
        raise AssertionError('Radius zero must be feasible for every exponent >=4')
    return dict(exponent=exponent, N=2*h, K=2*(exponent+1),
                constituent_profile=profile, profile=profile+tuple(h+i for i in profile),
                baseline_distance=length, right_distance=length,
                critical_group_sizes=counts, **best)
