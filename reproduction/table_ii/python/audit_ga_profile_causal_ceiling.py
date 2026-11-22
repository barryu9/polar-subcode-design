"""Audit GA/RM-polar profiles with an aligned immutable SPC suffix.

The distance bounds apply to all homogeneous causal supports on the fixed
profile, without an RM constraint or a restriction on the number of taps.
"""

def causal_distance_bounds(n, profile):
    profile = tuple(sorted(profile))
    if (n < 1 or n & (n - 1) or not profile
            or len(set(profile)) != len(profile)
            or any(i < 0 or i >= n for i in profile)):
        raise ValueError("Invalid length or information set")
    information = set(profile)
    frozen = tuple(i for i in range(n) if i not in information)
    last = max(frozen, default=-1)
    lower = min(1 << i.bit_count() for i in profile)
    tail = tuple(i for i in profile if i > last)
    upper = min((1 << i.bit_count() for i in tail), default=None)
    return dict(
        N=n, K=len(profile), unprecoded_dmin=lower,
        all_causal_support_distance_lower_bound=lower,
        last_frozen_index=last, immutable_suffix_dimension=len(tail),
        immutable_row_distance_upper_bound=upper,
        immutable_minimum_row_indices=[i for i in tail if 1 << i.bit_count() == upper],
        exact_dmin_for_every_causal_support=lower if upper == lower else None,
        unrestricted_causal_coefficients=sum(sum(i < j for i in profile) for j in frozen))
