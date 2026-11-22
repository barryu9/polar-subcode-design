"""Distance upper bounds from a forced Hamming constituent and a fixed row.

Canonical causal expressions on the profile of all indices of binary weight
at least two always yield an extended Hamming code. Their check columns form
a triangular permutation of the affine binary point set.
"""

from recursive_profile_completion import coordinates

def hamming_profile(length):
    return [i for i in range(length) if i.bit_count() >= 2]

def structural_bound(n, profile, parent_length):
    half = parent_length//2
    local = [i-(n-parent_length) for i in profile if i >= n-parent_length]
    left = [i for i in local if i < half]
    right = [i-half for i in local if i >= half]
    if right != hamming_profile(half):
        return None
    last_frozen = max(set(range(half))-set(left),default=-1)
    fixed = [(i,1 << i.bit_count()) for i in left if i > last_frozen]
    if not fixed:
        return None
    source,weight = min(fixed,key=lambda p:(p[1],p[0]))
    return dict(parent_length=parent_length, right_length=half, right_dimension=len(right),
                right_profile=right, right_internal_parameters=len(coordinates(half,right)),
                fixed_left_source=source, fixed_left_weight=weight,last_left_frozen=last_frozen,
                punctured_covering_radius=1,local_upper_bound=weight+2,
                repetitions=n//parent_length,full_upper_bound=(weight+2)*(n//parent_length))
