"""Profile-wide distance obstruction from a static prefix in a repeated suffix.

The static prefix supplies redundant coordinate signatures. Pigeonhole
collisions in the remaining check labels force weight-two suffix codewords.
Full-length witnesses are verified against every dynamic frozen equation.
"""

from math import comb

def bound(n, profile, length):
    if (n < 2 or n & (n - 1) or n < length or n % length
            or length < 2 or length & (length - 1)):
        raise ValueError('Expected an aligned power-of-two suffix')
    if len(set(profile)) != len(profile) or any(i < 0 or i >= n for i in profile):
        raise ValueError('Invalid information positions')
    shift = n - length
    local = tuple(i - shift for i in profile if i >= shift)
    if not local:
        return None
    prefix = 1 << (min(local).bit_length() - 1) if min(local) else 0
    if not prefix:
        return None
    checks = sorted(set(range(length)) - set(local))
    remaining = len(checks) - prefix
    signature_cap = prefix * (1 << remaining)
    if signature_cap >= length:
        return None
    quotient, remainder = divmod(length, signature_cap)
    pairs = ((signature_cap - remainder) * comb(quotient, 2)
             + remainder * comb(quotient + 1, 2))
    return dict(suffix_length=length, suffix_dimension=len(local), offset=shift,
                local_profile=list(local), frozen=checks, static_prefix=prefix,
                varying_check_labels=remaining, distinct_signature_upper_bound=signature_cap,
                minimum_equal_column_pairs=pairs, repetitions=n // length,
                distance_upper_bound=2 * (n // length),
                A_at_upper_bound_lower_bound=pairs)
