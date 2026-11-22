"""Universal weight-two obstruction from causal prefix check signatures.

Only coefficients in a short prefix are enumerated. Every remaining check
is allowed arbitrary labels, giving an upper bound on its separating power.
An occupancy collision bound is therefore valid for all causal supports.
"""

from collections import Counter
from math import comb

def collision_floor(points, bins):
    """Smallest number of equal-label pairs in any assignment to bins."""
    q, remainder = divmod(points, bins)
    return (bins - remainder) * comb(q, 2) + remainder * comb(q + 1, 2)

def prefix_bound(n, profile, length, max_prefix_coefficients=12):
    if (n < length or n & (n - 1) or length < 2 or length & (length - 1)
            or len(set(profile)) != len(profile) or any(i < 0 or i >= n for i in profile)):
        raise ValueError("Invalid profile or dyadic suffix length")
    shift = n - length
    local = sorted(i - shift for i in profile if i >= shift)
    if not local:
        return None
    checks = sorted(set(range(length)) - set(local))
    if not checks:
        return None
    edges = []
    for size, j in enumerate(checks, 1):
        edges.extend((j, i) for i in local if i < j)
        if len(edges) > max_prefix_coefficients:
            break
        prefix = checks[:size]
        remaining_bins = 1 << (len(checks) - size)
        # No prefix occupancy can overflow this capacity.
        if remaining_bins >= length:
            continue
        settings, worst_pairs, max_labels = [], None, 0
        for setting in range(1 << len(edges)):
            columns = {row: 0 for row in prefix}
            for bit, (row, source) in enumerate(edges):
                columns[row] |= ((setting >> bit) & 1) << source
            counts = Counter()
            for point in range(length):
                signature = 0
                for q, row in enumerate(prefix):
                    value = int(point & row == row)
                    value ^= sum(int(point & i == i) for i in local
                                 if columns[row] >> i & 1) & 1
                    signature |= value << q
                counts[signature] += 1
            pairs = sum(collision_floor(count, remaining_bins) for count in counts.values())
            worst_pairs = pairs if worst_pairs is None else min(worst_pairs, pairs)
            max_labels = max(max_labels, len(counts) * remaining_bins)
            settings.append(dict(setting=setting, signature_occupancies=dict(sorted(counts.items())),
                                 unavoidable_equal_pairs=pairs))
            if not pairs:
                break
        if worst_pairs:
            return dict(
                suffix_length=length, suffix_dimension=len(local), offset=shift,
                local_profile=local, frozen=checks, prefix_checks=prefix,
                prefix_coordinates=edges, prefix_assignments=1 << len(edges),
                exhaustive_prefix_settings=settings,
                remaining_checks=len(checks) - size,
                remaining_labels_per_prefix_signature=remaining_bins,
                distinct_signature_upper_bound=min(length, max_labels),
                minimum_equal_column_pairs=worst_pairs, repetitions=n // length,
                distance_upper_bound=2 * (n // length),
                A_at_upper_bound_lower_bound=worst_pairs,
                other_coefficients="Unrestricted; bounded only by remaining check-label capacity",
            )
    return None
