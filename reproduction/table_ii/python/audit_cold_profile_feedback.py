"""Independent scalar spectrum audit of constructed cold-start candidates."""

from bounded_outer_patterns import syndrome_ball
from obstruction_support_search import enumerate_words
from plotkin_relative_structure import RelativePlotkinSpectrum, small_error_counts
from verify_causal_flag_theory import polar_transform

def local_directory(n, profile, columns, cutoff, work):
    if n >= 8:
        return enumerate_words(n, profile, columns, min(cutoff, n), work)
    words, histogram = [], {0: 1}
    for message in range(1, 1 << len(profile)):
        u = sum(((message >> q) & 1) << i for q, i in enumerate(profile))
        u |= sum(((c & u).bit_count() & 1) << j for j, c in enumerate(columns))
        weight = polar_transform(u, n).bit_count()
        if weight <= cutoff:
            words.append((u, weight))
            histogram[weight] = histogram.get(weight, 0) + 1
    return words, dict(exact=True, spectrum=histogram, method="exhaustive_tiny_messages")

def forbidden_syndrome(columns, syndrome, rank, radius):
    """Test exact ball membership without a dense rank-sized table when possible."""
    if radius <= 4:
        return any(small_error_counts(columns, syndrome, radius))
    return bool(syndrome_ball(columns, rank, radius)[syndrome])

def relative_distance_check(n, profile, columns, target, work):
    """Check actual completed cosets, separately from source-pattern search."""
    if not profile:
        return dict(passed=True, method="zero_dimension", target=target)
    if not any(columns) and min(1 << i.bit_count() for i in profile) >= target:
        return dict(passed=True, method="static_polar_distance", target=target)
    if n < 8:
        words, meta = local_directory(n, profile, columns, target - 1, work)
        return dict(passed=not words, method="complete_words", metadata=meta)
    model = RelativePlotkinSpectrum(profile, columns)
    right = relative_distance_check(n // 2, model.right_profile, model.right_columns,
                                    (target + 1) // 2, work)
    if not right["passed"]:
        return dict(passed=False, reason="right_constituent", right=right)
    left, meta = local_directory(n // 2, model.left_profile, model.left_columns,
                                target - 1, work) if model.left_profile else ([], {})
    checks = tuple(j for j in range(n // 2) if j not in model.right_profile)
    separated = 0
    ranks, radii = {}, {}
    for u, weight in left:
        item = model.constraint(u, target - 1, materialize=False, max_radius=6)
        label = 0
        for q, j in enumerate(checks):
            if (columns[n // 2 + j] & u).bit_count() & 1:
                label ^= item["syndrome_map"][q]
        if forbidden_syndrome(item["error_columns"], label, item["rank"], item["radius"]):
            return dict(passed=False, reason="unseparated_coset", left_input=u,
                        left_weight=weight, rank=item["rank"], radius=item["radius"])
        separated += 1
        ranks[item["rank"]] = ranks.get(item["rank"], 0) + 1
        radii[item["radius"]] = radii.get(item["radius"], 0) + 1
    return dict(passed=True, method="complete_relative_coset_membership", target=target,
                right=right, left_directory=meta, separated_cosets=separated,
                quotient_ranks=ranks, radii=radii,
                proof="All critical left words are separated; heavier left projections are protected by the Plotkin weight inequality")
