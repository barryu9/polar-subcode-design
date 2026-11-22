"""Audit fixed profiles with support-independent causal distance bounds.

Archive snapshots are immutable. Passing every implemented necessary test
does not prove support existence. Unfinished error-ball tests never reject.
"""

import time
from audit_candidate_profile_distance import CODETABLE_64_52
from audit_ga_profile_causal_ceiling import causal_distance_bounds
from audit_reference_distance_headroom import suffix_bounds
from audit_static_prefix_suffix_obstruction import bound as static_prefix_bound
from audit_causal_prefix_signature import prefix_bound
from audit_hamming_interface_bound import structural_bound
from profile_distance_constraints import profile_constraint_audit, full_group_profile_audit
from profile_group_capacity import group_capacity
from profile_linear_bounds import structural_profile_audit, even_complement_sidon_obstruction

def error_ball_distance_upper(radius, repetitions):
    """Failure of injectivity for errors through t implies local d <= 2t."""
    return 2 * radius * repetitions

def audit(n, information, target,use_table=True):
    started = time.perf_counter()
    information = sorted(information)
    basic = causal_distance_bounds(n, information)
    suffixes = suffix_bounds(n, information)
    necessary = profile_constraint_audit(n, information, target,use_table=use_table)
    structural = structural_profile_audit(n, information, target)
    full_ball = full_group_profile_audit(n, information, target,use_table=use_table)
    evidence = [dict(method="suffix_packing", distance_upper_bound=b["full_upper_bound"],
                     suffix_length=b["length"], suffix_dimension=b["dimension"],
                     local_distance_upper_bound=b["local_upper_bound"]) for b in suffixes]

    def add(method, upper, **details):
        evidence.append(dict(method=method, distance_upper_bound=upper, **details))

    if basic["immutable_row_distance_upper_bound"] is not None:
        add("immutable_row", basic["immutable_row_distance_upper_bound"],
            indices=basic["immutable_minimum_row_indices"])
    if not all(structural[k] for k in ("immutable_row_pass", "coset_capacity_pass",
                                      "complement_capacity_pass")):
        add("profile_structural_capacity", target - 1, details=structural)

    wider = []
    for b in necessary:
        length, repeats, k = b["length"], b["repetitions"], b["information_count"]
        common = dict(suffix_length=length, suffix_dimension=k)
        local = [i - b["start"] for i in information if i >= b["start"]]
        sidon = even_complement_sidon_obstruction(length, local, b["effective_distance"])
        if sidon is not None:
            add("even_complement_sidon_moments", repeats * sidon["local_distance_upper_bound"],
                details=sidon, **common)
        if not b["distance_capacity_pass"]:
            add("suffix_dimension_capacity", repeats * (b["required_distance"] - 1), **common)
        if not b["parity_capacity_pass"]:
            add("suffix_parity_capacity", repeats * (b["effective_distance"] - 1), **common)
        if not b["single_check_pass"]:
            add("single_check_zero_column", repeats, **common)
        if not b["column_capacity_pass"]:
            add("static_column_capacity", 2 * repeats, **common)
        if not b["group_capacity_pass"]:
            radius = b["group_capacity"]["tested_error_radius"]
            add("early_error_ball_capacity", error_ball_distance_upper(radius, repeats),
                tested_error_radius=radius, **common)
        if length >= 2:
            fixed = static_prefix_bound(n, information, length)
            if fixed is not None:
                add("static_prefix_collision", fixed["distance_upper_bound"], details=fixed)
            if 2 * repeats < target:
                fixed = prefix_bound(n, information, length, max_prefix_coefficients=12)
                if fixed is not None:
                    add("causal_prefix_collision", fixed["distance_upper_bound"], details=fixed)
            if length >= 4:
                interface = structural_bound(n, information, length)
                if interface is not None:
                    add("hamming_interface", interface["full_upper_bound"], details=interface)
        if use_table and length == 64 and k >= 52:
            local_upper = 4 if b["even_subcode"] else 5
            add("binary_linear_code_table", repeats * local_upper,
                source=CODETABLE_64_52, **common)
        if not k or b["required_distance"] < 3:
            continue
        radius = min(2, (b["effective_distance"] - 1) // 2)
        for width in sorted({min(length, 16), min(length, 32)}):
            if width <= b["group_capacity"]["early_width"]:
                continue
            pattern = sum(1 << i for i in local if i < width)
            result = group_capacity(length, width, pattern, radius)
            passed = k <= result["max_information"]
            wider.append(dict(**common, pass_required=passed, details=result))
            if not passed:
                add("widened_causal_error_ball_capacity",
                    error_ball_distance_upper(radius, repeats), tested_error_radius=radius,
                    early_width=width, **common)

    for b in full_ball["conditions"]:
        if b["pass_required"] is False:
            radius = b["details"]["required_radius"]
            add("full_required_error_ball_capacity",
                error_ball_distance_upper(radius, n // b["length"]),
                suffix_length=b["length"], suffix_dimension=b["information_count"],
                tested_error_radius=radius)
    upper = min(b["distance_upper_bound"] for b in evidence)
    if 0 not in information:
        upper -= upper % 2
    lower = basic["unprecoded_dmin"]
    if upper < lower:
        raise AssertionError("Support-independent upper bound below the static-code distance")
    limit = min(b["distance_upper_bound"] for b in evidence)
    return dict(N=n, K=len(information), profile=information, target=target, dp=lower,
                external_code_table_enabled=use_table,
                all_support_distance_upper_bound=upper,
                exact_distance_for_every_support=lower if lower == upper else None,
                target_status="excluded" if upper < target else "not_excluded_not_constructed",
                necessary_condition_audit=necessary, structural_audit=structural,
                full_error_ball_audit=full_ball, widened_grouping_audit=wider,
                all_bounds=evidence,
                limiting_bounds=[b for b in evidence if b["distance_upper_bound"] == limit],
                support_constructed=False, elapsed_seconds=time.perf_counter() - started)
