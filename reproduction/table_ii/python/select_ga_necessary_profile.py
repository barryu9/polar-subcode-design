"""Select a GA-priority profile using table-free necessary distance checks.

Prefer each channel in decreasing GA reliability order if a completion is
still possible. Refine the profile constraints with proved local capacity
failures, then repeat. Passing finite checks is not support existence.
"""

import time
import z3
from profile_distance_constraints import suffix_caps, column_capacity_cap
from profile_linear_bounds import linear_dimension_bound, coset_capacities

def base_solver(n, k, target):
    variables = [z3.Bool(f"information_{i}") for i in range(n)]
    solver = z3.Solver()
    solver.set(random_seed=0)
    solver.add(z3.PbEq([(v, 1) for v in variables], k))
    for b in suffix_caps(n, target, use_table=False):
        start, length, distance = b["start"], b["length"], b["required_distance"]
        local = variables[start:]
        count = z3.Sum([z3.If(v, 1, 0) for v in local])
        solver.add(count <= b["max_information"])
        even_distance = distance + distance % 2
        solver.add(z3.Implies(z3.Not(local[0]),
                             count <= linear_dimension_bound(length, even_distance, False)))
        if distance >= 2:
            solver.add(z3.Implies(count == length - 1, z3.Not(local[0])))
        if length in (16, 32, 64) and distance >= 6:
            cap = length - 2*(length.bit_length()-1) - 1
            solver.add(z3.Implies(z3.And(z3.Not(local[0]), local[-1]), count <= cap))
        half = length // 2
        if half >= 4:
            # A forced extended-Hamming right component covers every punctured
            # syndrome within radius one for an immutable nonzero left row.
            right_hamming = z3.And(*[local[half+j] if j.bit_count() >= 2
                                    else z3.Not(local[half+j]) for j in range(half)])
            for i in range(half):
                if ((1 << i.bit_count()) + 2) * (n // length) < target:
                    immutable_left = z3.And(*local[i:half])
                    solver.add(z3.Not(z3.And(right_hamming, immutable_left)))
        caps, complement = coset_capacities(length, distance)
        for i, v in enumerate(local):
            later = local[i + 1:]
            later_count = z3.Sum([z3.If(w, 1, 0) for w in later]) if later else z3.IntVal(0)
            solver.add(z3.Implies(v, later_count <= caps[i]))
            if i < length - 1:
                solver.add(z3.Implies(z3.And(v, local[-1]), later_count <= complement[i]))
            if (1 << i.bit_count()) < distance:
                solver.add(z3.Implies(v, z3.Or([z3.Not(w) for w in later])))
            if distance >= 3 and i > 0:
                first = z3.And(v, *[z3.Not(w) for w in local[:i]])
                solver.add(z3.Implies(first, count <= column_capacity_cap(length, i)))
    return solver, variables

def ga_priority_completion(solver, variables, order, deadline):
    """Find the lexicographically preferred profile in a specified channel order."""
    selected = []
    solver.push()
    try:
        for i in order:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None, "budget_exhausted"
            solver.set(timeout=max(1, int(1000 * remaining)))
            solver.push()
            solver.add(variables[i])
            answer = solver.check()
            solver.pop()
            if answer == z3.unknown:
                return None, "solver_unknown"
            take = answer == z3.sat
            solver.add(variables[i] if take else z3.Not(variables[i]))
            if take:
                selected.append(i)
        if solver.check() != z3.sat:
            return None, "no_profile_in_current_relaxation"
        return sorted(selected), "selected"
    finally:
        solver.pop()

def capacity_cut(solver, variables, profile, start, length, width, cap):
    """A fixed early profile pattern bounds the whole suffix dimension."""
    information = set(profile)
    pattern = z3.And(*[variables[i] if i in information else z3.Not(variables[i])
                       for i in range(start, start + width)])
    count = z3.Sum([z3.If(v, 1, 0) for v in variables[start:]])
    solver.add(z3.Implies(pattern, count <= cap))
    return dict(start=start, length=length, early_width=width, max_information=cap)

def refine(solver, variables, profile, record):
    n = len(variables)
    cuts = []
    for b in record["necessary_condition_audit"]:
        if not b["group_capacity_pass"]:
            g = b["group_capacity"]
            cuts.append(capacity_cut(solver, variables, profile, b["start"], b["length"],
                                     g["early_width"], g["max_information"]))
    for b in record["widened_grouping_audit"]:
        if not b["pass_required"]:
            g = b["details"]
            cuts.append(capacity_cut(solver, variables, profile, n - b["suffix_length"],
                                     b["suffix_length"], g["early_width"], g["max_information"]))
    for b in record["full_error_ball_audit"]["conditions"]:
        if b["pass_required"] is False:
            g = b["details"]
            cuts.append(capacity_cut(solver, variables, profile, n - b["length"],
                                     b["length"], g["early_width"], g["max_information"]))
    if not cuts:
        # The full profile is excluded only after a proved necessary failure.
        information = set(profile)
        solver.add(z3.Or(*[z3.Not(v) if i in information else v
                           for i, v in enumerate(variables)]))
        cuts.append(dict(kind="excluded_profile", profile=profile))
    return cuts
