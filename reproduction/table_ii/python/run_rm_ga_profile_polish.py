"""Improve reliability locally from RM-GA profiles with exact distance checks.

The pilot retains a verified right constituent during left-profile polishing.
If an RM-GA seed is unverified, a separate bootstrap repairs a short suffix
and tests allocation edits in its parent. Scoped UNSAT is never promoted to
full-profile infeasibility. Reference coefficients are not read.
"""

import json
import math
from types import SimpleNamespace
from run_profile_only_targets import restricted_outer, rm_gap_membership
from run_reference_domain_sweep import write
from select_structured_domain import initial_structure, complete_domain
from verify_causal_flag_theory import polar_transform, rref

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def reliability_cost(profile, errors):
    """Additive GA channel cost, not a prediction of precoded SCL FER."""
    return -sum(math.log1p(-errors[i]) for i in profile)

def swaps(profile, errors, removable, addable, *, same_weight=False,
          minimum_row_weight=1, improvement_only=False):
    original = set(profile)
    baseline = reliability_cost(profile, errors)
    records = []
    for old in sorted(set(removable) & original):
        for new in sorted(set(addable) - original):
            if (1 << new.bit_count()) < minimum_row_weight:
                continue
            if same_weight and old.bit_count() != new.bit_count():
                continue
            candidate = sorted(original - {old} | {new})
            cost = reliability_cost(candidate, errors)
            if improvement_only and cost >= baseline - 1e-14:
                continue
            records.append(dict(profile=candidate, removed=old, added=new,
                GA_cost=cost, GA_cost_change=cost-baseline,
                GA_proxy=-math.expm1(-cost),
                removed_row_weight=1 << old.bit_count(),
                added_row_weight=1 << new.bit_count()))
    return sorted(records, key=lambda item: (item["GA_cost_change"], item["removed"], item["added"]))

def canonical_check(n, profile, columns, target):
    mask = sum(1 << i for i in profile)
    if len(columns) != n or any(c < 0 or c >> j or c & ~mask or
                               (j in profile and c) for j, c in enumerate(columns)):
        raise AssertionError("Noncanonical causal support")
    inputs = [(1 << i) | sum(((c >> i) & 1) << j for j, c in enumerate(columns))
              for i in profile]
    generators = [polar_transform(u, n) for u in inputs]
    if len(rref(generators)) != len(profile):
        raise AssertionError("Generator rank mismatch")
    witness = next((dict(source=i, input=u, codeword=x, weight=target)
                    for i, u, x in zip(profile, inputs, generators)
                    if x.bit_count() == target), None)
    return dict(causality=True, generator_rank=len(profile), witness=witness,
                dmin=target if witness else None, distance_lower_bound=target)

class ProtectedCompletionFailure(Exception):
    """Failure in a fixed protected domain, never an unrestricted exclusion."""

    def __init__(self, outcome):
        self.outcome = outcome
        super().__init__(outcome["status"])

def protected_recursive(n, profile, target, folder, args):
    """Carry actual row-threshold protection through every recursive level."""
    if 0 not in profile:
        target += target % 2
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "profile.json", dict(N=n, K=len(profile), profile=profile, target=target))
    baseline = min(1 << i.bit_count() for i in profile)
    if baseline >= target:
        columns = [0]*n
        write(folder / "verification.json", dict(distance_lower_bound=baseline,
            method="Exact unprecoded polar row-weight distance"))
        return columns
    if n <= 32:
        structure = initial_structure(n, profile, target)
        config = SimpleNamespace(work=args.work, seconds=args.seconds, nodes=200000)
        columns, outcome = complete_domain(n, profile, [0]*n, structure["rows"],
            structure["coordinates"], target, folder / "initial", config)
        if columns is None:
            raise ProtectedCompletionFailure(dict(status="short_initial_domain_"+outcome["status"],
                failure_N=n, target=target, nodes=outcome["nodes"],
                scope="Retained-core initial domain of this short profile"))
        write(folder / "support.json", dict(N=n, K=len(profile), profile=profile, columns=columns))
        return columns
    h = n//2
    right_profile = tuple(i-h for i in profile if i >= h)
    child = protected_recursive(h, right_profile, (target+1)//2, folder / f"right{h}", args)
    structure = initial_structure(n, profile, target)
    # A feasible short child does not justify freezing the parent's left
    # internal support. Test the full protected initial block before its
    # outer-only subset; this also captures causal checks in the left half.
    if n <= 128 and 1 <= len(structure["rows"]) <= 8:
        config = SimpleNamespace(work=args.work, seconds=args.seconds, nodes=50000)
        direct, outcome = complete_domain(n, profile, [0]*n, structure["rows"],
            structure["coordinates"], target, folder / "initial", config)
        if direct is not None:
            gap = rm_gap_membership(profile, direct, target)
            proof = canonical_check(n, profile, direct, target)
            proof.update(gap=gap, verification="Complete direct spectrum in the protected initial domain")
            write(folder / "verification.json", proof)
            write(folder / "support.json", dict(N=n, K=len(profile), profile=profile,
                columns=direct, **proof, initialization="all_zero", reference_support_read=False))
            return direct
        if outcome["status"] == "unsat":
            raise ProtectedCompletionFailure(dict(status="protected_initial_domain_unsat",
                failure_N=n, target=target, nodes=outcome["nodes"],
                scope="All listed initial coordinates, including internal left coefficients"))
    embedded = [0]*h + [c << h for c in child]
    gap = rm_gap_membership(profile, embedded, target)
    if any((1 << j.bit_count()) < baseline for j in structure["rows"]):
        raise ValueError("Ancestor row protection was lost")
    write(folder / "ancestor_gap.json", gap)
    write(folder / "fixed_constituents.json", dict(N=n, K=len(profile), profile=profile,
                                                  columns=embedded))
    config = SimpleNamespace(profile=folder / "fixed_constituents.json", output=folder / "outer",
        distance=target, cutoff=baseline, work=args.work, seconds=args.seconds, nodes=200000)
    restricted_outer(config)
    result = read(folder / "outer/summary.json")
    path = folder / "outer/support.json"
    if not path.exists():
        raise ProtectedCompletionFailure(dict(status="protected_outer_"+result["status"],
            failure_N=n, target=target, nodes=result["nodes"],
            scope="This fixed constituent pair and protected outer coefficient domain"))
    columns = read(path)["columns"]
    actual_gap = rm_gap_membership(profile, columns, target)
    if read(folder / "outer/relative_verified.json")["spectrum"] != {"0": 1}:
        raise AssertionError("The protected minimum layer survived")
    proof = canonical_check(n, profile, columns, target)
    proof.update(gap=actual_gap, verification="Complete minimum-layer spectrum and actual RM spectral gap",
                 lower_window_spectrum={0: 1}, recursive_right_lower_bound=(target+1)//2)
    write(folder / "verification.json", proof)
    write(folder / "support.json", dict(N=n, K=len(profile), profile=profile, columns=columns,
        **proof, initialization="all_zero", reference_support_read=False))
    return columns
