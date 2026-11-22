"""Audit low-weight multiplicities in a shortened suffix retaining an RM core.

The rank formulas cover cosets of RM(m-3,m) inside RM(m-2,m). The causal
late-source bound applies only while this full core and the affine checks are
preserved. It is not a bound over every expression on the fixed profile.
"""

from collections import Counter
from feasible_design_bank import validate_design
from structural_shared_spectrum import alternating_rows
from verify_causal_flag_theory import rref

def rank_audit(n, profile, columns, max_cosets=1_048_576):
    """Count rank-two/four quotient matrices after checking all RM guards."""
    profile = validate_design(n, profile, columns)
    m = n.bit_length()-1
    if m < 4:
        raise ValueError('Require a length of at least sixteen')
    core = tuple(i for i in range(n) if i.bit_count() >= 3)
    core_mask = sum(1 << i for i in core)
    weak = tuple(i for i in profile if i.bit_count() == 2)
    missing = sorted(set(core)-set(profile))
    affine_information = [i for i in profile if i.bit_count() < 2]
    affine_outputs = [j for j, c in enumerate(columns) if j.bit_count() < 2 and c]
    core_outputs = [j for j, c in enumerate(columns) if c & core_mask]
    core_constraint_rank = len(rref(c & core_mask for c in columns))
    guards = dict(full_core_information=not missing,
                  no_affine_information=not affine_information,
                  affine_frozen_expressions_zero=not affine_outputs,
                  core_sources_not_used=not core_outputs)
    result = dict(N=n, K=len(profile), m=m, core_dimension=len(core),
                  weak_information=list(weak), guards=guards,
                  applicable=all(guards.values()), missing_core=missing,
                  affine_information=affine_information,
                  nonzero_affine_expression_rows=affine_outputs,
                  core_source_expression_rows=core_outputs,
                  core_source_coefficient_rank=core_constraint_rank,
                  intersection_with_natural_RM_core_dimension=(
                      len(core)-core_constraint_rank if not missing else None),
                  scope='Full retained RM core and zero affine moments',
                  full_causal_domain_lower_bound_claim=False)
    if not result['applicable']:
        return result
    if (1 << len(weak)) > max_cosets:
        raise ValueError('The quotient exceeds the explicit coset limit')

    generators = [(1 << i) | sum(((c >> i) & 1) << j
                                  for j, c in enumerate(columns)) for i in weak]
    inputs = [0]
    for generator in generators:
        inputs += [u ^ generator for u in inputs]
    ranks = [len(rref(alternating_rows(u, m))) for u in inputs]
    histogram = Counter(ranks)
    if histogram[0] != 1 or any(rank % 2 for rank in ranks):
        raise AssertionError('Invalid quotient dimension or alternating rank')
    predicted = {0: 1, 4: (n//4)*histogram[2], 6: n*histogram[4]}

    # Late causal degree-two outputs all touch one of the last two axes.
    late = tuple(i for i in weak if i >= n//4)
    early_bits = sum(1 << k for k, i in enumerate(weak) if i not in late)
    late_cosets = [q for q in range(1, len(inputs)) if not q & early_bits]
    late_ranks = Counter(ranks[q] for q in late_cosets)
    if any(rank not in (2, 4) for rank in late_ranks):
        raise AssertionError('Causality contradicts the two-axis rank cover')
    lower = n*((1 << len(late))-1) if not histogram[2] else None
    early_groups = Counter(q & early_bits for q, rank in enumerate(ranks) if rank == 4)
    result.update(cosets=len(inputs), rank_histogram=dict(sorted(histogram.items())),
                  spectrum_through6=predicted, distance_at_least6=not histogram[2],
                  late_information=list(late), late_nonzero_cosets=len(late_cosets),
                  late_rank_histogram=dict(sorted(late_ranks.items())),
                  late_vertex_cover=[m-2, m-1],
                  A6_lower_bound_given_distance_at_least6=lower,
                  rank4_cosets_by_early_message=dict(sorted(early_groups.items())),
                  rank4_cosets_outside_late_subspace=histogram[4]-late_ranks[4],
                  attainability_of_lower_bound_proved=(
                      lower is not None and predicted[6] == lower),
                  conditional_bound_proof=[
                      'The full RM(m-3,m) core is the kernel of moments of degree at most two.',
                      'Weak information assignments identify disjoint cosets of that kernel.',
                      'Each rank-two coset has N/4 weight-four vectors; each rank-four coset has N weight-six vectors.',
                      'With only late information active, causal quadratic outputs touch one of the last two axes.',
                      'The resulting alternating matrices have rank at most four and are nonzero by systematicity.',
                      'Distance at least six excludes rank two, forcing all nonzero late assignments to rank four.',
                      'Hence A6 is at least N times (2^late_dimension-1).',
                      'Shortening a causal polar code to its last L inputs injects each suffix codeword with weight multiplied by full_N/L.'
                  ])
    return result
