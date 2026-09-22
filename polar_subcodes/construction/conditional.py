"""Deterministic cross-block completion by conditional expectations.

The two constituent codes are fixed during a completion, but need not be RM
codes. Failure of the expectation bound is not an infeasibility conclusion.
"""
from collections import defaultdict
from fractions import Fraction

from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum


def xor_mask(columns, mask):
    value = 0
    while mask:
        bit = mask & -mask
        value ^= columns[bit.bit_length()-1]
        mask ^= bit
    return value


def suffix_coordinates(columns):
    """Use rightmost independent variables so each pivot depends on earlier bits."""
    pivots, indices = {}, []
    for index in reversed(range(len(columns))):
        value, coordinates = columns[index], 0
        while value:
            p = value.bit_length()-1
            if p not in pivots:
                coordinates ^= 1 << len(indices)
                pivots[p] = value, coordinates
                indices.append(index)
                break
            other, label = pivots[p]
            value, coordinates = value ^ other, coordinates ^ label

    def encode(value):
        label = 0
        while value:
            p = value.bit_length()-1
            if p not in pivots:
                raise ValueError('Syndrome outside the map image')
            other, coordinates = pivots[p]
            value, label = value ^ other, label ^ coordinates
        return label
    return [encode(value) for value in columns], {index: q for q, index in enumerate(indices)}, encode


def complete(profile, columns, left_words, cutoff, forced=None):
    model = RelativePlotkinSpectrum(profile, columns)
    right_frozen = [j for j in range(model.half) if j not in model.right_profile]
    groups = defaultdict(list)
    info_mask = sum(1 << i for i in model.left_profile)
    for u, weight in left_words:
        if 0 < weight <= cutoff:
            message = u & info_mask
            groups[message.bit_length()-1].append((message, model.constraint(u, cutoff)))
    cross, trace, total_bound = {}, [], Fraction(0)
    all_constraints = [item for group in groups.values() for _, item in group]
    for item in all_constraints:
        total_bound += Fraction(len(item['forbidden']), 1 << item['rank'])
    forced = {} if forced is None else dict(forced)
    legal = {(model.half+j, i) for j in right_frozen for i in model.left_profile}
    if not set(forced) <= legal or any(value not in (0, 1) for value in forced.values()):
        raise ValueError('Forced entries must be binary legal cross-block coefficients')
    for source in model.left_profile:
        pending = []
        for message, item in groups[source]:
            offset = 0
            for i, value in cross.items():
                if message >> i & 1:
                    offset ^= value
            coordinates, pivots, encode = suffix_coordinates(item['syndrome_map'])
            labels = [encode(value) for value in item['forbidden']]
            pending.append(dict(columns=coordinates, pivots=pivots, labels=labels,
                                offset=xor_mask(coordinates, offset), remaining=item['rank']))
        value = 0
        start_bound = sum((Fraction(len(item['labels']), 1 << item['remaining'])
                           for item in pending), Fraction(0))
        for index, destination in enumerate(right_frozen):
            branch_bounds = [Fraction(0), Fraction(0)]
            partitions = []
            for item in pending:
                pivot = item['pivots'].get(index)
                if pivot is None:
                    bound = Fraction(len(item['labels']), 1 << item['remaining'])
                    branch_bounds[0] += bound
                    branch_bounds[1] += bound
                    partitions.append(None)
                else:
                    parity = (item['offset'] >> pivot) & 1
                    halves = [[], []]
                    for label in item['labels']:
                        halves[((label >> pivot) & 1) ^ parity].append(label)
                    denominator = 1 << (item['remaining']-1)
                    branch_bounds[0] += Fraction(len(halves[0]), denominator)
                    branch_bounds[1] += Fraction(len(halves[1]), denominator)
                    partitions.append(halves)
            key = (model.half+destination, source)
            chosen = forced.get(key, int(branch_bounds[1] < branch_bounds[0]))
            value |= chosen << index
            for item, halves in zip(pending, partitions):
                if halves is not None:
                    item['labels'] = halves[chosen]
                    item['remaining'] -= 1
                if chosen:
                    item['offset'] ^= item['columns'][index]
        bad = sum(bool(item['labels']) for item in pending)
        trace.append(dict(source=source, critical_cosets=len(pending),
                          initial_expectation=str(start_bound), bad_cosets=bad))
        if bad:
            return dict(status='conditional_bound_not_sufficient', source=source, trace=trace,
                        failure_not_infeasibility=True, union_bound=str(total_bound))
        cross[source] = value
    child = list(columns)
    for index, j in enumerate(right_frozen):
        child[model.half+j] &= ~info_mask
        child[model.half+j] |= sum(((value >> index) & 1) << i for i, value in cross.items())
    return dict(status='completed_cross_block', columns=child, trace=trace,
                union_bound=str(total_bound),
                success_lower_bound=str(max(Fraction(0), 1-total_bound)),
                density_scope='uniform unrestricted cross matrix, before forced entries',
                guaranteed_by_initial_bound=not forced and total_bound < 1,
                independent_constituent_verification_required=True)
