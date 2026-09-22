"""Exact compatibility and deterministic lifting of one critical message circuit."""
from collections import Counter
from functools import reduce
from operator import xor

from polar_subcodes.construction.conditional import xor_mask
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.algebra import polar_transform, rref
from polar_subcodes.algebra import solve


def walsh(values):
    result = list(values)
    size = 1
    while size < len(result):
        for start in range(0, len(result), 2*size):
            for i in range(start, start+size):
                a, b = result[i], result[i+size]
                result[i], result[i+size] = a+b, a-b
        size *= 2
    return result


def xor_convolution(first, second):
    if len(first) != len(second) or not first or len(first) & (len(first)-1):
        raise ValueError('Convolution arrays must have the same power-of-two length')
    a, b = walsh(first), walsh(second)
    result = walsh([x*y for x, y in zip(a, b)])
    if any(value % len(first) for value in result):
        raise AssertionError('Nonintegral exact Walsh inverse')
    return [value//len(first) for value in result]


def project(label, rows):
    return sum(((label & row).bit_count() & 1) << q for q, row in enumerate(rows))


def analyze_circuit(profile, columns, left_inputs, cutoff, max_common_rank=18, lift=True):
    """All outer coefficients free; other critical inputs are not included implicitly."""
    model = RelativePlotkinSpectrum(profile, columns)
    mask = sum(1 << i for i in model.left_profile)
    inputs = tuple(left_inputs)
    messages = tuple(u & mask for u in inputs)
    if (len(inputs) < 3 or len(set(messages)) != len(inputs) or 0 in messages
            or reduce(xor, messages, 0) or len(rref(messages)) != len(inputs)-1):
        raise ValueError('Expected one minimally dependent binary message set')
    for u in inputs:
        if any((u & ((1 << j) ^ model.left_columns[j])).bit_count() & 1
               for j in range(model.half) if j not in model.left_profile):
            raise ValueError('A circuit input is outside the left code')
    items = [model.constraint(u, cutoff) for u in inputs]
    union = reduce(lambda a, u: a | polar_transform(u, model.half), inputs, 0)
    _, common = solve([(model.check_columns[j], 0) for j in range(model.half)
                       if union >> j & 1], model.redundancy)
    common_rank = len(common)
    if common_rank > max_common_rank:
        raise ValueError('Common-quotient histogram exceeds the explicit size limit')
    histograms, projections = [], []
    for item in items:
        rows = []
        for common_row in common:
            solution = solve([(value, (common_row >> j) & 1)
                              for j, value in enumerate(item['syndrome_map'])], item['rank'])
            if solution is None or solution[1]:
                raise AssertionError('Common check lacks a unique quotient representation')
            rows.append(solution[0])
        bad = Counter(project(value, rows) for value in item['forbidden'])
        fiber_size = 1 << (item['rank']-common_rank)
        histogram = [fiber_size-bad[z] for z in range(1 << common_rank)]
        if min(histogram) < 0:
            raise AssertionError('A projected forbidden count exceeds the fiber')
        projections.append(rows)
        histograms.append(histogram)
    suffix = [None]*(len(inputs)+1)
    suffix[-1] = [1]+[0]*((1 << common_rank)-1)
    for i in reversed(range(len(inputs))):
        suffix[i] = xor_convolution(histograms[i], suffix[i+1])
    rank = sum(item['rank'] for item in items)-common_rank
    variables = model.redundancy*len(model.left_profile)
    compatible = suffix[0][0]
    result = dict(left_inputs=inputs, message_rank=len(inputs)-1,
        common_rank=common_rank, common_checks=common, quotient_ranks=[item['rank'] for item in items],
        forbidden_sizes=[len(item['forbidden']) for item in items],
        good_fiber_counts=histograms, projections=projections,
        joint_syndrome_rank=rank, outer_dimension=variables,
        compatible_good_syndrome_tuples=compatible,
        successful_outer_matrices=compatible*(1 << (variables-rank)),
        scope='Only the listed circuit constraints; pure-right and other messages require separate checks')
    if not compatible or not lift:
        return result
    targets, residual = [], 0
    for i, item in enumerate(items):
        z = next(z for z, count in enumerate(histograms[i])
                 if count and suffix[i+1][residual ^ z])
        residual ^= z
        origin, basis = solve([(row, (z >> q) & 1) for q, row in enumerate(projections[i])], item['rank'])
        target = next(origin ^ xor_mask(basis, value) for value in range(1 << len(basis))
                      if origin ^ xor_mask(basis, value) not in item['forbidden'])
        targets.append(target)
    if residual:
        raise AssertionError('Projected good syndromes do not close the circuit')
    equations = []
    f = model.redundancy
    for message, item, target in zip(messages, items, targets):
        for bit in range(item['rank']):
            row = sum(((value >> bit) & 1) << q for q, value in enumerate(item['syndrome_map']))
            coefficients = sum(row << (f*j) for j, source in enumerate(model.left_profile)
                               if message >> source & 1)
            equations.append((coefficients, (target >> bit) & 1))
    solution = solve(equations, variables)
    if solution is None:
        raise AssertionError('Compatible circuit signatures did not lift')
    outer = solution[0]
    rows = [j for j in range(model.half) if j not in model.right_profile]
    child = list(columns)
    for q, j in enumerate(rows):
        child[model.half+j] &= ~mask
        child[model.half+j] |= sum(((outer >> (f*t+q)) & 1) << source
                                   for t, source in enumerate(model.left_profile))
    result.update(selected_syndromes=targets, columns=child,
                  free_lift_dimension=len(solution[1]))
    return result
