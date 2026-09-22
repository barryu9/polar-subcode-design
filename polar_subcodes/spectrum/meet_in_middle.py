"""Exact small-radius coset counts using disjoint subset joins."""
from collections import Counter, defaultdict
from itertools import combinations
from math import comb

from polar_subcodes.algebra import polar_transform
from polar_subcodes.algebra import solve


def small_error_counts(columns, syndrome, radius):
    """Count each support once, including repeated or zero check columns.

    A size-r support has binom(r, floor(r/2)) ordered disjoint partitions.
    Store the two halves by syndrome, join, and divide by this multiplicity.
    """
    if not 0 <= radius <= 6 or syndrome < 0 or any(c < 0 for c in columns):
        raise ValueError('Expected nonnegative syndromes and radius at most six')
    tables = []
    for size in range((radius+1)//2+1):
        table = defaultdict(list)
        for positions in combinations(range(len(columns)), size):
            label = mask = 0
            for i in positions:
                label ^= columns[i]
                mask |= 1 << i
            table[label].append(mask)
        tables.append(table)
    counts = []
    for weight in range(radius+1):
        first = weight//2
        second = weight-first
        count = sum(not (a & b)
                    for label, masks in tables[first].items()
                    for a in masks for b in tables[second].get(label ^ syndrome, ()))
        multiplicity = comb(weight, first)
        if count % multiplicity:
            raise AssertionError('Subset join multiplicity mismatch')
        counts.append(count//multiplicity)
    return counts


def inspect_coset(model, left_input, cutoff):
    a = polar_transform(left_input, model.half)
    weight = a.bit_count()
    if not weight or weight > cutoff:
        raise ValueError('Expected a nonzero critical left word')
    inside = [t for t in range(model.half) if a >> t & 1]
    outside = [t for t in range(model.half) if not a >> t & 1]
    _, annihilator = solve([(model.check_columns[t], 0) for t in inside],
                           model.redundancy)

    def project(label):
        return sum(((label & row).bit_count() & 1) << q
                   for q, row in enumerate(annihilator))

    cross_input = sum(((column & left_input).bit_count() & 1) << q
                      for q, column in enumerate(model.columns[model.half:]))
    cross_word = polar_transform(cross_input, model.half)
    labels = [project(model.check_columns[t]) for t in outside]
    syndrome = 0
    for t, label in zip(outside, labels):
        if cross_word >> t & 1:
            syndrome ^= label
    kernel_dimension = weight-model.redundancy+len(annihilator)
    if kernel_dimension < 0:
        raise AssertionError('Negative shortening dimension')
    radius = (cutoff-weight)//2
    counts = small_error_counts(labels, syndrome, radius)
    return dict(left_input=left_input, left_weight=weight,
                radius=radius, projected_rank=len(annihilator),
                shortening_dimension=kernel_dimension,
                spectrum={weight+2*r: count << kernel_dimension
                          for r, count in enumerate(counts) if count})


def relative_spectrum(model, left_words, right_histogram, cutoff):
    right = {2*int(w): count for w, count in right_histogram.items()
             if 2*int(w) <= cutoff}
    mixed, details = Counter(), []
    for u, weight in left_words:
        if 0 < weight <= cutoff:
            item = inspect_coset(model, u, cutoff)
            if weight != item['left_weight']:
                raise AssertionError('Incorrect left word weight')
            details.append(item)
            mixed.update(item['spectrum'])
    total = Counter(right)
    total.update(mixed)
    return dict(spectrum=dict(sorted(total.items())),
                right_only_spectrum=right, cross_spectrum=dict(sorted(mixed.items())),
                details=details, cutoff=cutoff,
                exact_given_complete_constituent_catalogues=True)
