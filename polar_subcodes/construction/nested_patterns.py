"""Nested projected exclusions when all released coefficients cross one split."""
from collections import Counter
from types import SimpleNamespace
from polar_subcodes.construction.source_patterns import SourcePatternProblem
from polar_subcodes.construction.domains import initial_structure
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.algebra import all_words
from polar_subcodes.code import Design

def right_problem(profile, target, n=64):
    half = n//2
    structure = initial_structure(n, profile, target)
    rows, coordinates = structure['rows'], structure['coordinates']
    if any(j < half for j in rows):
        raise ValueError('This outer-domain reduction requires released rows in the right half')
    sources = tuple(sorted({i for _, i in coordinates}, reverse=True))
    if any(i >= half for i in sources):
        raise ValueError('This reduction preserves both inner constituents')
    model = RelativePlotkinSpectrum(profile, [0]*n)
    if max(len(model.left_profile),len(model.right_profile))>20:
        raise ValueError('Nested reference directories require constituent dimensions <=20')
    right = Counter(w for _, w in all_words(model.right_profile, model.right_columns))
    if any(count for w, count in right.items() if 0 < 2*w < target):
        raise ValueError('The fixed inner right code cannot meet the target')
    checks = tuple(j for j in range(half) if j not in model.right_profile)
    selected = [checks.index(j-half) for j in rows]
    groups = Counter()
    critical = 0
    for u, weight in all_words(model.left_profile, model.left_columns):
        if not 0 < weight < target:
            continue
        critical += 1
        item = model.constraint(u, target-1)
        images = [item['syndrome_map'][q] for q in selected]
        labels = [0]
        scope = sum(((u >> i) & 1) << q for q, i in enumerate(sources))
        for value in range(1 << len(rows)):
            if value:
                bit = value & -value
                labels.append(labels[value ^ bit] ^ images[bit.bit_length()-1])
            if labels[value] in item['forbidden']:
                groups[scope, value] += 1
    catalogue = SimpleNamespace(design=Design(n, profile), columns=[0]*n,
                                rows=rows, domain=coordinates)
    problem = SourcePatternProblem(catalogue, projected_groups=groups, max_width=12)
    return problem, dict(structure=structure, critical_left_cosets=critical,
        fixed_inner_right_spectrum=dict(right), statistics=problem.statistics(),
        multiplicity_semantics='Number of violated coset events, not codeword multiplicity')
