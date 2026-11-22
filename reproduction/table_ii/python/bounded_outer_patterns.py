"""Complete outer coset conditions projected onto a small released-row domain."""

from collections import Counter
from types import SimpleNamespace
import numpy as np
from compressed_source_search import SourcePatternProblem
from plotkin_relative_structure import RelativePlotkinSpectrum
from radius_three_completion import small_error_labels
from support_graph_policy import Design
from small_code_words import all_words

def syndrome_ball(columns, rank, radius):
    """Exact membership in a binary syndrome ball; repeated columns may cancel.

    Joining balls of radii a and b gives exactly the ball of radius a+b:
    overlapping error coordinates cancel and cannot increase the weight.
    """
    if not 0 <= rank <= 22 or not 0 <= radius <= 6:
        raise ValueError('Dense syndrome balls require rank <=22 and radius <=6')
    if any(value < 0 or value >> rank for value in columns):
        raise ValueError('Syndrome column exceeds its rank')
    columns = sorted(set(columns)-{0})
    first = small_error_labels(columns, radius//2)
    second = small_error_labels(columns, radius-radius//2)
    ball = np.zeros(1 << rank, dtype=np.bool_)
    if len(first)*len(second) <= radius*max(1, len(columns))*(1 << rank):
        for label in first:
            ball[second ^ label] = True
    else:
        ball[0] = True
        indices = np.arange(1 << rank, dtype=np.uint32)
        for _ in range(radius):
            previous = ball
            ball = previous.copy()
            for label in columns:
                ball |= previous[indices ^ label]
            if np.array_equal(ball, previous):
                break
    return ball

def outer_pattern_problem(profile, columns, rows, coordinates, target, emit=lambda event: None):
    model = RelativePlotkinSpectrum(profile, columns)
    if len(model.left_profile) > 20 or len(rows) > 12:
        raise ValueError('Explicit left catalogue or dense pattern width exceeds its limit')
    if any(j < model.half or j in profile for j in rows):
        raise ValueError('Released rows must be frozen positions in the right half')
    sources = tuple(sorted({i for _, i in coordinates}, reverse=True))
    if any(i not in model.left_profile for i in sources):
        raise ValueError('Both constituent supports must remain fixed')
    if any(columns[j] & ((1 << model.half)-1) for j in range(model.half, model.n)):
        raise ValueError('This pattern model expects initially zero outer support')
    checks = tuple(j for j in range(model.half) if j not in model.right_profile)
    selected = [checks.index(j-model.half) for j in rows]
    groups = Counter()
    left = all_words(model.left_profile, model.left_columns)
    critical = [(u, w) for u, w in left if 0 < w < target]
    for index, (u, _) in enumerate(critical):
        item = model.constraint(u, target-1, materialize=False, max_radius=6)
        ball = syndrome_ball(item['error_columns'], item['rank'], item['radius'])
        images = [item['syndrome_map'][q] for q in selected]
        labels = np.zeros(1 << len(rows), dtype=np.uint32)
        for value in range(1, len(labels)):
            bit = value & -value
            labels[value] = labels[value ^ bit] ^ images[bit.bit_length()-1]
        scope = sum(((u >> i) & 1) << q for q, i in enumerate(sources))
        for value in np.flatnonzero(ball[labels]):
            groups[scope, int(value)] += 1
        if index % 16 == 0 or index+1 == len(critical):
            emit(dict(kind='coset_prepared', index=index+1, total=len(critical),
                      rank=item['rank'], radius=item['radius']))
    catalogue = SimpleNamespace(design=Design(model.n, profile), columns=list(columns),
                                rows=tuple(rows), domain=tuple(coordinates))
    problem = SourcePatternProblem(catalogue, projected_groups=groups, max_width=12)
    return problem, left
