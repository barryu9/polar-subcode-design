"""Complete finite reference construction for tiny profiles.

This deliberately exponential implementation retains every domain shell and
constituent alternative. Use the production pipeline for the reported designs.
"""
from collections import defaultdict
from fractions import Fraction
import math
from polar_subcodes.construction.conditional import xor_mask
from polar_subcodes.algebra import polar_transform

def span(basis):
    values = {0}
    for v in basis:
        values |= {u ^ v for u in values}
    return values

def profile_rule(n, profile, target):
    baseline = min(1 << i.bit_count() for i in profile)
    rows = tuple(j for j in range(n) if j not in profile and j > min(profile)
                 and (1 << j.bit_count()) >= baseline)
    core = tuple(i for i in profile if (1 << i.bit_count()) >= target)
    edges = tuple((j, i) for j in range(n) if j not in profile for i in profile if i < j)
    first = tuple(q for q, (j, i) in enumerate(edges) if j in rows and i not in core)
    missing_core = tuple(q for q, (j, i) in enumerate(edges)
                         if j in rows and i in core)
    missing_rows = tuple(sorted(
        (q for q, (j, _) in enumerate(edges) if j not in rows),
        key=lambda q: (-(1 << edges[q][0].bit_count()), *edges[q])))
    return baseline, rows, core, edges, first, missing_core+missing_rows

def codewords(n, profile, edges, support):
    basis = []
    for source in profile:
        u = 1 << source
        for q, (j, i) in enumerate(edges):
            if i == source and support >> q & 1:
                u ^= 1 << j
        basis.append(polar_transform(u, n))
    return span(basis)

def word_list(n, checks, cutoff):
    if cutoff < 0:
        return []
    if n == 1:
        return [u for u in (0, 1) if u <= cutoff
                and all(not ((h & u).bit_count() & 1) for h in checks)]
    half = n // 2
    mask = (1 << half)-1
    left = [h for h in checks if h.bit_length() <= half]
    right = [h for h in checks if h.bit_length() > half]
    words = []
    for a in word_list(half, left, cutoff):
        s = polar_transform(a, half)
        inside = [i for i in range(half) if s >> i & 1]
        outside = [i for i in range(half) if not s >> i & 1]
        radius = (cutoff-s.bit_count()) // 2
        for e in range(1 << len(outside)):
            if e.bit_count() > radius:
                continue
            outside_value = sum(((e >> q) & 1) << i for q, i in enumerate(outside))
            for z in range(1 << len(inside)):
                t = outside_value | sum(((z >> q) & 1) << i for q, i in enumerate(inside))
                b = polar_transform(t, half)
                if all(not (((h & mask & a).bit_count()
                             + ((h >> half) & b).bit_count()) & 1) for h in right):
                    words.append(a | (b << half))
    return words

def source_candidates(groups, domains):
    def reduce(assigned, current):
        current = dict(current)
        current.update({i: (v,) for i, v in assigned.items()})
        for q, forbidden in groups.items():
            remaining = [i for i in domains if q >> i & 1 and i not in assigned]
            offset = 0
            for i, value in assigned.items():
                if q >> i & 1:
                    offset ^= value
            if not remaining and offset in forbidden:
                return None
            if len(remaining) == 1:
                i = remaining[0]
                current[i] = tuple(v for v in current[i] if (v ^ offset) not in forbidden)
        return current if all(current.values()) else None

    def visit(assigned, current):
        current = reduce(assigned, current)
        if current is None:
            return
        if len(assigned) == len(domains):
            yield assigned
            return
        i = max(set(domains)-assigned.keys())
        children = []
        for value in current[i]:
            child = dict(assigned)
            child[i] = value
            reduced = reduce(child, current)
            if reduced is not None:
                size = math.prod(len(reduced[k]) for k in domains if k not in child)
                children.append((-size, value.bit_count(), value, child, reduced))
        for _, _, _, child, reduced in sorted(children, key=lambda entry: entry[:3]):
            yield from visit(child, reduced)
    yield from visit({}, domains)

def outer_completion(groups, width, source_count):
    stats = {'failed_columns': 0}

    def image_data(i, previous, bits):
        available = [v for v in range(1 << width)
                     if all((v >> j & 1) == bit for j, bit in enumerate(bits))]
        events = []
        for message, columns, forbidden in groups.get(i, ()):
            offset = xor_mask(previous, message & ((1 << i)-1))
            image = {xor_mask(columns, value ^ offset) for value in available}
            events.append((image, forbidden))
        return events

    def mean(i, previous, bits):
        return sum((Fraction(len(image & bad), len(image))
                    for image, bad in image_data(i, previous, bits)), Fraction(0))

    def complete(i, previous, bits):
        if i == source_count:
            return previous
        if any(image <= bad for image, bad in image_data(i, previous, bits)):
            return None
        if len(bits) == width:
            value = sum(bit << j for j, bit in enumerate(bits))
            answer = complete(i+1, previous+(value,), ())
            if answer is None:
                stats['failed_columns'] += 1
            return answer
        for bit in sorted((0, 1), key=lambda b: (mean(i, previous, bits+(b,)), b)):
            answer = complete(i, previous, bits+(bit,))
            if answer is not None:
                return answer
        return None
    return complete(0, (), ()), stats

def support_stream(n, profile, target):
    if not profile:
        yield 0
        return
    if target > n:
        return
    _, _, _, edges, first, order = profile_rule(n, profile, target)
    domains = [(0, tuple(first))]
    free = list(first)
    for edge in order:
        domains.append((1 << edge, tuple(free)))
        free.append(edge)
    for origin, free in domains:
        rows = sorted({edges[e][0] for e in free})
        row_id = {j: b for b, j in enumerate(rows)}
        columns = [sum(((origin >> e) & 1) << i
                       for e, (j, i) in enumerate(edges) if j == row)
                   for row in range(n)]
        checks = [(1 << j) | columns[j] for j in range(n)
                  if j not in profile and j not in rows]
        patterns = defaultdict(set)
        sources = sorted({edges[e][1] for e in free})
        for u in word_list(n, checks, target-1):
            if not u or (u & -u).bit_length()-1 not in profile:
                continue
            q = sum(u & (1 << i) for i in sources)
            beta = sum((((u & ((1 << j) | columns[j])).bit_count() & 1) << b)
                       for b, j in enumerate(rows))
            patterns[q].add(beta)
        column_domains = {i: tuple(sorted(span([1 << row_id[j]
                          for e, (j, k) in enumerate(edges) if e in free and k == i])))
                          for i in sources}
        for assignment in source_candidates(patterns, column_domains):
            support = origin
            for edge in free:
                j, i = edges[edge]
                support ^= ((assignment[i] >> row_id[j]) & 1) << edge
            yield support

def trial_construct(n, profile, target):
    """Complete split construction with equivalent unprojected outer sets."""
    if not profile:
        return 0
    if target > n:
        return None
    if n == 1:
        return 0
    h = n//2
    left = tuple(i for i in profile if i < h)
    right = tuple(i-h for i in profile if i >= h)

    def edges_of(length, info):
        return tuple((j, i) for j in range(length) if j not in info
                     for i in info if i < j)

    def coefficients(edges, value):
        return {pair: (value >> q) & 1 for q, pair in enumerate(edges)}

    left_edges, right_edges = edges_of(h, left), edges_of(h, right)
    rows = tuple(j for j in range(h) if j not in right)
    for ls in sorted(range(1 << len(left_edges)), key=lambda s: (s.bit_count(), s)):
        lvalues = coefficients(left_edges, ls)
        # A fresh generator is essential for each distinct left constituent.
        for rs in support_stream(h, right, (target+1)//2):
            rvalues = coefficients(right_edges, rs)
            checks = [(1 << j) | sum(value << i for (k, i), value in rvalues.items()
                                     if k == j) for j in rows]
            groups = defaultdict(list)
            for message in range(1, 1 << len(left)):
                u = sum(((message >> q) & 1) << i for q, i in enumerate(left))
                u |= sum((sum(value*((u >> i) & 1)
                              for (k, i), value in lvalues.items() if k == j) % 2) << j
                         for j in range(h) if j not in left)
                a = polar_transform(u, h)
                if a.bit_count() >= target:
                    continue
                bad = set()
                for v in range(1 << h):
                    if (a ^ v).bit_count()+v.bit_count() < target:
                        b = polar_transform(v, h)
                        bad.add(sum(((b & check).bit_count() & 1) << q
                                    for q, check in enumerate(checks)))
                groups[message.bit_length()-1].append(
                    (message, tuple(1 << j for j in range(len(rows))), bad))
            outer, _ = outer_completion(groups, len(rows), len(left))
            if outer is None:
                continue
            values = dict(lvalues)
            values.update({(h+j, h+i): value for (j, i), value in rvalues.items()})
            values.update({(h+j, i): (outer[q] >> t) & 1
                           for q, i in enumerate(left) for t, j in enumerate(rows)})
            edges = edges_of(n, profile)
            result = sum(values[pair] << q for q, pair in enumerate(edges))
            assert all(not w or w.bit_count() >= target
                       for w in codewords(n, profile, edges, result))
            return result
    return None
