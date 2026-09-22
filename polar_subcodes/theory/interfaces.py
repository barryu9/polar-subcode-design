"""Exact outer completion through shared linear-check interfaces.

This research implementation counts only the supplied exclusion constraints.
It uses standard tree-based elimination, with polar-specific constraint rows
built by ``polar_outer_events``. Large interfaces are rejected, not truncated.
"""
from collections import Counter
from dataclasses import dataclass, field

from polar_subcodes.construction.conditional import xor_mask
from polar_subcodes.theory.circuits import project
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.algebra import solve


def echelon(rows):
    pivots = {}
    for row in rows:
        while row:
            pivot = row.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    return tuple(pivots[pivot] for pivot in sorted(pivots, reverse=True))


def coordinate_solver(rows):
    pivots = {}
    for i, row in enumerate(rows):
        tag = 1 << i
        while row:
            pivot = row.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = row, tag
                break
            other, label = pivots[pivot]
            row ^= other
            tag ^= label

    def coordinates(row):
        tag = 0
        while row:
            pivot = row.bit_length()-1
            if pivot not in pivots:
                raise ValueError('Vector is not in the specified row space')
            other, label = pivots[pivot]
            row ^= other
            tag ^= label
        return tag

    return coordinates


def intersection(first, second):
    pivots = {row.bit_length()-1: (row, row) for row in echelon(first)}
    common = []
    for row in second:
        tag = 0
        while row:
            pivot = row.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = row, tag
                break
            other, label = pivots[pivot]
            row ^= other
            tag ^= label
        if not row and tag:
            common.append(tag)
    return echelon(common)


def normalize_event(event, dimension):
    """Remove unreachable forbidden outputs after an affine restriction."""
    rows = tuple(event['rows'])
    offset = event.get('offset', 0)
    bad = set(event['forbidden'])
    if (dimension < 0 or any(row < 0 or row >> dimension for row in rows)
            or offset < 0 or offset >> len(rows)
            or any(value < 0 or value >> len(rows) for value in bad)):
        raise ValueError('Constraint dimensions or labels are invalid')
    pivots, basis, positions = {}, [], []
    for i, row in enumerate(rows):
        residual = row
        while residual:
            pivot = residual.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = residual
                basis.append(row)
                positions.append(i)
                break
            residual ^= pivots[pivot]
    coordinates = coordinate_solver(basis)
    output_map = [coordinates(row) for row in rows]
    reachable_bad = set()
    for value in bad:
        shifted = value ^ offset
        label = sum(((shifted >> i) & 1) << q for q, i in enumerate(positions))
        if project(label, output_map) == shifted:
            reachable_bad.add(label)
    return dict(rows=tuple(basis), forbidden=reachable_bad)


@dataclass
class Node:
    indices: tuple
    rows: tuple
    children: tuple = ()
    separator: tuple = ()
    counts: list = field(default_factory=list)
    witnesses: dict = field(default_factory=dict)


def balanced_tree(indices):
    indices = tuple(indices)
    if not indices:
        raise ValueError('A tree needs at least one leaf')
    if len(indices) == 1:
        return indices[0]
    middle = len(indices)//2
    return balanced_tree(indices[:middle]), balanced_tree(indices[middle:])


def prepare_interfaces(events, dimension, tree=None):
    normalized = [normalize_event(event, dimension) for event in events]
    if not normalized:
        return normalized, None, []
    if tree is None:
        tree = balanced_tree(range(len(events)))
    nodes = []

    def build(part):
        if isinstance(part, int):
            if not 0 <= part < len(events):
                raise ValueError('Unknown event index in tree')
            node = Node((part,), echelon(normalized[part]['rows']))
        else:
            if len(part) != 2:
                raise ValueError('An internal tree node needs two children')
            left, right = (build(child) for child in part)
            node = Node(left.indices+right.indices, echelon(left.rows+right.rows), (left, right))
        nodes.append(node)
        return node

    root = build(tree)
    if sorted(root.indices) != list(range(len(events))):
        raise ValueError('The tree must contain every event exactly once')

    def boundary(node, outside):
        node.separator = intersection(node.rows, outside)
        if node.children:
            left, right = node.children
            boundary(left, echelon(outside+right.rows))
            boundary(right, echelon(outside+left.rows))

    boundary(root, ())
    return normalized, root, nodes


def count_interfaces(events, dimension, tree=None, max_width=16, lift=True):
    """Return an exact count and a deterministic witness in free parameters."""
    normalized, root, nodes = prepare_interfaces(events, dimension, tree)
    if root is None:
        return dict(successful_parameters=1 << dimension, parameter=0,
                    joint_rank=0, max_interface_rank=0, merge_pair_visits=0)
    width = max(len(node.separator) for node in nodes)
    if width > max_width:
        raise ValueError(f'Interface rank {width} exceeds explicit limit {max_width}')
    pair_visits = 0
    for node in nodes:
        c = len(node.separator)
        if not node.children:
            event = normalized[node.indices[0]]
            coordinates = coordinate_solver(event['rows'])
            projection = [coordinates(row) for row in node.separator]
            histogram = Counter(project(value, projection) for value in event['forbidden'])
            fiber = 1 << (len(event['rows'])-c)
            node.counts = [fiber-histogram[z] for z in range(1 << c)]
            continue
        left, right = node.children
        common = intersection(left.rows, right.rows)
        left_coordinates = coordinate_solver(left.separator)
        right_coordinates = coordinate_solver(right.separator)
        left_match = [left_coordinates(row) for row in common]
        right_match = [right_coordinates(row) for row in common]
        both_coordinates = coordinate_solver(left.separator+right.separator)
        parent_projection = [both_coordinates(row) for row in node.separator]
        node.counts = [0]*(1 << c)
        right_by_match = {}
        for y, count in enumerate(right.counts):
            if count:
                right_by_match.setdefault(project(y, right_match), []).append((y, count))
        for x, count_x in enumerate(left.counts):
            if not count_x:
                continue
            for y, count_y in right_by_match.get(project(x, left_match), ()):
                pair_visits += 1
                label = project(x | (y << len(left.separator)), parent_projection)
                node.counts[label] += count_x*count_y
                if lift:
                    node.witnesses.setdefault(label, (x, y))
    result = dict(successful_parameters=root.counts[0] << (dimension-len(root.rows)),
                  compatible_joint_outputs=root.counts[0], joint_rank=len(root.rows),
                  max_interface_rank=width, merge_pair_visits=pair_visits,
                  interface_ranks=[dict(events=node.indices, rank=len(node.separator)) for node in nodes])
    if not lift or not root.counts[0]:
        return result
    equations = []

    def recover(node, label):
        if node.children:
            x, y = node.witnesses[label]
            recover(node.children[0], x)
            recover(node.children[1], y)
            return
        event = normalized[node.indices[0]]
        coordinates = coordinate_solver(event['rows'])
        projection = [coordinates(row) for row in node.separator]
        solution = solve([(row, (label >> bit) & 1) for bit, row in enumerate(projection)], len(event['rows']))
        if solution is None:
            raise AssertionError('An attainable interface has no lift')
        origin, basis = solution
        # At most |B|+1 distinct candidates are needed in a nonempty good fiber.
        target = next(origin ^ xor_mask(basis, value)
                      for value in range(min(1 << len(basis), len(event['forbidden'])+1))
                      if origin ^ xor_mask(basis, value) not in event['forbidden'])
        equations.extend((row, (target >> bit) & 1) for bit, row in enumerate(event['rows']))

    recover(root, 0)
    solution = solve(equations, dimension)
    if solution is None:
        raise AssertionError('Compatible interface signatures did not lift')
    parameter, kernel = solution
    if any((project(parameter, event['rows']) ^ event.get('offset', 0)) in event['forbidden']
           for event in events):
        raise AssertionError('Recovered parameter violates an exclusion')
    result.update(parameter=parameter, free_lift_dimension=len(kernel))
    return result


def polar_outer_events(profile, columns, left_inputs, cutoff):
    """Build m tensor Q rows, with T packed column by column; all T entries free."""
    model = RelativePlotkinSpectrum(profile, columns)
    inputs = tuple(left_inputs)
    if len(set(inputs)) != len(inputs):
        raise ValueError('Duplicate critical left input')
    events = []
    f = model.redundancy
    for u in inputs:
        if u <= 0 or u >> model.half or any(
                (u & ((1 << j) ^ model.left_columns[j])).bit_count() & 1
                for j in range(model.half) if j not in model.left_profile):
            raise ValueError('Expected a nonzero input in the fixed left code')
        item = model.constraint(u, cutoff)
        rows = []
        for bit in range(item['rank']):
            row = sum(((value >> bit) & 1) << q for q, value in enumerate(item['syndrome_map']))
            rows.append(sum(row << (f*t) for t, i in enumerate(model.left_profile) if u >> i & 1))
        events.append(dict(rows=tuple(rows), forbidden=item['forbidden'], left_input=u))
    return events, f*len(model.left_profile)


def lift_outer_parameter(profile, columns, parameter):
    model = RelativePlotkinSpectrum(profile, columns)
    f = model.redundancy
    if parameter < 0 or parameter >> (f*len(model.left_profile)):
        raise ValueError('Outer parameter is out of range')
    mask = sum(1 << i for i in model.left_profile)
    rows = [j for j in range(model.half) if j not in model.right_profile]
    child = list(columns)
    for q, j in enumerate(rows):
        child[model.half+j] &= ~mask
        child[model.half+j] |= sum(((parameter >> (f*t+q)) & 1) << source
                                  for t, source in enumerate(model.left_profile))
    return child
