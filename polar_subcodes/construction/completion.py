"""Streaming exact coset exclusions and conditional-expectation bit backtracking."""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import chain, combinations

import numpy as np

from polar_subcodes.construction.conditional import suffix_coordinates, xor_mask


@lru_cache(maxsize=8)
def combination_indices(length, size):
    if length < size:
        return np.empty((0, size), dtype=np.int32)
    return np.fromiter(chain.from_iterable(combinations(range(length), size)),
                       dtype=np.int32).reshape(-1, size)


def small_error_labels(columns, radius):
    if not 0 <= radius <= 3 or any(value < 0 or value >= 1 << 63 for value in columns):
        raise ValueError('Expected nonnegative 63-bit labels and radius at most three')
    columns = np.asarray(columns, dtype=np.uint64)
    parts = [np.array([0], dtype=np.uint64)]
    for size in range(1, radius+1):
        indices = combination_indices(len(columns), size)
        parts.append(np.bitwise_xor.reduce(columns[indices], axis=1))
    return np.unique(np.concatenate(parts))


@dataclass
class CosetExclusion:
    message: int
    coordinates: tuple
    pivots: dict
    labels: np.ndarray
    rank: int
    left_input: int


def prepare_exclusion(model, left_input, cutoff, info_mask):
    item = model.constraint(left_input, cutoff, materialize=False)
    if item['rank'] >= 63:
        raise ValueError('Dense label arrays require a projected rank below 63')
    coordinates, pivots, encode = suffix_coordinates(item['syndrome_map'])
    labels = small_error_labels([encode(value) for value in item['error_columns']], item['radius'])
    return CosetExclusion(
        left_input & info_mask, tuple(coordinates), pivots, labels, item['rank'], left_input)


class SourceCompletion:
    """Every rejected prefix is covered by an actual forbidden coset event.

    Conditional expectation orders branches, but an expectation of at least one
    is never used as an infeasibility test. Unvisited sibling prefixes persist.
    """
    def __init__(self, constraints, previous_cross, width, frontier=None):
        self.constraints = constraints
        self.width = width
        if any(len(item.coordinates) != width for item in constraints):
            raise ValueError('Cross-column widths do not match')
        self.rank = max((item.rank for item in constraints), default=0)
        self.root = []
        for item in constraints:
            shift = 0
            for source, value in previous_cross.items():
                if (item.message >> source) & 1:
                    shift ^= value
            self.root.append((item.labels, xor_mask(item.coordinates, shift), item.rank))
        self.frontier = [(0, 0, self.root)] if frontier is None else [
            (position, value, None) for position, value in frontier]
        self.nodes, self.covered_prefixes, self.solutions = 0, 0, 0
        self.greedy_path_only = True

    def potential(self, state):
        return sum(len(labels) << (self.rank-remaining) for labels, _, remaining in state)

    def split(self, state, position, bit):
        child = []
        for item, (labels, offset, remaining) in zip(self.constraints, state):
            pivot = item.pivots.get(position)
            if pivot is not None:
                parity = ((offset >> pivot) & 1) ^ bit
                labels = labels[((labels >> pivot) & 1) == parity]
                remaining -= 1
            if bit:
                offset ^= item.coordinates[position]
            child.append((labels, offset, remaining))
        return child

    def reconstruct(self, position, value):
        if position < 0 or position > self.width or value < 0 or value >> position:
            raise ValueError('Malformed saved bit prefix')
        state = self.root
        for index in range(position):
            state = self.split(state, index, (value >> index) & 1)
        return state

    def snapshot(self):
        return [[position, value] for position, value, _ in self.frontier]

    def step(self):
        if not self.frontier:
            return dict(status='unsat')
        position, value, state = self.frontier.pop()
        state = self.reconstruct(position, value) if state is None else state
        self.nodes += 1
        if any(len(labels) == 1 << remaining for labels, _, remaining in state):
            self.covered_prefixes += 1
            self.greedy_path_only = False
            return dict(status='pending')
        if position == self.width:
            if self.potential(state):
                raise AssertionError('An accepted leaf contains a forbidden label')
            self.solutions += 1
            return dict(status='sat', value=value,
                        greedy_path_only=self.greedy_path_only)
        children = []
        for bit in (0, 1):
            child = self.split(state, position, bit)
            children.append((self.potential(child), bit, child))
        # LIFO: the smaller exact conditional expectation is visited first.
        for _, bit, child in sorted(children, key=lambda entry: (entry[0], entry[1]), reverse=True):
            self.frontier.append((position+1, value | (bit << position), child))
        return dict(status='pending')

    def expectation(self):
        return Fraction(self.potential(self.root), 1 << self.rank)

    def expectation_path(self, value):
        if value < 0 or value >> self.width:
            raise ValueError('Source pattern exceeds the cross-block width')
        state = self.root
        result = [dict(fixed_bits=0, expectation=str(self.expectation()))]
        for position in range(self.width):
            state = self.split(state, position, (value >> position) & 1)
            result.append(dict(
                fixed_bits=position+1,
                expectation=str(Fraction(self.potential(state), 1 << self.rank))))
        return result

    def prefix_completion_bound(self, fixed_bits, prefix_value):
        state = self.reconstruct(fixed_bits, prefix_value)
        expectation = Fraction(self.potential(state), 1 << self.rank)
        remaining = self.width-fixed_bits
        lower = max(Fraction(0), (1-expectation)*(1 << remaining))
        count = (lower.numerator+lower.denominator-1)//lower.denominator
        return dict(
            fixed_bits=fixed_bits, prefix_value=prefix_value, remaining_dimension=remaining,
            conditional_expected_bad_cosets=str(expectation),
            successful_patterns_lower_bound=count,
            total_remaining_patterns=1 << remaining,
            conditional_success_probability_lower_bound=str(max(Fraction(0), 1-expectation)))
