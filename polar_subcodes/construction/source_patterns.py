"""Exact source-pattern search in a complete, coefficient-restricted catalogue."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math


def set_bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length()-1
        mask ^= bit


def xor_translation_table(patterns, width):
    """Translate a bitset by all XOR offsets using block-swap permutations."""
    size = 1 << width
    if patterns < 0 or patterns >> size:
        raise ValueError('Pattern bitset exceeds its width')
    full = (1 << size)-1
    masks = [full//((1 << (1 << q))+1) for q in range(width)]
    translated = [patterns]
    for shift in range(1, size):
        bit = shift & -shift
        value = translated[shift ^ bit]
        low = masks[bit.bit_length()-1]
        translated.append(((value & low) << bit) | ((value >> bit) & low))
    return tuple(translated)


@dataclass
class PatternState:
    scopes: list
    residuals: list
    domains: list
    values: list


class SourcePatternProblem:
    """A word survives precisely when XOR of its source patterns is its defect."""
    def __init__(self, catalogue, *, projected_groups=None, max_width=8):
        self.catalogue = c = catalogue
        self.width = len(c.rows)
        if not 0 <= self.width <= max_width <= 12:
            raise ValueError('Dense pattern width exceeds the configured limit (at most twelve)')
        self.sources = tuple(sorted({i for _, i in c.domain}, reverse=True))
        self.source_index = {i: k for k, i in enumerate(self.sources)}
        row_masks = [sum(1 << b for b, i in c.domain if i == source)
                     for source in self.sources]
        self.domains = [sum(1 << v for v in range(1 << self.width) if not v & ~mask)
                        for mask in row_masks]
        self.groups = Counter()
        information = set(c.design.profile)
        for u, weight in (() if projected_groups is not None else c.words):
            if weight <= 0 or (u & -u).bit_length()-1 not in information:
                continue
            scope = sum(((u >> source) & 1) << k
                        for k, source in enumerate(self.sources))
            defect = sum((((u >> row) & 1) ^ ((u & c.columns[row]).bit_count() & 1)) << b
                         for b, row in enumerate(c.rows))
            self.groups[scope, defect] += 1
        if projected_groups is not None:
            for (scope, defect), count in projected_groups.items():
                if (scope < 0 or scope >> len(self.sources) or defect < 0
                        or defect >> self.width or count <= 0):
                    raise ValueError('Invalid projected family multiplicity')
                self.groups[scope, defect] += count
        by_scope = defaultdict(int)
        for scope, defect in self.groups:
            by_scope[scope] |= 1 << defect
        ordered = sorted(by_scope.items())
        self.scopes = [scope for scope, _ in ordered]
        self.forbidden = [patterns for _, patterns in ordered]
        self.translations = [xor_translation_table(patterns, self.width)
                             for patterns in self.forbidden]
        self.incidence = [[] for _ in self.sources]
        for g, scope in enumerate(self.scopes):
            for k in set_bits(scope):
                self.incidence[k].append(g)
        self.initial = PatternState(
            self.scopes.copy(), [0]*len(ordered), self.domains.copy(),
            [-1]*len(self.sources))
        self.root_conflict = False
        for scope, patterns in ordered:
            if not scope:
                self.root_conflict |= bool(patterns & 1)
            elif not scope & (scope-1):
                k = scope.bit_length()-1
                self.initial.domains[k] &= ~patterns
                self.root_conflict |= not self.initial.domains[k]
        signature = dict(sources=self.sources, rows=c.rows, domains=self.domains,
                         columns=c.columns, groups=sorted(self.groups.items()))
        self.signature = hashlib.sha256(json.dumps(signature).encode()).hexdigest()

    def statistics(self):
        return dict(
            variables=len(self.catalogue.domain), source_variables=len(self.sources),
            source_order=self.sources, pattern_domain_sizes=[d.bit_count() for d in self.domains],
            distinct_input_patterns=len(self.scopes), word_groups=len(self.groups),
            represented_words=sum(self.groups.values()),
            group_multiplicities=dict(Counter(self.groups.values())),
            forbidden_set_sizes=dict(Counter(mask.bit_count() for mask in self.forbidden)),
            root_domain_sizes=[d.bit_count() for d in self.initial.domains],
            constant_obstruction=self.root_conflict, signature=self.signature)

    def assign(self, state, k, value):
        if state.values[k] != -1 or not (state.domains[k] >> value) & 1:
            raise ValueError('Assignment is not in the remaining source domain')
        child = PatternState(state.scopes.copy(), state.residuals.copy(),
                             state.domains.copy(), state.values.copy())
        child.values[k] = value
        child.domains[k] = 1 << value
        keep = ~(1 << k)
        for g in self.incidence[k]:
            scope = child.scopes[g] & keep
            residual = child.residuals[g] ^ value
            child.scopes[g], child.residuals[g] = scope, residual
            if not scope:
                if (self.forbidden[g] >> residual) & 1:
                    return None
            elif not scope & (scope-1):
                other = scope.bit_length()-1
                child.domains[other] &= ~self.translations[g][residual]
                if not child.domains[other]:
                    return None
        return child

    def deltas(self, values):
        if len(values) != len(self.sources):
            raise ValueError('Expected one output pattern for every source')
        result = [0]*self.width
        for k, (source, value) in enumerate(zip(self.sources, values)):
            if value < 0 or not (self.domains[k] >> value) & 1:
                raise ValueError('Noncausal source pattern')
            for row in set_bits(value):
                result[row] |= 1 << source
        return result

    def survivors(self, values):
        self.deltas(values)
        total = 0
        for (scope, defect), count in self.groups.items():
            pattern = 0
            for k in set_bits(scope):
                pattern ^= values[k]
            if pattern == defect:
                total += count
        return total


class PatternSearch:
    """Finite DFS with exact last-source propagation and a resumable frontier."""
    def __init__(self, problem, variable_order='reverse', value_order='sparse'):
        if variable_order not in ('reverse', 'mrv') or value_order not in ('sparse', 'lookahead', 'product'):
            raise ValueError('Unknown deterministic branching order')
        self.problem = problem
        self.variable_order, self.value_order = variable_order, value_order
        self.nodes, self.conflicts, self.probes, self.deepest = 0, 0, 0, 0
        self.solution = None
        self.stack = []
        if not problem.root_conflict:
            if problem.sources:
                self.stack.append(self.make_frame(problem.initial, []))
            else:
                self.solution = []

    def make_frame(self, state, prefix):
        remaining = [k for k, value in enumerate(state.values) if value == -1]
        if self.variable_order == 'reverse':
            k = remaining[0]
        else:
            k = min(remaining, key=lambda j: (
                state.domains[j].bit_count(), -len(self.problem.incidence[j]), j))
        values = sorted(set_bits(state.domains[k]), key=lambda v: (v.bit_count(), v))
        if self.value_order in ('lookahead', 'product'):
            scored = []
            for value in values:
                child = self.problem.assign(state, k, value)
                self.probes += 1
                if child is None:
                    self.conflicts += 1
                    continue
                sizes = [child.domains[j].bit_count() for j in remaining if j != k]
                score = (math.prod(sizes) if self.value_order == 'product'
                         else sum(math.log2(size) for size in sizes))
                scored.append((-score, value.bit_count(), value))
            values = [value for _, _, value in sorted(scored)]
        return dict(state=state, prefix=prefix, variable=k, values=values, cursor=0)

    @property
    def status(self):
        if self.solution is not None:
            return 'sat'
        return 'pending' if self.stack else 'unsat'

    def step(self):
        if self.status != 'pending':
            return self.status
        frame = self.stack[-1]
        if frame['cursor'] >= len(frame['values']):
            self.stack.pop()
            return self.status
        k, value = frame['variable'], frame['values'][frame['cursor']]
        frame['cursor'] += 1
        self.nodes += 1
        child = self.problem.assign(frame['state'], k, value)
        if child is None:
            self.conflicts += 1
            return self.status
        prefix = frame['prefix']+[(k, value)]
        self.deepest = max(self.deepest, len(prefix))
        if len(prefix) == len(self.problem.sources):
            if self.problem.survivors(child.values):
                raise AssertionError('Pattern propagation accepted a forbidden word')
            self.solution = child.values
        else:
            self.stack.append(self.make_frame(child, prefix))
        return self.status

    def snapshot(self):
        return dict(
            version=1, signature=self.problem.signature,
            variable_order=self.variable_order, value_order=self.value_order,
            nodes=self.nodes, conflicts=self.conflicts, probes=self.probes, deepest=self.deepest,
            status=self.status, solution=self.solution,
            frames=[{k: v for k, v in frame.items() if k != 'state'} for frame in self.stack])

    @classmethod
    def restore(cls, problem, snapshot):
        if snapshot['version'] != 1 or snapshot['signature'] != problem.signature:
            raise ValueError('Frontier does not describe this exact pattern problem')
        obj = cls.__new__(cls)
        obj.problem = problem
        for key in ('variable_order', 'value_order', 'nodes', 'conflicts', 'probes',
                    'deepest', 'solution'):
            setattr(obj, key, snapshot[key])
        obj.stack = []
        for stored in snapshot['frames']:
            state = problem.initial
            for k, value in stored['prefix']:
                state = problem.assign(state, k, value)
                if state is None:
                    raise ValueError('Inconsistent saved prefix')
            k = stored['variable']
            values = stored['values']
            if (state.values[k] != -1 or len(set(values)) != len(values)
                    or not 0 <= stored['cursor'] <= len(values)
                    or any(not (state.domains[k] >> value) & 1 for value in values)):
                raise ValueError('Invalid saved frame')
            obj.stack.append(dict(stored, state=state))
        if obj.status != snapshot['status']:
            raise ValueError('Saved status disagrees with the frontier')
        return obj
