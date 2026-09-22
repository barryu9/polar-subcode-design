"""Algorithm 1: exact Count/Gen recursion with truncated WEF terminals."""
from collections import Counter
from math import comb
from numbers import Integral
from time import perf_counter

from .algebra import polar, prepare, insert, reduce_vector, kernel_directions


class WorkLimitError(RuntimeError):
    """Exact computation aborted. No partial spectrum is returned."""


class Budget:
    def __init__(self, limit):
        self.limit, self.stats = limit, Counter()

    def use(self, name, amount=1):
        self.stats[name] += amount
        self.stats["work_units"] += amount
        if self.stats["work_units"] > self.limit:
            raise WorkLimitError("Algorithm 1 work limit exceeded; increase max_work or lower D.")


def lower_bound(q, first):
    while len(q) > 1:
        m = len(q) // 2
        q = tuple(min(a, b) if first < m else a + b for a, b in zip(q[:m], q[m:]))
        first %= m
    return q[0]


class Family:
    def __init__(self, a, s, q, rows, cutoff, work):
        self.work = work
        work.use("affine_families")
        self.m = m = len(q) // 2
        self.base, self.eta, self.free = 0, 0, 0
        positive = []
        for j, (alpha, beta) in enumerate(zip(q[:m], q[m:])):
            one = (s >> j) & 1
            if one:
                self.base += min(alpha, beta)
                self.eta |= int(alpha > beta) << j
            delta = abs(alpha - beta) if one else alpha + beta
            if delta:
                positive.append((delta, j))
            else:
                self.free |= 1 << j
        self.E = cutoff - self.base
        self.pivots, residual = {}, {}
        self.inconsistent = self.E < 0
        for right, left in rows:
            row = right
            rhs = ((left & a).bit_count() ^ (row & self.eta).bit_count()) & 1
            while row & self.free:
                pivot = (row & self.free).bit_length() - 1
                if pivot not in self.pivots:
                    self.pivots[pivot] = row, rhs
                    break
                old, bit = self.pivots[pivot]
                row, rhs = row ^ old, rhs ^ bit
            else:
                while row:
                    pivot = row.bit_length() - 1
                    if pivot not in residual:
                        residual[pivot] = row, rhs
                        break
                    old, bit = residual[pivot]
                    row, rhs = row ^ old, rhs ^ bit
                if row == 0 and rhs:
                    self.inconsistent = True
        self.rank = len(residual)
        self.target = sum(bit << k for k, (_, bit) in enumerate(residual.values()))
        self.factor = 1 << (self.free.bit_count() - len(self.pivots))
        self.items = sorted((delta, j, sum(((row >> j) & 1) << k
                                          for k, (row, _) in enumerate(residual.values())))
                            for delta, j in positive)
        self.directions = None

    def patterns(self):
        """Sorted-cost subset DFS: never constructs a high-weight layer."""
        if self.inconsistent:
            return
        items = [item for item in self.items if item[0] <= self.E]
        suffix = [{} for _ in range(len(items) + 1)]
        for i in range(len(items) - 1, -1, -1):
            suffix[i] = suffix[i + 1].copy()
            insert(suffix[i], items[i][2])
        def visit(start, syndrome, error, cost):
            self.work.use("deviation_nodes")
            if reduce_vector(suffix[start], syndrome):
                return
            if syndrome == 0:
                yield error, cost
            for i in range(start, len(items)):
                delta, j, column = items[i]
                if cost + delta > self.E:
                    break
                yield from visit(i + 1, syndrome ^ column, error | (1 << j), cost + delta)
        yield from visit(0, self.target, 0, 0)

    def histogram(self, policy):
        if self.inconsistent:
            return Counter()
        p = len(self.items)
        h = min(p, self.E // self.items[0][0]) if p else 0
        qbound = sum(comb(p, j) for j in range(h + 1))
        dp_bound = (p + 1) * (self.E + 1) * (1 << self.rank)
        use_dp = policy == "dp" or policy == "auto" and dp_bound < self.m ** 2 * qbound
        if use_dp:
            self.work.use("dp_families")
            states = {(0, 0): 1}
            for delta, _, column in self.items:
                self.work.use("dp_states", len(states))
                following = dict(states)
                for (syndrome, cost), count in states.items():
                    if cost + delta <= self.E:
                        key = syndrome ^ column, cost + delta
                        following[key] = following.get(key, 0) + count
                states = following
            return Counter({self.base + cost: self.factor * count
                            for (syndrome, cost), count in states.items() if syndrome == self.target})
        self.work.use("enumerated_families")
        result = Counter()
        for _, cost in self.patterns():
            result[self.base + cost] += self.factor
        return result

    def vectors(self):
        if self.directions is None:
            self.directions = kernel_directions(self.free, self.pivots)
        for error, cost in self.patterns():
            anchor = error
            for pivot in sorted(self.pivots):
                row, rhs = self.pivots[pivot]
                if ((row & anchor).bit_count() & 1) != rhs:
                    anchor ^= 1 << pivot
            yield anchor ^ self.eta, self.directions, cost


class Solver:
    def __init__(self, n, checks, D, policy, work):
        self.n, self.checks, self.D, self.policy, self.work = n, checks, D, policy, work
        self.contexts = {}

    def rows(self, offset, size):
        key = offset, size
        if key not in self.contexts:
            m, mask = size // 2, (1 << (size // 2)) - 1
            self.contexts[key] = [
                (polar((row >> (offset + m)) & mask, m, True), (row >> offset) & mask)
                for j, row in sorted(self.checks.items()) if offset + m <= j < offset + size]
        return self.contexts[key]

    def wef_pair(self, q, prefix):
        self.work.use("wef_nodes")
        if len(q) == 1:
            return Counter({0: 1}), Counter({q[0]: 1}) if q[0] <= self.D else Counter()
        even = prefix if len(prefix) % 2 == 0 else prefix[:-1]
        e, o = even[::2], even[1::2]
        f = self.wef_pair(q[::2], tuple(a ^ b for a, b in zip(e, o)))
        g = self.wef_pair(q[1::2], o)
        def multiply(a, b):
            if not a or not b or min(a) + min(b) > self.D:
                return Counter()
            result = Counter()
            for wa, ca in a.items():
                if wa > self.D - min(b):
                    continue
                for wb, cb in b.items():
                    if wa + wb <= self.D:
                        self.work.use("coefficient_products")
                        result[wa + wb] += ca * cb
            return result
        if len(prefix) % 2:
            b = prefix[-1]
            return multiply(f[b], g[0]), multiply(f[1 - b], g[1])
        return (multiply(f[0], g[0]) + multiply(f[1], g[1]),
                multiply(f[0], g[1]) + multiply(f[1], g[0]))

    def generate(self, offset, q, first):
        self.work.use("generation_calls")
        if lower_bound(q, first) > self.D:
            return
        if len(q) == 1:
            self.work.use("generated_pairs")
            yield 1, 1
            return
        m = len(q) // 2
        if first >= m:
            for b, t in self.generate(offset + m, tuple(a + b for a, b in zip(q[:m], q[m:])), first - m):
                self.work.use("generated_pairs")
                yield b << m, t | (t << m)
            return
        for a, s in self.generate(offset, tuple(min(a, b) for a, b in zip(q[:m], q[m:])), first):
            family = Family(a, s, q, self.rows(offset, len(q)), self.D, self.work)
            # Transform each kernel direction once per family.
            directions = kernel_directions(family.free, family.pivots)
            family.directions = directions
            du = [polar(v, m) << m for v in directions]
            dx = [v | (v << m) for v in directions]
            for anchor, _, _ in family.vectors():
                u, x = a | (polar(anchor, m) << m), (s ^ anchor) | (anchor << m)
                previous = 0
                for number in range(1 << len(directions)):
                    if number:
                        gray = number ^ (number >> 1)
                        changed = (gray ^ previous).bit_length() - 1
                        u, x = u ^ du[changed], x ^ dx[changed]
                        previous = gray
                    self.work.use("generated_pairs")
                    yield u, x

    def count(self, offset, q, first):
        self.work.use("count_calls")
        if lower_bound(q, first) > self.D:
            return Counter()
        if not any(offset + first < j < offset + len(q) for j in self.checks):
            self.work.use("wef_terminals")
            return self.wef_pair(q, (0,) * first)[1]
        m = len(q) // 2
        if first >= m:
            return self.count(offset + m, tuple(a + b for a, b in zip(q[:m], q[m:])), first - m)
        result = Counter()
        for a, s in self.generate(offset, tuple(min(a, b) for a, b in zip(q[:m], q[m:])), first):
            result.update(Family(a, s, q, self.rows(offset, len(q)), self.D, self.work).histogram(self.policy))
        return result


def validate(n, checks, D, punctured, shortened, policy, max_work):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1 or n & (n - 1):
        raise ValueError("n must be a positive power of two.")
    if isinstance(D, bool) or not isinstance(D, Integral) or D < 0:
        raise ValueError("D must be a nonnegative integer.")
    if policy not in ("auto", "enumerate", "dp"):
        raise ValueError("policy must be auto, enumerate, or dp.")
    if isinstance(max_work, bool) or not isinstance(max_work, Integral) or not 1 <= max_work <= 2**63 - 1:
        raise ValueError("max_work must be a positive signed 64-bit integer.")
    checks = tuple(checks)
    if any(isinstance(row, bool) or not isinstance(row, Integral) or not 0 <= row < 1 << n for row in checks):
        raise ValueError("checks must be nonnegative n-bit integer row masks.")
    punctured, shortened = tuple(punctured), tuple(shortened)
    for positions in (punctured, shortened):
        if len(set(positions)) != len(positions) or any(isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < n for j in positions):
            raise ValueError("Rate-matching coordinates must be distinct indices in [0,n).")
    if set(punctured) & set(shortened):
        raise ValueError("Punctured and shortened positions must be disjoint.")
    return tuple(int(x) for x in checks), tuple(sorted(punctured)), tuple(sorted(shortened))


def algorithm1(n, checks, D, *, punctured=(), shortened=(), backend="python",
               policy="auto", max_work=10_000_000):
    """Return exact distinct transmitted-word multiplicities through D.

    checks: integer masks for rows of H in H u^T=0, x=u F^(tensor log2 n).
    No SNR, random channel sample, decoder list or incomplete result is used.
    """
    checks, punctured, shortened = validate(n, checks, D, punctured, shortened, policy, max_work)
    if backend == "cpp":
        from .native import compute
        return compute(n, checks, D, punctured, shortened, policy, max_work)
    if backend != "python":
        raise ValueError("backend must be python or cpp.")
    start = perf_counter()
    prepared, dimensions = prepare(n, checks, punctured, shortened)
    actual_D = min(D, n - len(punctured) - len(shortened))
    q = tuple(0 if j in punctured else actual_D + 1 if j in shortened else 1 for j in range(n))
    middle = perf_counter()
    work = Budget(max_work)
    solver = Solver(n, prepared, actual_D, policy, work)
    spectrum = Counter({0: 1})
    for i in range(n):
        if i not in prepared:
            spectrum.update(solver.count(0, q, i))
    stop = perf_counter()
    if spectrum[0] != 1:
        raise AssertionError("Projection-injective representative invariant failed.")
    return dict(spectrum=dict(sorted(spectrum.items())), exact=True, n=n, cutoff=D,
                evaluated_cutoff=actual_D, backend="python", policy=policy,
                **dimensions, preprocessing_seconds=middle-start,
                counting_seconds=stop-middle, elapsed_seconds=stop-start, stats=dict(work.stats))
