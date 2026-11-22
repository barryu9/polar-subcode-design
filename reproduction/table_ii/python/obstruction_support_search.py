"""Deterministic word-specialization of joint elimination with exact separation.

The solver accumulates necessary word-rejection conditions. A complete spectrum
oracle is required before success is reported; a timeout is never infeasibility.
Only a proved inconsistent collection is used for obstruction-guided expansion.
"""

from collections import Counter
import ctypes
from native_paths import library_path
import json
from pathlib import Path
import time
import z3
from external_low_weight_spectrum import WorkLimitError

_LIB = None

def enumerate_words(n, profile, columns, cutoff, max_work=100_000_000):
    """Complete positive-weight catalogue of an actual, unrelaxed code."""
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(library_path('_candidate_block_neighborhood_kernel')))
        ptr = ctypes.POINTER(ctypes.c_uint8)
        _LIB.block_neighborhood.argtypes = [ctypes.c_int, ptr, ptr, ptr,
                                           ctypes.c_int, ctypes.c_uint64]
        _LIB.block_neighborhood.restype = ctypes.c_void_p
        _LIB.polar_spectrum_free.argtypes = [ctypes.c_void_p]
        _LIB.polar_spectrum_free.restype = None
    info = set(profile)
    raw = (ctypes.c_uint8*(n*n))(*(columns[j] >> i & 1 for j in range(n) for i in range(n)))
    pointer = _LIB.block_neighborhood(n, (ctypes.c_uint8*n)(*(int(i in info) for i in range(n))),
                                     raw, (ctypes.c_uint8*n)(), cutoff, max_work)
    if not pointer:
        raise MemoryError('Native allocation failed')
    try:
        result = json.loads(ctypes.string_at(pointer))
    finally:
        _LIB.polar_spectrum_free(pointer)
    if not result['exact']:
        raise (WorkLimitError if 'work limit' in result['error'] else RuntimeError)(result['error'])
    words = sorted((int(u, 2), w) for u, w in result.pop('words'))
    if len({u for u, _ in words}) != len(words):
        raise RuntimeError('Duplicate oracle words')
    histogram = dict(sorted(Counter(w for _, w in words).items()))
    return words, dict(result, spectrum={0: 1, **histogram})

def xor_terms(terms):
    while len(terms) > 1:
        terms = [z3.Xor(terms[k], terms[k+1]) if k+1 < len(terms) else terms[k]
                 for k in range(0, len(terms), 2)]
    return terms[0] if terms else z3.BoolVal(False)

class WordBlockSystem:
    """For singleton families, a nonzero syndrome is exactly an OR of checks."""
    def __init__(self, reference, domains, rows):
        self.reference, self.domains, self.rows = list(reference), domains, tuple(sorted(rows))
        self.variables = {(j, i): z3.Bool(f'd_{j}_{i}') for j in self.rows for i in domains[j]}
        self.solver = z3.Solver()
        self.solver.set(random_seed=0, threads=1, unsat_core=True)
        self.parities, self.words, self.tags = {}, {}, {}

    def condition(self, u):
        checks = []
        for j in self.rows:
            sources = tuple(i for i in self.domains[j] if u >> i & 1)
            key = j, sources
            if key not in self.parities:
                self.parities[key] = xor_terms([self.variables[j, i] for i in sources])
            defect = (u & ((1 << j) ^ self.reference[j])).bit_count() & 1
            checks.append(self.parities[key] != z3.BoolVal(bool(defect)))
        return z3.simplify(z3.Or(checks))

    def add(self, records):
        added = 0
        for u, weight in records:
            if u in self.words:
                continue
            if not u or weight <= 0:
                raise ValueError('Positive-weight words required')
            condition = self.condition(u)
            self.words[u] = weight
            if z3.is_true(condition):
                continue
            tag = z3.Bool(f'word_{len(self.tags)}')
            self.tags[str(tag)] = u
            self.solver.assert_and_track(condition, tag)
            added += 1
        return added

    def check(self, seconds):
        begin = time.perf_counter()
        self.solver.set(timeout=max(1, int(seconds*1000)))
        answer = self.solver.check()
        result = dict(status=str(answer), seconds=time.perf_counter()-begin)
        if answer == z3.sat:
            model = self.solver.model()
            columns = self.reference.copy()
            for (j, i), variable in self.variables.items():
                if z3.is_true(model.eval(variable, model_completion=True)):
                    columns[j] ^= 1 << i
            result['columns'] = columns
        elif answer == z3.unsat:
            result['obstruction'] = sorted(self.tags[str(t)] for t in self.solver.unsat_core())
        else:
            result['reason'] = self.solver.reason_unknown()
        return result
