"""Joint whole-row design from a complete relaxed low-weight catalogue.

Every forbidden word must violate at least one restored check. All retained
coefficients in the selected rows are variables; there is no RM restriction.
Solver timeouts leave a bounded search result, never an optimality claim.
"""

from collections import Counter
import time
import z3

class JointBlockDesigner:
    def __init__(self, catalogue, first_source=0, seed=1):
        self.catalogue = c = catalogue
        self.coordinates = [(b, i) for b, i in c.domain if i >= first_source]
        self.variables = {(b, i): z3.Bool(f"delta_{b}_{i}") for b, i in self.coordinates}
        self.masks = [sum(1 << i for bb, i in self.coordinates if bb == b)
                      for b in range(len(c.rows))]
        self.parities = {}
        self.solver = z3.Solver()
        self.solver.set(random_seed=seed)
        groups = Counter()
        information = set(c.design.profile)
        for u, w in c.words:
            if (u & -u).bit_length()-1 not in information:
                continue
            key = tuple((u & self.masks[b],
                         ((u >> j) & 1) ^ ((u & c.columns[j]).bit_count() & 1))
                        for b, j in enumerate(c.rows))
            groups[w < c.cutoff, key] += 1
        self.objective = []
        self.constant = 0
        for (lower, key), multiplicity in groups.items():
            errors = [self.parity(b, mask) != z3.BoolVal(bool(defect))
                      for b, (mask, defect) in enumerate(key)]
            survives = z3.simplify(z3.Not(z3.Or(errors)))
            if lower:
                self.solver.add(z3.Not(survives))
            elif z3.is_true(survives):
                self.constant += multiplicity
            elif not z3.is_false(survives):
                self.objective.append((survives, multiplicity))
        self.groups = len(groups)

    def parity(self, b, mask):
        key = b, mask
        if key not in self.parities:
            mask &= self.masks[b]
            if not mask:
                value = z3.BoolVal(False)
            elif not mask & (mask-1):
                value = self.variables[b, mask.bit_length()-1]
            else:
                low = (mask & -mask).bit_length()-1
                high = mask.bit_length()-1
                level = (low ^ high).bit_length()-1
                split = (high >> level) << level
                left = mask & ((1 << split)-1)
                # Fixed index intervals share parity subexpressions across words.
                value = z3.Xor(self.parity(b, left), self.parity(b, mask ^ left))
            self.parities[key] = value
        return self.parities[key]

    def check(self, bound, timeout_ms):
        start = time.perf_counter()
        if bound < self.constant:
            return dict(status='unsat', bound=bound, seconds=0,
                        reason='constant_survivors', deltas=None)
        constraint = (z3.PbLe(self.objective, bound-self.constant)
                      if self.objective else z3.BoolVal(True))
        self.solver.set(timeout=max(1, int(timeout_ms)))
        status = self.solver.check(constraint)
        result = dict(status=str(status), bound=bound,
                      seconds=time.perf_counter()-start, deltas=None)
        if status == z3.sat:
            model = self.solver.model()
            deltas = [0]*len(self.catalogue.rows)
            for (b, i), variable in self.variables.items():
                if z3.is_true(model.eval(variable, model_completion=True)):
                    deltas[b] |= 1 << i
            value, lower = self.catalogue.score(deltas)
            if lower or value > bound:
                raise RuntimeError('Joint model disagrees with complete catalogue')
            result.update(deltas=deltas, value=value)
        elif status == z3.unknown:
            result['reason'] = self.solver.reason_unknown()
        return result

    def minimize(self, seconds=60, call_seconds=10, max_calls=8):
        c = self.catalogue
        best, lower = c.score([0]*len(c.rows))
        if lower:
            raise ValueError('Reference must have no positive weights below the target')
        start = time.perf_counter()
        deltas = [0]*len(c.rows)
        floor, trace = self.constant, []
        bound = floor
        for iteration in range(max_calls):
            remaining = seconds-(time.perf_counter()-start)
            if remaining <= 0 or floor == best:
                break
            answer = self.check(bound, min(remaining, call_seconds)*1000)
            trace.append(answer)
            if answer['status'] == 'sat':
                best, deltas = answer['value'], answer['deltas']
                bound = (floor+best-1)//2
            elif answer['status'] == 'unsat':
                floor = max(floor, bound+1)
                bound = best-1 if iteration == 0 else (floor+best-1)//2
            else:
                # A feasible nearby improvement can be easier than elimination.
                bound = best-1 if bound != best-1 else (floor+best-1)//2
        return dict(value=best, deltas=deltas, lower_bound=floor,
                    optimum_proved=floor == best, variables=len(self.variables),
                    word_groups=self.groups, constant_survivors=self.constant,
                    seconds=time.perf_counter()-start, trace=trace)
