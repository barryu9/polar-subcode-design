"""Shared-coefficient causal search with exact separation and timeout branching.

Expansions are explicit coordinate sets. Optional divisibility constraints define
a further structured subset. Exhausted solver time is never called infeasibility.
"""

import time
import z3

class BranchSolver:
    def __init__(self, solver, variables, seconds):
        self.solver, self.variables, self.seconds = solver, variables, seconds
        self.frontier = [[]]
        self.calls = []

    def next_model(self, deadline):
        while self.frontier and time.perf_counter() < deadline:
            prefix = self.frontier.pop()
            assumptions = [v if bit else z3.Not(v) for v,bit in zip(self.variables,prefix)]
            self.solver.set(timeout=max(1,int(1000*min(self.seconds,deadline-time.perf_counter()))))
            start = time.perf_counter()
            answer = self.solver.check(*assumptions)
            self.calls.append(dict(depth=len(prefix),status=str(answer),seconds=time.perf_counter()-start))
            if answer == z3.sat:
                self.frontier.append(prefix)
                return 'sat',self.solver.model()
            if answer == z3.unknown:
                if len(prefix) == len(self.variables):
                    self.frontier.append(prefix)
                    return 'unknown',None
                self.frontier.extend([prefix+[True],prefix+[False]])
        return ('unknown' if self.frontier else 'unsat'),None
