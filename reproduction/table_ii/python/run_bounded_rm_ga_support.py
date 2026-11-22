"""Bounded zero-initialized support construction for a selected profile.

Use exact word rejection, complete source-pattern search, and verified Plotkin
lifts. All limits and fixed-pair obstructions retain their precise scopes.
"""

from collections import defaultdict
import json
import time
from types import SimpleNamespace
import z3
from external_low_weight_spectrum import WorkLimitError
from obstruction_support_search import enumerate_words, WordBlockSystem
from run_expanded_distance_search import BranchSolver
from run_reference_domain_sweep import domain, write
from run_profile_target_construction import complete_outer
from select_structured_domain import initial_structure, complete_domain

class Trial:
    def __init__(self, args):
        self.args = args
        self.started = time.perf_counter()
        self.deadline = self.started + args.total_seconds
        self.stages = []

    def remaining(self):
        return max(0., min(self.args.seconds, self.deadline-time.perf_counter()))

    def record(self, stage, result):
        self.stages.append(dict(stage=stage, **result))
        write(self.args.output / "stages.json", self.stages)
        print(json.dumps(dict(stage=stage, status=result.get("status"))), flush=True)

    def lazy(self, n, profile, baseline, edges, distance, folder):
        folder.mkdir(parents=True)
        domains = defaultdict(list)
        for j, i in edges:
            domains[j].append(i)
        system = WordBlockSystem(baseline, domains, sorted(domains))
        order = sorted(system.variables, key=lambda e: (-e[1], e[0]))
        brancher = BranchSolver(system.solver, [system.variables[e] for e in order], 2)
        end = time.perf_counter() + self.remaining()
        columns, trace, best = list(baseline), [], None
        status = "budget_exhausted"
        child = None
        source_mask = sum(1 << i for i in {i for j, i in edges})
        write(folder / "domain.json", dict(N=n, profile=profile, baseline=baseline,
            coordinates=edges, target=distance, reference_support_read=False))
        try:
            for iteration in range(1000):
                if time.perf_counter() >= end:
                    break
                words, metadata = enumerate_words(n, profile, columns, distance-1, self.args.work)
                score = tuple(metadata["spectrum"].get(w, 0) for w in range(1, distance))
                event = dict(iteration=iteration, spectrum=metadata["spectrum"], bad=len(words))
                trace.append(event)
                write(folder / "trace.json", trace)
                if best is None or score < best:
                    best = score
                    write(folder / "best.json", dict(N=n, profile=profile, columns=columns, **event))
                if not words:
                    child, status = columns, "verified_feasible"
                    write(folder / "word_verification.json", metadata)
                    break
                groups = {}
                for u, w in words:
                    defect = sum(((u & ((1 << j) ^ baseline[j])).bit_count() % 2) << j
                                 for j in domains)
                    groups.setdefault((u & source_mask, defect), (u, w))
                system.add(groups.values())
                answer, model = brancher.next_model(end)
                if answer != "sat":
                    status = "selected_domain_unsat" if answer == "unsat" else "budget_or_solver_unknown"
                    break
                columns = list(baseline)
                for (j, i), variable in system.variables.items():
                    if z3.is_true(model.eval(variable, model_completion=True)):
                        columns[j] ^= 1 << i
        except WorkLimitError:
            status = "oracle_work_limit"
        result = dict(status=status, variables=len(edges), iterations=len(trace),
            maximum_branch_depth=max((v['depth'] for v in brancher.calls), default=0),
            scope="Only the coordinates and fixed baseline in domain.json")
        write(folder / "result.json", result)
        write(folder / "branches.json", dict(order=order, frontier=brancher.frontier, calls=brancher.calls))
        (folder / "constraints.smt2").write_text(system.solver.to_smt2(), encoding="utf-8")
        self.record(str(folder.name), result)
        return child

    def dense(self, n, profile, distance, folder):
        structure = initial_structure(n, profile, distance)
        active = sorted({q for q, i in structure["coordinates"]})
        if not 1 <= len(active) <= 8:
            return None
        positions = {q: r for r, q in enumerate(active)}
        rows = tuple(structure["rows"][q] for q in active)
        coordinates = tuple((positions[q], i) for q, i in structure["coordinates"])
        args = SimpleNamespace(seconds=self.remaining(), nodes=self.args.nodes, work=self.args.work)
        try:
            child, result = complete_domain(n, profile, [0]*n, rows, coordinates,
                                            distance, folder, args)
        except WorkLimitError:
            result, child = dict(status="oracle_work_limit", scope="Initial retained-core domain"), None
        self.record(folder.name, result)
        return child

    def build(self, n, profile, distance, folder):
        folder.mkdir(parents=True, exist_ok=True)
        if 0 not in profile:
            distance += distance % 2
        write(folder / "profile.json", dict(N=n, K=len(profile), profile=profile, target=distance))
        baseline = min(1 << i.bit_count() for i in profile)
        if baseline >= distance:
            self.record(folder.name, dict(status="static_distance_sufficient", dmin=baseline,
                proof="Unprecoded polar distance equals the minimum selected row weight"))
            return [0]*n
        if self.remaining() <= 0:
            return None
        structure, initial_edges = domain(n, profile, distance, "initial")
        active_count = len({j for j, i in initial_edges})
        if n <= 128 and 1 <= active_count <= 8:
            child = self.dense(n, profile, distance, folder / "initial_dense")
            if child is not None:
                return child
        if n >= 32 and self.remaining() > 0:
            h = n//2
            left = tuple(i for i in profile if i < h)
            right = tuple(i-h for i in profile if i >= h)
            right_target = (distance+1)//2
            child = self.build(h, right, right_target, folder / f"suffix{h}") if right else None
            if child is not None:
                embedded = [0]*h + [c << h for c in child]
                left_min = min((1 << i.bit_count() for i in left), default=n+1)
                if left_min >= distance:
                    self.record(folder.name, dict(status="distance_protected_lift", left_distance=left_min,
                        right_lower_bound=right_target,
                        proof="Mixed weight is at least wt(a), and pure-right weights are doubled"))
                    return embedded
                if self.remaining() > 0:
                    if (distance-1-left_min)//2 <= 3:
                        try:
                            completed, result = complete_outer(profile, embedded, distance,
                                self.args.work, self.remaining(), self.args.nodes, folder / "outer")
                            self.record(folder.name + "_outer", result)
                            if completed is not None:
                                return completed
                        except (WorkLimitError, ValueError) as error:
                            self.record(folder.name + "_outer", dict(status="oracle_or_representation_limit", detail=str(error)))
                    if self.remaining() > 0:
                        edges = [(j, i) for j in range(h, n) if j not in profile for i in left]
                        completed = self.lazy(n, profile, embedded, edges, distance, folder / "outer_lazy")
                        if completed is not None:
                            return completed
        if self.remaining() > 0:
            child = self.lazy(n, profile, [0]*n, initial_edges, distance, folder / "initial_lazy")
            if child is not None:
                return child
        if self.remaining() > 0:
            _, expanded = domain(n, profile, distance, "core_relaxed")
            if expanded != initial_edges:
                return self.lazy(n, profile, [0]*n, expanded, distance, folder / "expanded_lazy")
        return None
