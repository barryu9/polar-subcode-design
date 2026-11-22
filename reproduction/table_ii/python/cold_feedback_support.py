"""Zero-initialized construction with scoped, shared-coefficient feedback.

No reference profile, constituent template, or checkpoint is read. The dense
source-pattern backend is tried when its exact representation is available.
The fallback keeps one full causal coefficient system and relaxes only locks
identified in an UNSAT core. UNKNOWN never releases a lock as a proof step.
"""

from pathlib import Path
import time
from types import SimpleNamespace
import z3
from bounded_outer_patterns import outer_pattern_problem
from compressed_source_search import PatternSearch
from compressed_source_search import SourcePatternProblem
from external_low_weight_spectrum import WorkLimitError
from obstruction_support_search import enumerate_words
from run_bounded_rm_ga_support import Trial
from run_reference_domain_sweep import clause_for_input, domain, write
from support_graph_policy import Design
from verify_causal_flag_theory import polar_transform

class ProvedSuffixInfeasible(Exception):
    """A full causal right-suffix exclusion is a necessary profile obstruction."""

    def __init__(self, n, profile, target, folder):
        self.certificate = dict(length=n, local_profile=list(profile), target=target,
                                proof_folder=str(folder),
                                scope="All causal coefficients for this suffix profile",
                                parity_rounding_valid=0 not in profile)
        super().__init__("A necessary right constituent has no causal support completion")

class LockedCoefficientSystem:
    """Witness clauses are global; coefficient locks define the current domain."""

    def __init__(self, n, profile, active, baseline=None):
        self.n, self.profile = n, tuple(profile)
        self.baseline = list(baseline or [0] * n)
        self.edges = tuple((j, i) for j in range(n) if j not in profile
                           for i in profile if i < j)
        self.variables = {e: z3.Bool(f"c_{e[0]}_{e[1]}") for e in self.edges}
        self.active = set(active)
        if not self.active <= set(self.edges):
            raise ValueError("Noncausal active coordinate")
        self.solver = z3.Solver()
        self.solver.set(random_seed=0)
        self.locks = {e: z3.Bool(f"lock_{e[0]}_{e[1]}") for e in self.edges}
        self.lock_edges = {str(v): e for e, v in self.locks.items()}
        for e, v in self.variables.items():
            self.solver.add(z3.Implies(self.locks[e], z3.Not(v)))
        self.witnesses = []
        self.seen = set()

    def add(self, u, weight):
        if u in self.seen:
            return
        if polar_transform(u, self.n).bit_count() != weight or weight <= 0:
            raise ValueError("Invalid low-weight witness")
        self.seen.add(u)
        # Adjust the observed input by the fixed baseline. Variables are deltas.
        corrected = u
        for j, c in enumerate(self.baseline):
            if (u & c).bit_count() & 1:
                corrected ^= 1 << j
        tag = z3.Bool(f"word_{len(self.witnesses)}")
        self.solver.assert_and_track(clause_for_input(corrected, self.variables), tag)
        self.witnesses.append(dict(input=u, weight=weight))

    def check(self, seconds):
        self.solver.set(timeout=max(1, int(seconds * 1000)))
        return self.solver.check(*[self.locks[e] for e in self.edges if e not in self.active])

    def obstruction(self):
        core = sorted(map(str, self.solver.unsat_core()))
        locks = sorted(self.lock_edges[t] for t in core if t in self.lock_edges)
        indices = sorted(int(t[5:]) for t in core if t.startswith("word_"))
        return dict(scope="Current baseline plus active coefficient coordinates",
                    locked_coordinates=locks, witness_indices=indices,
                    witnesses=[self.witnesses[i] for i in indices],
                    full_profile_infeasible=not locks,
                    full_delta_domain_spans_all_causal_supports=True)

    def standalone_obstruction(self, obstruction):
        """Export hard witness clauses and locks, without implicit assumptions."""
        solver = z3.Solver()
        solver.set(random_seed=0)
        for witness in obstruction["witnesses"]:
            u = witness["input"]
            corrected = u
            for j, c in enumerate(self.baseline):
                if (u & c).bit_count() & 1:
                    corrected ^= 1 << j
            solver.add(clause_for_input(corrected, self.variables))
        solver.add(*[z3.Not(self.variables[e]) for e in obstruction["locked_coordinates"]])
        return solver

    def release(self, obstruction, batch=4):
        """Remove core locks, preferring those affecting more core witnesses."""
        witnesses = obstruction["witnesses"]
        candidates = sorted(obstruction["locked_coordinates"], key=lambda e: (
            -sum((w["input"] >> e[1]) & 1 for w in witnesses), e[0], -e[1]))
        chosen = candidates[:batch]
        self.active.update(chosen)
        return chosen

    def columns(self):
        model = self.solver.model()
        columns = list(self.baseline)
        for (j, i), v in self.variables.items():
            if z3.is_true(model.eval(v, model_completion=True)):
                columns[j] ^= 1 << i
        return columns

class FeedbackTrial(Trial):
    """Extend the existing recursive constructor without historical templates."""

    def build(self, n, profile, distance, folder):
        if not profile:
            self.record(folder.name, dict(status="zero_dimensional_constituent"))
            return [0] * n
        if n >= 32 and min(1 << i.bit_count() for i in profile) < distance:
            _, edges = domain(n, profile, distance, "initial")
            right = tuple(i - n // 2 for i in profile if i >= n // 2)
            if (right and min(1 << i.bit_count() for i in right) >= (distance + 1) // 2
                    and self.remaining() > 0):
                child = self.projected(n, profile, [0] * n, edges, distance,
                                       folder / "initial_projected")
                if child is not None:
                    return child
            active = tuple(sorted({j for j, _ in edges}))
            if n <= 128 and 8 < len(active) <= 12 and self.remaining() > 0:
                child = self.direct_patterns(n, profile, distance, active, edges,
                                             folder / "released_row_patterns")
                if child is not None:
                    return child
        return super().build(n, profile, distance, folder)

    def direct_patterns(self, n, profile, distance, rows, edges, folder):
        """Complete released-row catalogue with an explicit representation guard."""
        folder.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        try:
            mother = tuple(sorted(set(profile) | set(rows)))
            words, metadata = enumerate_words(n, mother, [0] * n, distance - 1, self.args.work)
        except WorkLimitError:
            self.record(folder.name, dict(status="oracle_work_limit"))
            return None
        sources = tuple(sorted({i for _, i in edges}))
        source_mask = sum(1 << i for i in sources)
        scopes = {u & source_mask for u, _ in words
                  if (u & -u).bit_length() - 1 in profile}
        width = len(rows)
        estimated_bytes = len(scopes) * (1 << width) * (((1 << width) + 7) // 8 + 40)
        if estimated_bytes > 256 * 1024 * 1024:
            self.record(folder.name, dict(status="pattern_memory_guard", estimated_bytes=estimated_bytes))
            return None
        index = {j: q for q, j in enumerate(rows)}
        raw = SimpleNamespace(design=Design(n, profile), columns=[0] * n, rows=rows,
                              domain=tuple((index[j], i) for j, i in edges), words=words)
        problem = SourcePatternProblem(raw, max_width=12)
        write(folder / "model.json", dict(statistics=problem.statistics(),
              enumeration=metadata, estimated_translation_bytes=estimated_bytes))
        search = PatternSearch(problem, "reverse", "product")
        end = time.perf_counter() + self.remaining()
        while search.status == "pending" and search.nodes < self.args.nodes and time.perf_counter() < end:
            search.step()
        write(folder / "frontier.json", search.snapshot())
        result = dict(status=search.status, nodes=search.nodes,
                      seconds=time.perf_counter() - start,
                      scope="Initial released-row coefficient domain", complete_constraints=True)
        write(folder / "result.json", result)
        self.record(folder.name, result)
        if search.solution is None:
            return None
        if problem.survivors(search.solution):
            raise AssertionError("A released-row low-weight word survived")
        columns = [0] * n
        for j, delta in zip(rows, problem.deltas(search.solution)):
            columns[j] = delta
        return columns

    def dense(self, n, profile, distance, folder):
        # Existing dense word catalogues remain useful for small inner blocks.
        return super().dense(n, profile, distance, folder)

    def projected(self, n, profile, baseline, edges, distance, folder):
        rows = tuple(sorted({j for j, _ in edges}))
        index = {j: q for q, j in enumerate(rows)}
        pairs = tuple((index[j], i) for j, i in edges)
        if (not rows or len(rows) > 12 or
                any(j < n // 2 or i >= n // 2 for j, i in edges)):
            return None
        folder.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            right_profile = tuple(i - n // 2 for i in profile if i >= n // 2)
            right_columns = [c >> (n // 2) for c in baseline[n // 2:]]
            if right_profile:
                bad, _ = enumerate_words(n // 2, right_profile, right_columns,
                                         (distance - 1) // 2, self.args.work)
                if bad:
                    self.record(folder.name, dict(status="right_constituent_distance_failure",
                        scope="Fixed right constituent only; projected outer constraints cannot repair it"))
                    return None
            problem, left = outer_pattern_problem(
                profile, baseline, rows, pairs, distance,
                lambda event: print(event, flush=True))
        except ValueError as error:
            self.record(folder.name, dict(status="representation_not_applicable", detail=str(error)))
            return None
        except WorkLimitError:
            self.record(folder.name, dict(status="oracle_work_limit"))
            return None
        write(folder / "model.json", dict(statistics=problem.statistics(), left_words=len(left),
              directory_seconds=time.perf_counter() - started, rows=rows, coordinates=edges))
        search = PatternSearch(problem, "reverse", "product")
        end = time.perf_counter() + self.remaining()
        while search.status == "pending" and search.nodes < self.args.nodes and time.perf_counter() < end:
            search.step()
        write(folder / "frontier.json", search.snapshot())
        result = dict(status=search.status, nodes=search.nodes,
                      seconds=time.perf_counter() - started,
                      scope="Fixed constituents and listed outer coefficient coordinates",
                      complete_constraints=True)
        write(folder / "result.json", result)
        self.record(folder.name, result)
        if search.solution is None:
            return None
        if problem.survivors(search.solution):
            raise AssertionError("Incomplete pattern elimination")
        columns = list(baseline)
        for j, delta in zip(rows, problem.deltas(search.solution)):
            columns[j] ^= delta
        return columns

    def lazy(self, n, profile, baseline, edges, distance, folder):
        # Dispatch the complete projected model before lazy word rejection.
        # For an outer stage, first try the retained-core subset of its edges.
        _, initial = domain(n, profile, distance, "initial")
        retained = tuple(e for e in initial if e in set(edges))
        child = self.projected(n, profile, baseline, retained, distance, folder / "patterns")
        if child is not None:
            return child
        return self.locked_search(n, profile, baseline, edges, distance, folder)

    def locked_search(self, n, profile, baseline, edges, distance, folder):
        folder.mkdir(parents=True, exist_ok=True)
        system = LockedCoefficientSystem(n, profile, edges, baseline)
        write(folder / "domain.json", dict(N=n, profile=profile, baseline=baseline,
              initial_coordinates=edges, complete_causal_coordinates=system.edges, target=distance))
        columns, trace, expansions = list(baseline), [], []
        end = time.perf_counter() + self.remaining()
        status, child = "budget_exhausted", None
        try:
            for iteration in range(self.args.iterations):
                if time.perf_counter() >= end:
                    break
                words, meta = enumerate_words(n, profile, columns, distance - 1, self.args.work)
                event = dict(iteration=iteration, spectrum=meta["spectrum"], bad=len(words),
                             active_coefficients=len(system.active))
                trace.append(event)
                write(folder / "trace.json", trace)
                print(event, flush=True)
                if not words:
                    child, status = columns, "verified_feasible"
                    write(folder / "word_verification.json", meta)
                    break
                for u, w in words:
                    system.add(u, w)
                answer = z3.unknown
                while time.perf_counter() < end:
                    answer = system.check(end - time.perf_counter())
                    if answer == z3.sat:
                        columns = system.columns()
                        break
                    if answer == z3.unknown:
                        status = "solver_unknown"
                        break
                    obstruction = system.obstruction()
                    standalone = system.standalone_obstruction(obstruction)
                    standalone.set(timeout=2000)
                    rechecked = standalone.check()
                    if rechecked == z3.sat:
                        raise AssertionError("UNSAT core failed independent reconstruction")
                    obstruction["standalone_recheck"] = str(rechecked)
                    (folder / f"obstruction_{len(expansions):03d}.smt2").write_text(
                        standalone.to_smt2(), encoding="utf-8")
                    write(folder / f"obstruction_{len(expansions):03d}.json", obstruction)
                    if not obstruction["locked_coordinates"]:
                        status = "full_causal_infeasible"
                        break
                    if len(expansions) >= self.args.expansions:
                        status = "selected_domain_unsat_expansion_limit"
                        break
                    released = system.release(obstruction, self.args.release_batch)
                    feedback = dict(kind="core_lock_release", released=released,
                        witness_indices=obstruction["witness_indices"],
                        core_locked_coordinates=obstruction["locked_coordinates"],
                        global_profile_exclusion=False, feasibility_after_release_not_guaranteed=True)
                    expansions.append(feedback)
                    write(folder / "feedback.json", expansions)
                    print(feedback, flush=True)
                if answer != z3.sat:
                    break
        except WorkLimitError:
            status = "oracle_work_limit"
        write(folder / "witnesses.json", system.witnesses)
        write(folder / "active_coordinates.json", sorted(system.active))
        (folder / "constraints.smt2").write_text(system.solver.to_smt2(), encoding="utf-8")
        result = dict(status=status, variables=len(system.active), complete_variables=len(system.edges),
                      iterations=len(trace), expansions=expansions,
                      full_profile_infeasible=status == "full_causal_infeasible",
                      scope="Shared full causal coefficient system with explicit domain locks")
        write(folder / "result.json", result)
        self.record(folder.name, result)
        if status == "full_causal_infeasible":
            raise ProvedSuffixInfeasible(n, profile, distance, folder)
        return child

def trial_options(output, seconds=10, total_seconds=40, work=3_000_000):
    return SimpleNamespace(output=Path(output), seconds=seconds, total_seconds=total_seconds,
                           nodes=100_000, work=work, iterations=200, expansions=12, release_batch=4)
