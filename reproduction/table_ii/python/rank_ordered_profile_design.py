"""Rank-ordered profile design with exact domains and proof-scoped feedback.

The rank oracle skips only proved necessary-condition failures. Its output
order equals (rank sum, increasing rank tuple), not greedy channel selection.
An unresolved support task suspends that profile instead of excluding it.
"""

from functools import lru_cache
import json
import time
from types import SimpleNamespace
import z3
from compressed_source_search import PatternSearch, SourcePatternProblem
from obstruction_support_search import enumerate_words
from profile_linear_bounds import basic_bound, coset_capacities
from run_manuscript_n256_reduced_joint import verify
from run_reference_domain_sweep import clause_for_input, write
from support_graph_policy import Design
from verify_causal_flag_theory import polar_transform, rref

class Unresolved(RuntimeError):
    """An incomplete computation, never an infeasibility statement."""

def causal_edges(n, profile):
    info = set(profile)
    return tuple((j, i) for j in range(n) if j not in info
                 for i in profile if i < j)

class RankOracle:
    """An exact alternative implementation of the rank-tuple heap order."""

    def __init__(self, n, k, order, target=None):
        if sorted(order) != list(range(n)) or not 0 <= k <= n:
            raise ValueError("Invalid rank order or dimension")
        self.n, self.k, self.order = n, k, tuple(order)
        self.ranks = {i: r for r, i in enumerate(order)}
        self.caps = [[n] * n for _ in range(2)]
        self.end_caps = [n] * n
        self.exclusions = []
        self.prechecks = []
        if target is not None:
            length = 1
            while length <= n:
                delta = (target * length + n - 1) // n
                cap = basic_bound(length, delta, use_table=False)
                self.end_caps[n - length] = min(self.end_caps[n - length], cap)
                self.prechecks.append(dict(length=length, target=delta, dimension_cap=cap,
                                           proof="Singleton, Hamming and Griesmer bounds"))
                capacities, complement = coset_capacities(length, delta)
                for i in range(length):
                    j = n - length + i
                    ordinary = capacities[i]
                    symmetric = min(ordinary, complement[i]) if i < length - 1 else ordinary
                    if (1 << i.bit_count()) < delta:
                        ordinary = min(ordinary, length - i - 2)
                        symmetric = min(symmetric, length - i - 2)
                    self.caps[0][j] = min(self.caps[0][j], ordinary)
                    self.caps[1][j] = min(self.caps[1][j], symmetric)
                length *= 2

    def exclude_pattern(self, start, pattern):
        """The caller must supply a proved full-support exclusion or an output."""
        selected = frozenset(pattern)
        if not 0 <= start < self.n or any(i < start or i >= self.n for i in selected):
            raise ValueError("Invalid suffix pattern")
        self.exclusions.append((start, selected))

    def next(self, seconds):
        deadline = time.perf_counter() + seconds
        ones = [sum(1 << q for q, (_, pattern) in enumerate(self.exclusions) if i in pattern)
                for i in range(self.n)]
        ending = [sum(1 << q for q, (start, _) in enumerate(self.exclusions) if start == i)
                  for i in range(self.n)]

        @lru_cache(None)
        def best(i, later, matches, last):
            if time.perf_counter() >= deadline:
                raise Unresolved("rank_selection_time_limit")
            if later > self.k or later + i + 1 < self.k:
                return None
            if i < 0:
                return (0, 0, 0) if later == self.k else None
            answer = None
            for take in (0, 1):
                selected_last = take if i == self.n - 1 else last
                if take and later > self.caps[selected_last][i]:
                    continue
                if later + take > self.end_caps[i]:
                    continue
                matching = matches & (ones[i] if take else ~ones[i])
                if matching & ending[i]:
                    continue
                child = best(i - 1, later + take, matching, selected_last)
                if child is None:
                    continue
                rank = self.ranks[i]
                value = (child[0] + take * rank,
                         child[1] | (take << (self.n - 1 - rank)),
                         child[2] | (take << i))
                # For a fixed cardinality, larger rank-membership bitmask
                # means a lexicographically smaller increasing rank tuple.
                if answer is None or (value[0], -value[1]) < (answer[0], -answer[1]):
                    answer = value
            return answer

        answer = best(self.n - 1, 0, (1 << len(self.exclusions)) - 1, 0)
        if answer is None:
            return None
        profile = [i for i in range(self.n) if answer[2] >> i & 1]
        ranks = tuple(sorted(self.ranks[i] for i in profile))
        assert len(profile) == self.k and sum(ranks) == answer[0]
        return dict(profile=profile, rank_sum=answer[0], rank_tuple=ranks,
                    rank_dp_states=best.cache_info().currsize)

def verify_support(n, profile, columns, target, work):
    info = sum(1 << i for i in profile)
    if any(c < 0 or c >> j or c & ~info or (j in profile and c)
           for j, c in enumerate(columns)):
        raise AssertionError("Noncanonical support")
    generators = [polar_transform((1 << i) | sum(((c >> i) & 1) << j
                   for j, c in enumerate(columns)), n) for i in profile]
    assert len(rref(generators)) == len(profile)
    counted = verify(n, profile, columns, target - 1, work, both=True)
    assert counted["cpp"]["spectrum"] == {0: 1}
    witness = next((dict(source=i, codeword=x, weight=target)
                    for i, x in zip(profile, generators) if x.bit_count() == target), None)
    return dict(causality=True, rank=len(profile), complete_cutoff=target - 1,
                spectrum={0: 1}, cpp_python_agree=True, witness=witness,
                distance_lower_bound=target, dmin=target if witness else None)

def solve_domain(n, profile, target, edges, folder, args, emit):
    """Complete common-supercode conditions, followed by exact verification."""
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / "result.json"
    if saved.exists():
        result = json.loads(saved.read_text())
        if result["status"] in ("sat", "unsat"):
            return result
    rows = tuple(sorted({j for j, _ in edges}))
    identity = dict(N=n, profile=profile, target=target, edges=edges)
    write(folder / "domain.json", identity)
    directory_path = folder / "directory.json"
    if directory_path.exists():
        directory = json.loads(directory_path.read_text())
        words = [(int(u), w) for u, w in directory["words"]]
    else:
        common = tuple(sorted(set(profile) | set(rows)))
        emit("directory_start", N=n, K=len(profile), target=target,
             coefficients=len(edges), supercode_dimension=len(common))
        words, metadata = enumerate_words(n, common, [0] * n, target - 1, args.work)
        words = [(u, w) for u, w in words if (u & -u).bit_length() - 1 in profile]
        write(directory_path, dict(words=[(str(u), w) for u, w in words],
                                   complete=True, metadata=metadata))
    emit("domain_ready", N=n, target=target, coefficients=len(edges),
         conditions_before_grouping=len(words), rows=len(rows))
    deadline = min(time.perf_counter() + args.support_seconds,
                   getattr(args, "deadline", float("inf")))
    backend = getattr(args, "backend", "auto")
    if backend == "patterns" and len(rows) > 8:
        return dict(status="unresolved", reason="Pattern backend supports at most eight rows")
    if backend == "patterns" or (backend == "auto" and len(rows) <= 8):
        positions = {j: b for b, j in enumerate(rows)}
        catalogue = SimpleNamespace(design=Design(n, profile), columns=[0] * n,
            rows=rows, words=words, domain=tuple((positions[j], i) for j, i in edges))
        problem = SourcePatternProblem(catalogue)
        frontier = folder / "frontier.json"
        search = (PatternSearch.restore(problem, json.loads(frontier.read_text()))
                  if frontier.exists() else PatternSearch(problem, "reverse", "product"))
        last = time.perf_counter()
        while search.status == "pending" and time.perf_counter() < deadline:
            search.step()
            if time.perf_counter() - last >= 10:
                write(frontier, search.snapshot())
                emit("source_pattern_progress", N=n, nodes=search.nodes, deepest=search.deepest)
                last = time.perf_counter()
        write(frontier, search.snapshot())
        result = dict(status=search.status, nodes=search.nodes, statistics=problem.statistics())
        columns = [0] * n
        if search.status == "sat":
            for j, delta in zip(rows, problem.deltas(search.solution)):
                columns[j] = delta
    else:
        variables = {edge: z3.Bool(f"s_{edge[0]}_{edge[1]}") for edge in edges}
        solver = z3.Solver()
        solver.set(random_seed=0, threads=1)
        row_mask = sum(1 << j for j in rows)
        source_mask = sum(1 << i for i in {i for _, i in edges})
        groups = {}
        for u, w in words:
            groups.setdefault((u & row_mask, u & source_mask), u)
        for index, u in enumerate(groups.values()):
            if index % 256 == 0 and time.perf_counter() >= deadline:
                return dict(status="unresolved", reason="Constraint encoding time limit",
                            encoded=index, total=len(groups))
            solver.add(clause_for_input(u, variables))
        (folder / "constraints.smt2").write_text(solver.to_smt2(), encoding="utf-8")
        solver.set(timeout=max(1, int((deadline - time.perf_counter()) * 1000)))
        answer = solver.check()
        result = dict(status=str(answer), grouped_conditions=len(groups),
                      reason_unknown=solver.reason_unknown() if answer == z3.unknown else None)
        columns = [0] * n
        if answer == z3.sat:
            model = solver.model()
            for (j, i), variable in variables.items():
                if z3.is_true(model.eval(variable, model_completion=True)):
                    columns[j] |= 1 << i
    if result["status"] == "sat":
        proof = verify_support(n, profile, columns, target, args.work)
        result.update(columns=columns, verification=proof)
    result.update(scope="This complete coordinate domain", coefficients=len(edges))
    write(saved, result)
    emit("domain_finished", N=n, target=target, status=result["status"], coefficients=len(edges))
    return result
