"""Profile-only deterministic domain selection and exact support completion.

The first domain releases frozen rows whose polar row weight is at least the
unprecoded distance. High-weight information rows are retained as a pure core.
No guard layout, checkpoint coefficients, or reference code is read.
"""

import json
import time
from block_neighborhood import BlockCatalogue
from compressed_source_search import PatternSearch, SourcePatternProblem
from run_manuscript_n256_extension import write
from run_manuscript_n256_reduced_joint import verify
from support_graph_policy import Design
from external_low_weight_spectrum import WorkLimitError

def initial_structure(n, profile, distance):
    profile = tuple(sorted(profile))
    if (n < 8 or n & (n-1) or len(set(profile)) != len(profile)
            or not profile or any(i < 0 or i >= n for i in profile)
            or not 1 <= distance <= n):
        raise ValueError("Invalid length, profile, or distance")
    baseline = min(1 << i.bit_count() for i in profile)
    rows = tuple(j for j in range(profile[0]+1, n)
                 if j not in profile and (1 << j.bit_count()) >= baseline)
    core = tuple(i for i in profile if (1 << i.bit_count()) >= distance)
    coordinates = tuple((q, i) for q, j in enumerate(rows)
                        for i in profile if i < j and i not in core)
    return dict(baseline=baseline, rows=rows, core=core,
                coordinates=coordinates)

class CoordinateCatalogue:
    """Keep the exact word catalogue while restricting coefficient coordinates."""
    def __init__(self, raw, coordinates):
        self.raw = raw
        self.design, self.columns, self.rows = raw.design, raw.columns, raw.rows
        self.cutoff, self.words = raw.cutoff, raw.words
        self.domain = tuple(coordinates)
        if len(set(self.domain)) != len(self.domain) or not set(self.domain) <= set(raw.domain):
            raise ValueError("Invalid restricted coordinates")
        self.masks = [sum(1 << i for q, i in self.domain if q == k)
                      for k in range(len(self.rows))]

    def child(self, deltas):
        if len(deltas) != len(self.rows) or any(
                value < 0 or value & ~mask for value, mask in zip(deltas, self.masks)):
            raise ValueError("Update leaves the selected domain")
        return self.raw.child(deltas)

def complete_domain(n, profile, columns, rows, coordinates, distance, folder, args):
    resume = getattr(args, "resume", False)
    folder.mkdir(parents=True, exist_ok=resume)
    start = time.perf_counter()
    identity = json.loads(json.dumps(dict(N=n, profile=profile, columns=columns,
        rows=rows, coordinates=coordinates, distance=distance)))
    identity_path = folder / "input.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Saved domain belongs to different inputs")
    candidate_path = folder / "unverified_candidate.json"
    if resume and identity_path.exists() and candidate_path.exists():
        saved = json.loads(candidate_path.read_text())
        result = json.loads((folder / "solver_result.json").read_text())
        return verify_domain_candidate(n, profile, saved["columns"], distance,
                                       folder, args, result)
    write(identity_path, identity)
    raw = BlockCatalogue(Design(n, profile), columns, rows, distance-1,
                         args.work, causal_leaders_only=True)
    cat = CoordinateCatalogue(raw, coordinates)
    problem = SourcePatternProblem(cat)
    write(folder/"model.json", problem.statistics())
    frontier = folder / "frontier.json"
    search = (PatternSearch.restore(problem, json.loads(frontier.read_text()))
              if resume and frontier.exists() else PatternSearch(problem, "reverse", "product"))
    node_limit = search.nodes + args.nodes
    tick = time.perf_counter()
    last = tick
    while (search.status == "pending" and search.nodes < node_limit
           and time.perf_counter()-tick < args.seconds):
        search.step()
        if time.perf_counter()-last >= 10:
            print(json.dumps(dict(kind="search", domain=str(folder), nodes=search.nodes,
                                  deepest=search.deepest)), flush=True)
            write(folder/"frontier.json", search.snapshot())
            last = time.perf_counter()
    write(folder/"frontier.json", search.snapshot())
    result = dict(status=search.status, nodes=search.nodes, deepest=search.deepest,
                  seconds=time.perf_counter()-start,
                  unresolved_is_not_infeasible=search.status == "pending",
                  statistics=problem.statistics())
    child = None
    if search.solution is not None:
        if problem.survivors(search.solution):
            raise AssertionError("Invalid source-pattern solution")
        child = cat.child(problem.deltas(search.solution))
        write(folder/"solver_result.json", result)
        write(candidate_path, dict(N=n, profile=profile, columns=child, target=distance))
        return verify_domain_candidate(n, profile, child, distance, folder, args, result)
    write(folder/"result.json", result)
    print(json.dumps(dict(kind="domain_completed", **result)), flush=True)
    return child, result

def verify_domain_candidate(n, profile, child, distance, folder, args, result):
    """Keep a solved domain separate from its independent verification budget."""
    result = dict(result)
    work = getattr(args, "verification_work", args.work)
    tick = time.perf_counter()
    try:
        checked = verify(n, profile, child, distance-1, work, both=True)
    except WorkLimitError as error:
        result.update(status="verification_pending", solver_status="sat",
                      reason=str(error), verification_work=work,
                      candidate_path=str(folder/"unverified_candidate.json"))
        write(folder/"result.json", result)
        return None, result
    if checked["cpp"]["spectrum"] != {0: 1}:
        raise AssertionError("Complete verification failed")
    write(folder/"verified.json", checked)
    result.update(status="sat", solver_status="sat", distance_lower_bound=distance,
                  verification_seconds=time.perf_counter()-tick, verification_work=work)
    write(folder/"result.json", result)
    print(json.dumps(dict(kind="domain_completed", **result)), flush=True)
    return child, result
