"""Deterministic structural profile proposals with proof-scoped support tasks.

Only a channel ordering is required. Restricted failures guide scheduling, not
global exclusion. Completed candidates undergo independent physical checks.
The focused phase is bounded; retained tasks describe the complete fallback.
"""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from audit_general_candidate_profiles import audit
from external_low_weight_spectrum import WorkLimitError
from rank_ordered_profile_design import RankOracle, causal_edges, solve_domain
from run_manuscript_n256_reduced_joint import verify
from run_profile_only_targets import restricted_outer, rm_gap_membership
from run_profile_target_construction import complete_outer
from run_reference_domain_sweep import write
from run_rm_ga_profile_polish import canonical_check
from select_ga_necessary_profile import base_solver, ga_priority_completion, refine
from select_structured_domain import complete_domain, initial_structure

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def identifier(profile):
    return hashlib.sha256(json.dumps(tuple(profile)).encode()).hexdigest()[:12]

def protected_edges(n, profile, target):
    if not profile:
        return ()
    baseline = min(1 << i.bit_count() for i in profile)
    return tuple((j, i) for j in range(n) if j not in profile
                 and (1 << j.bit_count()) >= baseline
                 for i in profile if i < j and (1 << i.bit_count()) < target)

def repair_neighbors(n, profile, target, length, protected_tail, ranks):
    """Preserve a verified shorter tail and prefer transfers out of a hard suffix."""
    original = set(profile)
    start = n - length
    stop = n - protected_tail
    baseline = min(1 << i.bit_count() for i in profile)
    groups = []
    if protected_tail and length < n:
        groups.append(("sibling_transfer", range(max(0, n - 2 * length), start)))
    groups.append(("within_block", range(start, stop)))
    records = []
    for mode, destinations in groups:
        batch = []
        for old in sorted(original):
            if not start <= old < stop or (1 << old.bit_count()) >= target:
                continue
            for new in destinations:
                if new in original or new.bit_count() != old.bit_count():
                    continue
                candidate = sorted(original - {old} | {new})
                assert min(1 << i.bit_count() for i in candidate) >= baseline
                batch.append(dict(profile=candidate, removed=old, added=new, mode=mode,
                    rank_change=ranks[new] - ranks[old], feedback_length=length,
                    preserved_tail=protected_tail,
                    priority_tuple=sorted(ranks[i] for i in candidate)))
        records.extend(sorted(batch, key=lambda r: (r["rank_change"], r["priority_tuple"],
                                                   r["removed"], r["added"])))
    return records

def feedback_progress(n, profile, result, proposal, static_tail_feedback=False):
    """Admit another focused batch after a longer or lower-dimensional obstacle."""
    length = result["failure_length"]
    previous = proposal["generated_at_failure"]
    dimension = sum(i >= n - length for i in profile)
    return length > previous or (
        static_tail_feedback and length == previous
        and dimension < proposal.get("parent_suffix_dimension", dimension))

def select_seed(n, k, target, order, seconds, folder, emit):
    solver, variables = base_solver(n, k, target)
    deadline = time.perf_counter() + seconds
    for iteration in range(32):
        profile, status = ga_priority_completion(solver, variables, order, deadline)
        if profile is None:
            raise RuntimeError("Seed selection unresolved: " + status)
        checked = audit(n, profile, target, False)
        write(folder / f"seed_check_{iteration:02d}.json", checked)
        emit("seed_necessary_check", iteration=iteration, status=checked["target_status"])
        if checked["target_status"] != "excluded":
            return profile
        write(folder / f"seed_refinement_{iteration:02d}.json",
              refine(solver, variables, profile, checked))
    raise RuntimeError("Seed refinement limit reached")

def final_check(n, profile, columns, target, work, prefer_relative=False):
    """Use a verified gap only when the actual code satisfies its hypotheses."""
    def relative_proof():
        from audit_cold_profile_feedback import relative_distance_check
        checked = relative_distance_check(n, profile, columns, target, work)
        if not checked["passed"]:
            raise AssertionError("Independent relative distance check failed")
        return dict(**canonical_check(n, profile, columns, target),
                    spectrum_through_cutoff={0: 1}, cutoff=target - 1,
                    cpp_python_agree=False, gap=None, physical_verification=True,
                    verification_method="complete_relative_coset_membership",
                    relative_verification=checked)

    if prefer_relative:
        try:
            return relative_proof()
        except (WorkLimitError, ValueError):
            pass
    try:
        gap = rm_gap_membership(profile, columns, target)
    except ValueError:
        gap = None
    if gap is None and not prefer_relative:
        try:
            return relative_proof()
        except (WorkLimitError, ValueError):
            pass
    cutoff = gap["minimum_weight"] if gap else target - 1
    try:
        spectrum = verify(n, profile, columns, cutoff, work, both=True)
    except WorkLimitError:
        return dict(**relative_proof(), scalar_work_limit=True)
    assert spectrum["cpp"]["spectrum"] == {0: 1}
    proof = canonical_check(n, profile, columns, target)
    return dict(**proof, spectrum_through_cutoff={0: 1}, cutoff=cutoff,
                cpp_python_agree=True, gap=gap, physical_verification=True)

def protected_trial(n, profile, target, root, args, emit):
    """Bottom-up direct protected domains, then a scoped fixed-pair completion."""
    child, nontrivial_tail = None, 0
    verification_work = getattr(args, "verification_work", args.work)
    outer_seconds = getattr(args, "outer_seconds", args.support_seconds)
    length = 1
    while length <= n:
        local = tuple(i - (n - length) for i in profile if i >= n - length)
        delta = (target * length + n - 1) // n
        key = hashlib.sha256(json.dumps((length, local, delta)).encode()).hexdigest()[:16]
        folder = root / "suffix_cache" / key
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "outcome.json"
        saved = read(path) if path.exists() else None
        retry = (getattr(args, "retry_incomplete", False) or
                 (saved and saved.get("status") == "verification_pending"
                  and verification_work > saved.get("verification_work", 0)))
        baseline = min((1 << i.bit_count() for i in local), default=length + 1)
        joint_cap = getattr(args, 'joint_coefficients', 0)
        joint_max_length = getattr(args, 'joint_max_length', n)
        joint_structure = (initial_structure(length, local, delta)
                           if joint_cap and length >= 8 and baseline < delta else None)
        joint_possible = (length <= joint_max_length and joint_structure is not None
                          and 0 < len(joint_structure['coordinates']) <= joint_cap)
        fixed_pair_cache = saved and saved.get('method') == 'fixed_pair_outer'
        pair_changed = False
        if fixed_pair_cache and child is not None:
            pair = [0] * (length // 2) + [c << (length // 2) for c in child]
            pair_changed = saved.get('fixed_pair_sha256') != hashlib.sha256(json.dumps(pair).encode()).hexdigest()
        reopen_pair = (fixed_pair_cache and saved.get('status') != 'sat'
                       and (joint_possible or pair_changed))
        if baseline >= delta:
            result = dict(status="sat", columns=[0] * length, method="Unprecoded distance",
                          distance_lower_bound=baseline)
        elif saved and not reopen_pair and (saved["status"] in ("sat", "unsat") or not retry):
            result = saved
        else:
            emit("support_trial", N=length, K=len(local), target=delta, cache=key)
            write(folder / "profile.json", dict(N=length, profile=local, target=delta))
            structure = joint_structure or (initial_structure(length, local, delta) if length >= 8 else None)
            tick = time.perf_counter()
            phase = "construction"
            candidate_paths = [folder / "unverified_candidate.json",
                               folder / "outer/unverified_candidate.json"]
            candidate_paths.extend(sorted(folder.glob("attempt_*/outer/unverified_candidate.json")))
            saved_candidate = next((p for p in candidate_paths if p.exists()), None)
            try:
                with (folder / "progress.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    if saved_candidate:
                        candidate = read(saved_candidate)
                        assert candidate["profile"] == list(local) and candidate["target"] == delta
                        columns = candidate["columns"]
                        result = candidate.get("construction_result", dict(method="saved_candidate"))
                        result.setdefault("method", "saved_candidate")
                    elif structure and 1 <= len(structure["rows"]) <= args.direct_rows:
                        config = SimpleNamespace(work=args.work, seconds=args.support_seconds,
                                                 nodes=args.nodes, resume=True,
                                                 verification_work=verification_work)
                        columns, info = complete_domain(length, local, [0] * length,
                            structure["rows"], structure["coordinates"], delta,
                            folder / "direct", config)
                        result = dict(status=info["status"], nodes=info["nodes"],
                            scope="Full retained-core coordinate domain, not full causal space",
                            method="direct_protected", coefficients=len(structure["coordinates"]))
                        if info["status"] == "verification_pending":
                            result.update(failure_kind="verification_budget", solver_status="sat",
                                          verification_work=verification_work,
                                          candidate_path=info["candidate_path"], reason=info["reason"])
                    elif joint_possible:
                        # A fixed-pair failure does not reject the joint initial domain.
                        edges = tuple((structure['rows'][q], i)
                                      for q,i in structure['coordinates'])
                        answer = solve_domain(length, local, delta, edges,
                                              folder/'joint_initial', args, emit)
                        columns = answer.get('columns')
                        result = dict(status=answer['status'],method='joint_initial_domain',
                                      scope='All coefficients in the local retained-core initial domain',
                                      coefficients=len(edges),joint_result=answer)
                    elif child is not None and length > 1:
                        # A partial outer build is not a completed cache entry.
                        attempt = folder
                        resume_outer = ((folder / 'outer/frontier.json').exists()
                                        and getattr(args, 'retry_incomplete', False))
                        if not resume_outer and any((folder / name).exists() for name in ("outer", "projected_outer")):
                            attempt = folder / ("attempt_" + str(time.time_ns()))
                            attempt.mkdir()
                        h = length // 2
                        embedded = [0] * h + [c << h for c in child]
                        write(folder / "fixed_pair.json", dict(N=length, K=len(local),
                            profile=local, columns=embedded))
                        left_min = min((1 << i.bit_count() for i in local if i < h), default=delta)
                        radius = max(0, (delta - 1 - left_min) // 2)
                        gap = None
                        columns = None
                        projected = None
                        if getattr(args, "pattern_outer", False):
                            from cold_feedback_support import FeedbackTrial, trial_options
                            options = trial_options(attempt / "projected_outer", args.support_seconds,
                                                    args.support_seconds, args.work)
                            options.nodes = args.nodes
                            solver = FeedbackTrial(options)
                            edges = tuple((j, i) for j, i in protected_edges(length, local, delta)
                                          if i < h <= j)
                            columns = solver.projected(length, local, embedded, edges, delta,
                                                       attempt / "projected_outer")
                            projected = solver.stages
                        if radius > args.radius_limit:
                            try:
                                gap = rm_gap_membership(local, embedded, delta)
                            except ValueError:
                                pass
                        if columns is not None:
                            info = read(attempt / "projected_outer/result.json")
                            outer_scope = "Fixed constituent pair and protected outer coordinates"
                        elif gap is None:
                            columns, info = complete_outer(local, embedded, delta, args.work,
                                outer_seconds, args.nodes, attempt / "outer", resume=resume_outer)
                            outer_scope = "All outer coefficients for one fixed constituent pair"
                        else:
                            # Every cross coefficient preserves the parent's row threshold.
                            assert all((1 << j.bit_count()) >= gap["minimum_weight"]
                                       for j in structure["rows"])
                            config = SimpleNamespace(profile=folder / "fixed_pair.json",
                                output=attempt / "outer", distance=delta, cutoff=gap["minimum_weight"],
                                work=args.work, seconds=args.support_seconds, nodes=args.nodes)
                            restricted_outer(config)
                            info = read(attempt / "outer/summary.json")
                            support = attempt / "outer/support.json"
                            columns = read(support)["columns"] if support.exists() else None
                            outer_scope = "One fixed constituent pair and its protected outer domain"
                        result = dict(status=info["status"], nodes=info["nodes"],
                            scope=outer_scope, method="fixed_pair_outer", gap=gap,
                            projected_backend=projected,
                            unreduced_radius=radius,
                            fixed_pair_sha256=hashlib.sha256(json.dumps(embedded).encode()).hexdigest())
                    else:
                        columns = None
                        result = dict(status="unresolved", method="direct_unsupported",
                                      scope="Implementation limit only")
                    if columns is not None:
                        write(folder / "unverified_candidate.json", dict(
                            N=length, K=len(local), profile=local, columns=columns, target=delta,
                            construction_result=result))
                        phase = "verification"
                        emit("verification_started", N=length, K=len(local), target=delta,
                             verification_work=verification_work)
                        proof = final_check(length, local, columns, delta, verification_work,
                                            getattr(args, "pattern_outer", False))
                        result.update(status="sat", columns=columns, verification=proof)
                        write(folder / "support.json", dict(N=length, K=len(local), profile=local,
                            columns=columns, verification=proof))
            except (WorkLimitError, ValueError) as error:
                result = dict(status="verification_pending" if phase == "verification" else "unresolved",
                    reason=str(error), scope="Implementation or work limit",
                    failure_kind=("verification_budget" if phase == "verification" else
                                  "enumeration_budget" if isinstance(error, WorkLimitError) else
                                  "representation_limit"), verification_work=verification_work)
            result["seconds"] = time.perf_counter() - tick
            write(path, result)
            emit("support_result", N=length, target=delta, cache=key,
                 **{k: v for k, v in result.items() if k not in ("columns", "verification", "gap")})
        if result["status"] != "sat":
            return dict(status="retained", failure_length=length, protected_tail=nontrivial_tail,
                        verified_tail=length // 2,
                        suffix_cache=key, result=result, full_profile_excluded=False)
        child = result["columns"]
        if baseline < delta:
            nontrivial_tail = length
        length *= 2
    proof = result.get("verification")
    if proof is None:
        proof = final_check(n, profile, child, target, verification_work,
                            getattr(args, "pattern_outer", False))
    return dict(status="verified", columns=child, verification=proof)

def advance_full_job(profile, n, target, folder, args, emit):
    """A retained complete-domain task, with no restricted failure promoted."""
    folder.mkdir(parents=True, exist_ok=True)
    free = set(protected_edges(n, profile, target))
    all_edges = set(causal_edges(n, profile))
    index = 0
    while True:
        answer = solve_domain(n, profile, target, tuple(sorted(free)),
                              folder / f"domain_{index:04d}", args, emit)
        if answer["status"] == "sat":
            return dict(status="verified", columns=answer["columns"], verification=answer["verification"])
        if answer["status"] != "unsat":
            return dict(status="retained", full_profile_excluded=False, result=answer)
        remaining = sorted(all_edges - free)
        if not remaining:
            return dict(status="excluded", scope="Full canonical space", full_profile_excluded=True)
        edge = remaining[0]
        write(folder / f"release_{index:04d}.json", dict(assumptions=remaining, released=edge,
              value_forced=False, proof="Complete domain rejection"))
        free.add(edge)
        index += 1

def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    n, k, target = args.n, args.k, args.target
    if args.order_file:
        order = read(args.order_file)["order"]
    else:
        raw = read(args.reliability)
        if (raw["N"], raw["K"]) != (n, k):
            raise ValueError("Channel input has another length or rate")
        means = {row["index"]: row["llr_mean"] for row in raw["channels"]}
        order = sorted(range(n), key=lambda i: ((-i.bit_count(), -means[i], i)
                                               if args.order == "rm-ga" else (-means[i], i)))
    if sorted(order) != list(range(n)):
        raise ValueError("The supplied ordering is not a permutation")
    ranks = {i: r for r, i in enumerate(order)}
    protocol = dict(N=n, K=k, target=target, order=order,
        ordering="supplied" if args.order_file else args.order,
        seed_rule="Lexicographic preference under proved necessary conditions",
        neighbor_rule="Preserve a completed shorter tail; sibling transfers before in-block exchanges",
        neighbor_priority="rank change, increasing rank tuple, removed index, added index",
        focus_batch=args.focus_batch, direct_rows=args.direct_rows, nodes=args.nodes,
        static_tail_feedback=args.static_tail_feedback,
        pattern_outer=args.pattern_outer,
        radius_limit=args.radius_limit,
        reference_profile_read=False, reference_support_read=False, known_successful_seed_read=False,
        initial_support="zero", full_profile_exclusion_requires_full_scope=True,
        global_rank_optimality_claim=False, random_sampling=False,
        evidence="Historically motivated rule, not a blind or preregistered test",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        order_input_sha256=hashlib.sha256((args.order_file or args.reliability).read_bytes()).hexdigest())
    write(args.output / "protocol.json", protocol)
    with (args.output / "events.jsonl").open("w", encoding="utf-8") as log:
        def emit(kind, **values):
            row = dict(kind=kind, elapsed=time.perf_counter() - started, **values)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)

        seed = select_seed(n, k, target, order, args.seed_seconds, args.output, emit)
        write(args.output / "seed_profile.json", dict(N=n, K=k, target=target, profile=seed))
        queue = [dict(profile=seed, generated_at_failure=0, parent=None)]
        queued = {tuple(seed)}
        retained, outputs, trials = [], [], []
        coverage = RankOracle(n, k, order, target)
        coverage.exclude_pattern(0, seed)  # Avoid duplicate admissions, not a feasibility cut.
        last_service = 0
        while len(trials) < args.trials and len(outputs) < args.outputs:
            if len(trials) - last_service >= args.focus_batch or not queue:
                last_service = len(trials)
                if retained:
                    job = retained.pop(0)
                    visits = job.get("service_visits", 0) + 1
                    job["service_visits"] = visits
                    service_args = SimpleNamespace(**(vars(args) | dict(
                        support_seconds=args.support_seconds * (1 << (visits - 1)),
                        work=args.work * (1 << (visits - 1)))))
                    emit("complete_job_service", profile_id=job["id"])
                    try:
                        outcome = advance_full_job(tuple(job["profile"]), n, target,
                            args.output / "complete_jobs" / job["id"], service_args, emit)
                    except WorkLimitError as error:
                        outcome = dict(status="retained", reason=str(error), full_profile_excluded=False)
                    job["complete_service"] = outcome
                    if outcome["status"] == "retained":
                        retained.append(job)
                    elif outcome["status"] == "verified":
                        outputs.append(dict(profile=job["profile"], **outcome))
                candidate = coverage.next(args.seed_seconds)
                if candidate:
                    p = tuple(candidate["profile"])
                    queued.add(p)
                    coverage.exclude_pattern(0, p)
                    queue.insert(0, dict(profile=list(p), generated_at_failure=0, parent=None,
                                         admission="coverage"))
                write(args.output / "state.json", dict(trials=trials, retained=retained,
                      queue=queue, outputs=outputs, distinct_profiles_admitted=len(queued)))
                if len(outputs) >= args.outputs:
                    break
            if not queue:
                candidate = coverage.next(args.seed_seconds)
                if candidate is None:
                    if retained:
                        continue
                    break
                p = tuple(candidate["profile"])
                queued.add(p)
                coverage.exclude_pattern(0, p)
                queue.append(dict(profile=list(p), generated_at_failure=0, parent=None, admission="coverage"))
            proposal = queue.pop(0)
            profile = tuple(proposal["profile"])
            key = identifier(profile)
            folder = args.output / "profiles" / key
            write(folder / "proposal.json", proposal)
            checked = audit(n, profile, target, False)
            write(folder / "necessary_check.json", checked)
            emit("profile_trial", number=len(trials) + 1, profile_id=key,
                 removed=proposal.get("removed"), added=proposal.get("added"),
                 parent=proposal.get("parent"))
            if checked["target_status"] == "excluded":
                result = dict(status="excluded", full_profile_excluded=True, scope="Proved necessary condition")
            else:
                result = protected_trial(n, profile, target, args.output, args, emit)
            if result["status"] == "verified":
                record = dict(N=n, K=k, profile=profile, target=target, **result,
                              reference_support_read=False, initialization="zero")
                write(folder / "support.json", record)
                outputs.append(record)
                emit("verified_design", profile_id=key, verification=result["verification"])
            elif result["status"] == "retained":
                retained.append(dict(id=key, profile=profile, result=result))
                if feedback_progress(n, profile, result, proposal, args.static_tail_feedback):
                    tail = (result["verified_tail"] if args.static_tail_feedback
                            else result["protected_tail"])
                    neighbors = repair_neighbors(n, profile, target, result["failure_length"],
                                                 tail, ranks)
                    admitted = []
                    for row in neighbors:
                        p = tuple(row["profile"])
                        if p in queued:
                            continue
                        queued.add(p)
                        coverage.exclude_pattern(0, p)
                        admitted.append(dict(**row, parent=key,
                            parent_suffix_dimension=sum(i >= n - result["failure_length"] for i in profile),
                            generated_at_failure=result["failure_length"]))
                    queue = admitted + queue
                    write(folder / "feedback_proposals.json", admitted)
                    emit("structural_feedback", profile_id=key, failure_length=result["failure_length"],
                         protected_tail=result["protected_tail"], new_candidates=len(admitted),
                         preserved_tail=tail,
                         original_profile_retained=True)
            write(folder / "result.json", result)
            trials.append(dict(id=key, proposal=proposal, result=result))
            write(args.output / "state.json", dict(trials=trials, retained=retained,
                  queue=queue, outputs=outputs, distinct_profiles_admitted=len(queued)))
        summary = dict(status="verified" if outputs else "incomplete", trials=len(trials),
            verified_profiles=len(outputs), retained_profiles=len(retained),
            seconds=time.perf_counter() - started, outputs=outputs,
            global_rank_optimality_claim=False, exhaustive_search_completed=False)
        write(args.output / "summary.json", summary)
        emit("finished", **{k: v for k, v in summary.items() if k != "outputs"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reliability", type=Path)
    parser.add_argument("--order", choices=("rm-ga", "ga"), default="rm-ga")
    inputs.add_argument("--order-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-seconds", type=float, default=90)
    parser.add_argument("--support-seconds", type=float, default=10)
    parser.add_argument("--nodes", type=int, default=50000)
    parser.add_argument("--direct-rows", type=int, default=8)
    parser.add_argument("--radius-limit", type=int, default=3)
    parser.add_argument("--work", type=int, default=300000000)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--focus-batch", type=int, default=64)
    parser.add_argument("--static-tail-feedback", action="store_true",
                        help="Also preserve verified static tails and continue dimension-reducing transfers")
    parser.add_argument("--pattern-outer", action="store_true",
                        help="Try the exact protected outer-pattern backend before full outer completion")
    parser.add_argument("--outputs", type=int, default=1)
    parser.add_argument("--backend", choices=("auto", "z3", "patterns"), default="auto")
    args = parser.parse_args()
    if args.n < 8 or args.n & (args.n - 1) or not 0 < args.k <= args.n or not 1 <= args.target <= args.n:
        parser.error("Require a power-of-two length, nonzero dimension, and a valid target")
    if args.focus_batch < 1 or not 1 <= args.direct_rows <= 8:
        parser.error("Invalid scheduling or direct-width limit")
    run(args)
