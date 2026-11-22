"""Bounded GA-improving exchanges using the structural construction workflow.

Every candidate starts its expression construction from zero. Local cache
reuse is confined to this run. Restricted failure leaves profile feasibility
undecided, and the best previously verified design is always retained.
"""

from collections import Counter
import hashlib
import math
from pathlib import Path
import time
from audit_general_candidate_profiles import audit
from run_reference_domain_sweep import write
from run_rm_ga_profile_polish import read, reliability_cost, swaps
from structure_feedback_profile_design import protected_trial

def run(args):
    seed, ga = read(args.seed), read(args.reliability)
    n, profile = seed["N"], tuple(seed["profile"])
    if (ga["N"], ga["K"]) != (n, len(profile)):
        raise ValueError("The reliability data do not match the code parameters")
    proof = seed.get("verification", seed)
    lower = proof.get("distance_lower_bound", proof.get("dmin",
                     seed.get("distance_lower_bound", seed.get("dmin", 0))))
    if lower is None or lower < args.target:
        raise ValueError("A verified feasible seed is required")
    if "channels" in ga:
        errors = {row["index"]: row["bit_error"] for row in ga["channels"]}
        design_snr = ga["design_EbN0_dB"]
    else:
        errors = dict(enumerate(ga["pe"]))
        design_snr = ga["GA_design_EbN0_dB"]
    if set(errors) != set(range(n)) or any(not 0 <= p < 1 for p in errors.values()):
        raise ValueError("Expected one valid GA bit-error estimate per position")
    baseline = reliability_cost(profile, errors)
    floor = args.minimum_row_weight or min(1 << i.bit_count() for i in profile)
    domain = range(n // 2, n) if args.side == "right" else range(n)
    candidates = swaps(profile, errors, domain, domain, minimum_row_weight=floor,
                       improvement_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", dict(
        N=n, K=len(profile), target=args.target, seed=str(args.seed.resolve()),
        seed_sha256=hashlib.sha256(args.seed.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        construction_sha256=hashlib.sha256(Path(__file__).with_name(
            "structure_feedback_profile_design.py").read_bytes()).hexdigest(),
        GA_proxy=-math.expm1(-baseline), design_EbN0_dB=design_snr,
        reliability_source=str(args.reliability.resolve()),
        reliability_sha256=hashlib.sha256(args.reliability.read_bytes()).hexdigest(),
        side=args.side, minimum_row_weight=floor, trial_limit=args.trials,
        support_seconds=args.support_seconds, nodes=args.nodes,
        direct_rows=args.direct_rows, radius_limit=args.radius_limit, work=args.work,
        pattern_outer=getattr(args, "pattern_outer", False),
        joint_coefficients=getattr(args, "joint_coefficients", 0),
        candidates=candidates, initialization="zero expressions for each candidate",
        random_sampling=False, global_optimality_claim=False))
    started = time.perf_counter()
    outcomes, accepted = [], []
    attempted = 0
    with (args.output / "events.jsonl").open("w", encoding="utf-8") as log:
        import json

        def emit(kind, **values):
            row = dict(kind=kind, elapsed=time.perf_counter() - started, **values)
            line = json.dumps(row)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

        for ordinal, candidate in enumerate(candidates):
            if attempted >= args.trials:
                break
            folder = args.output / f"candidate_{ordinal:04d}"
            checked = audit(n, candidate["profile"], args.target, False)
            write(folder / "profile.json", dict(N=n, K=len(profile), **candidate))
            write(folder / "necessary_check.json", checked)
            if checked["target_status"] == "excluded":
                result = dict(status="excluded", full_profile_excluded=True,
                              scope="Proved necessary condition")
            else:
                attempted += 1
                emit("candidate_trial", ordinal=ordinal, attempt=attempted,
                     removed=candidate["removed"], added=candidate["added"],
                     GA_proxy=candidate["GA_proxy"])
                result = protected_trial(n, tuple(candidate["profile"]), args.target,
                                         args.output, args, emit)
                if result["status"] == "verified":
                    if result["verification"]["distance_lower_bound"] < args.target:
                        raise AssertionError("Candidate has no required distance guarantee")
                    write(folder / "support.json", dict(
                        N=n, K=len(profile), profile=candidate["profile"], target=args.target,
                        columns=result["columns"], verification=result["verification"],
                        GA_proxy=candidate["GA_proxy"], initialization="zero"))
                    accepted.append(dict(**candidate, folder=str(folder.resolve())))
            write(folder / "result.json", result)
            outcomes.append(dict(**candidate, folder=str(folder.resolve()),
                                 result=result, accepted=result["status"] == "verified"))
            write(args.output / "outcomes.json", outcomes)
            emit("candidate_result", ordinal=ordinal, status=result["status"],
                 full_profile_excluded=result.get("full_profile_excluded", False))
    best = min(accepted, key=lambda row: row["GA_cost"]) if accepted else None
    summary = dict(N=n, K=len(profile), target=args.target,
        baseline_GA_proxy=-math.expm1(-baseline), candidate_count=len(candidates),
        necessary_checks=len(outcomes), construction_trials=attempted,
        statuses=dict(Counter(row["result"]["status"] for row in outcomes)),
        accepted=len(accepted), best=best, seconds=time.perf_counter() - started,
        best_verified_support=str(Path(best["folder"]) / "support.json") if best
                              else str(args.seed.resolve()),
        global_optimality_claim=False, all_profile_domains_exhausted=False)
    write(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "best"}))
