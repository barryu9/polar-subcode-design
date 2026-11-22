"""Bounded two-exchange refinement with unchanged polar-row-weight counts.

Candidates are generated from the current design and GA costs only. No
reference profile or coefficient is used to prioritize the proposals.
"""

import argparse
from collections import Counter
import contextlib
from itertools import combinations
import math
from pathlib import Path
import time
from types import SimpleNamespace
from audit_general_candidate_profiles import audit
from refine_n64_shared_suffixes import checked_support
from run_reference_domain_sweep import write
from run_rm_ga_profile_polish import read, reliability_cost
from structure_feedback_profile_design import protected_trial

def proposals(n, profile, errors):
    chosen = set(profile)
    baseline = reliability_cost(profile, errors)
    costs = {i: -math.log1p(-errors[i]) for i in range(n)}
    additions = {}
    for pair in combinations(sorted(set(range(n)) - chosen), 2):
        key = tuple(sorted(i.bit_count() for i in pair))
        additions.setdefault(key, []).append(pair)
    records = []
    for removed in combinations(sorted(chosen), 2):
        key = tuple(sorted(i.bit_count() for i in removed))
        for added in additions.get(key, ()):
            delta = sum(costs[i] for i in added) - sum(costs[i] for i in removed)
            if delta >= -1e-14:
                continue
            candidate = sorted(chosen - set(removed) | set(added))
            records.append(dict(profile=candidate, removed=removed, added=added,
                GA_cost=baseline + delta, GA_proxy=-math.expm1(-(baseline + delta))))
    return sorted(records, key=lambda r: (r["GA_cost"], r["removed"], r["added"]))

def run(args):
    data, ga = read(args.seed), read(args.reliability)
    n, profile = data["N"], tuple(data["profile"])
    if (n, len(profile)) != (ga["N"], ga["K"]):
        raise ValueError("Mismatched GA inputs")
    errors = {r["index"]: r["bit_error"] for r in ga["channels"]}
    candidates = proposals(n, profile, errors)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", dict(seed=str(args.seed.resolve()), target=args.target,
        baseline_GA_proxy=-math.expm1(-reliability_cost(profile, errors)),
        reference_profile_read=False, reference_support_read=False,
        preserves_row_weight_multiset=True, constructor="protected_trial from zero",
        candidate_count=len(candidates), candidates=candidates))
    start = time.perf_counter()
    outcomes, successes = [], []
    config = SimpleNamespace(work=args.work, support_seconds=args.seconds,
        nodes=50000, direct_rows=8, radius_limit=3, pattern_outer=True)
    with (args.output / "progress.log").open("w", encoding="utf-8") as log:
        def emit(kind, **kwargs):
            import json
            log.write(json.dumps(dict(kind=kind, **kwargs)) + "\n")
            log.flush()
        for ordinal, item in enumerate(candidates[:args.candidates]):
            folder = args.output / f"candidate_{ordinal:04d}"
            checked = audit(n, item["profile"], args.target, False)
            write(folder / "necessary_check.json", checked)
            if checked["target_status"] == "excluded":
                result = dict(status="excluded", full_profile_excluded=True)
            else:
                with contextlib.redirect_stdout(log):
                    result = protected_trial(n, tuple(item["profile"]), args.target,
                                             args.output, config, emit)
            record = dict(**item, ordinal=ordinal, result=result)
            if result["status"] == "verified":
                support = checked_support(n, item["profile"], result["columns"], args.target, args.work)
                write(folder / "support.json", support)
                record["support"] = str((folder / "support.json").resolve())
                successes.append(record)
            outcomes.append(record)
            write(args.output / "outcomes.json", outcomes)
            print(f"CANDIDATE {ordinal} {item['removed']} -> {item['added']} "
                  f"proxy={item['GA_proxy']:.9f} {result['status']}", flush=True)
            if len(successes) >= args.outputs:
                break
    summary = dict(N=n, K=len(profile), target=args.target,
        baseline_GA_proxy=-math.expm1(-reliability_cost(profile, errors)),
        candidate_count=len(candidates), checked=len(outcomes),
        statuses=dict(Counter(r["result"]["status"] for r in outcomes)),
        best=successes[0] if successes else None, seconds=time.perf_counter() - start,
        global_or_local_optimum_proved=False)
    write(args.output / "summary.json", summary)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--reliability", type=Path, required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=96)
    p.add_argument("--outputs", type=int, default=1)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--work", type=int, default=30000000)
    run(p.parse_args())
