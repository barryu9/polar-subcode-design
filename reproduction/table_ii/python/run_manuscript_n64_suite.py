"""Replay structural profile design and distance-constrained refinement.

This driver reuses the manuscript-aligned construction backends. All first
designs start from zero expressions, and no reference designs are imported.
Finite trial limits do not establish global infeasibility or optimality.
"""

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from types import SimpleNamespace
from polar_code import rp_funcs
from polish_structured_profiles import run as polish
from run_manuscript_n256_reduced_joint import verify
from run_reference_domain_sweep import write
from run_rm_ga_profile_polish import canonical_check, read, reliability_cost
from structure_feedback_profile_design import identifier, run as construct

CASES = {32: 12}

class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()

def channel_data(n, k, snr):
    means = list(map(float, rp_funcs.rp_GA(n, k, snr).channel_params))
    order = sorted(range(n), key=lambda i: (-means[i], i))
    ranks = {i: q for q, i in enumerate(order)}
    return dict(N=n, K=k, design_EbN0_dB=snr,
                convention="Natural input coordinates; zero-based GA ranks",
                channels=[dict(index=i, rank=ranks[i], llr_mean=means[i],
                    bit_error=0.5 * math.erfc(math.sqrt(max(0.0, means[i])) / 2))
                    for i in range(n)])

def run_case(k, root, args):
    target, n = CASES[k], 64
    folder = root / f"n{n}_k{k}_d{target}"
    folder.mkdir()
    start = time.perf_counter()
    ga = channel_data(n, k, args.snr)
    write(folder / "ga_reliability.json", ga)
    config = dict(n=n, k=k, target=target, reliability=folder / "ga_reliability.json",
                  order="rm-ga", order_file=None, output=folder / "construction",
                  seed_seconds=60, support_seconds=args.seconds, nodes=50000,
                  direct_rows=8, radius_limit=3, work=args.work,
                  trials=args.construction_trials, focus_batch=8,
                  static_tail_feedback=True, pattern_outer=True, outputs=1, backend="auto")
    print(json.dumps(dict(kind="case_start", N=n, K=k, target=target)), flush=True)
    construct(SimpleNamespace(**config))
    summary = read(folder / "construction/summary.json")
    if not summary["outputs"]:
        result = dict(N=n, K=k, target=target, status="unresolved",
                      construction=summary, seconds=time.perf_counter() - start)
        write(folder / "summary.json", result)
        return result
    initial = summary["outputs"][0]
    initial.setdefault("N", n)
    initial.setdefault("K", k)
    write(folder / "initial_support.json", initial)
    current_path = folder / "initial_support.json"
    errors = {r["index"]: r["bit_error"] for r in ga["channels"]}
    history = []
    for iteration in range(args.refinement_rounds):
        current = read(current_path)
        plan = SimpleNamespace(seed=current_path, reliability=folder / "ga_reliability.json",
            target=target, output=folder / f"refinement_{iteration:02d}",
            side="all", minimum_row_weight=None, trials=args.refinement_trials,
            support_seconds=args.seconds, nodes=50000, direct_rows=8,
            radius_limit=3, pattern_outer=True, work=args.work)
        polish(plan)
        refined = read(plan.output / "summary.json")
        history.append(refined)
        if not refined["accepted"]:
            break
        current_path = Path(refined["best_verified_support"])
    best = read(current_path)
    tick = time.perf_counter()
    # Include the target layer, not merely the empty window below it.
    checked = verify(n, best["profile"], best["columns"], target, args.work, both=True)
    histogram = checked["cpp"]["spectrum"]
    if any(w and w < target and count for w, count in histogram.items()):
        raise AssertionError("Independent final enumeration found a forbidden codeword")
    proof = canonical_check(n, best["profile"], best["columns"], target)
    proof.update(spectrum_through_target=histogram, cpp_python_agree=True,
                 dmin=target if histogram.get(target, 0) else None,
                 distance_lower_bound=target if histogram.get(target, 0) else target + 1)
    final = dict(N=n, K=k, profile=best["profile"], columns=best["columns"],
                 target=target, verification=proof, initialization="zero",
                 reference_design_read=False)
    write(folder / "final_support.json", final)
    write(folder / "final_enumeration.json", checked)
    before = set(initial["profile"])
    after = set(best["profile"])
    result = dict(N=n, K=k, target=target, status="verified", profile_id=identifier(best["profile"]),
        dmin=proof["dmin"], Admin=histogram.get(target, 0),
        dp=min(1 << i.bit_count() for i in best["profile"]),
        seed_dp=min(1 << i.bit_count() for i in initial["profile"]),
        initial_GA_proxy=-math.expm1(-reliability_cost(initial["profile"], errors)),
        final_GA_proxy=-math.expm1(-reliability_cost(best["profile"], errors)),
        removed_from_first_feasible=sorted(before - after),
        added_to_first_feasible=sorted(after - before),
        construction_seconds=summary["seconds"],
        refinement_seconds=sum(r["seconds"] for r in history),
        verification_seconds=time.perf_counter() - tick,
        construction_trials=summary["trials"], refinement_rounds=len(history),
        refinement_construction_trials=sum(r["construction_trials"] for r in history),
        refinement_statuses=[r["statuses"] for r in history],
        final_support=str((folder / "final_support.json").resolve()),
        seconds=time.perf_counter() - start, global_optimality_claim=False)
    write(folder / "summary.json", result)
    print(json.dumps(dict(kind="case_finished", **result)), flush=True)
    return result

def run(args):
    import sys
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("structure_feedback_profile_design.py"),
               Path(__file__).with_name("polish_structured_profiles.py")]
    write(args.output / "protocol.json", dict(
        cases={k: CASES[k] for k in args.ks}, design_EbN0_dB=args.snr,
        initial_order="RM-GA", improvement_cost="Additive GA cost; smaller preferred",
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        support_initialization="zero", reference_profile_read=False, reference_support_read=False,
        random_sampling=False, finite_run=True, complete_coverage_exhausted=False,
        construction_trials=args.construction_trials, refinement_rounds=args.refinement_rounds,
        refinement_trials=args.refinement_trials, local_solver_seconds=args.seconds,
        work=args.work, python=platform.python_version(), platform=platform.platform(),
        interpretation="Finite structured construction and refinement, not an exhaustive optimum proof"))
    results = []
    with (args.output / "progress.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            for k in args.ks:
                results.append(run_case(k, args.output, args))
                write(args.output / "summary.json", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--snr", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--construction-trials", type=int, default=32)
    parser.add_argument("--refinement-rounds", type=int, default=2)
    parser.add_argument("--refinement-trials", type=int, default=8)
    parser.add_argument("--work", type=int, default=300000000)
    run(parser.parse_args())
