"""Same-run spectrum/prefix refinement without historical seed selection."""

import math
from pathlib import Path
from types import SimpleNamespace
from independent_reproduction import read, write

def refine(experiment, source, output, length, seed, frames, snr, trials,
           *, catalogue_variants=True):
    import run_joint_scl_profile_design as joint
    from polar_code.rp_funcs import rp_GA
    from polar_code.SCL_full_decoder import simulate_awgn_llr_batch
    from run_diverse_profile_suite import identities
    from run_scaled_spectral_profile_feedback import directory
    from structure_feedback_profile_design import final_check
    from select_designs_scl import noiseless_check

    raw = read(source)
    n, k, target = experiment.n, experiment.k, experiment.target
    costs = [-math.log1p(-float(p)) for p in rp_GA(n, k, 2.25).pe]
    llr = simulate_awgn_llr_batch(n, k, snr, range(seed, seed + frames))
    archive = {}
    write(output / "protocol.json", dict(source=str(source), seed=seed, frames=frames,
        design_EbN0_dB=snr, length=length, trials=trials, seconds=14400,
        catalogue_variant_sources=catalogue_variants,
        historical_input=False, acceptance_identity_used=False,
        oracle="Complete suffix catalogue; independent full-code distance; whole layer counted after freezing"))

    def export(profile, columns, feedback, proposal, proof, depth):
        ph, sh = identities(n, profile, columns)
        name = ph[:12] + "_" + sh[:12]
        if name in archive:
            return archive[name]
        cost = sum(costs[i] for i in profile)
        entry = dict(N=n, K=k, profile=list(profile), columns=list(columns),
            name=name, profile_hash=ph, support_hash=sh, GA_cost=cost,
            GA_proxy=-math.expm1(-cost), dp=min(1 << i.bit_count() for i in profile),
            target=target, A_target=feedback["count"], source_A_target=feedback["source_A_target"],
            source_feedback_exact=feedback.get("source_A_target") is not None,
            feedback_target=target // (n // length), feedback_length=length,
            feedback_full_layer=length == n, Admin=feedback["count"] if length == n else None,
            distance_lower_bound=target, verification=proof, proposal=proposal, depth=depth)
        noiseless_check(entry, 32, "cpp")
        entry["prefix_feedback"] = joint.prefix_assessment(entry, llr)
        entry["pruning_errors"] = entry["prefix_feedback"]["pruning_errors"]
        path = output / "designs" / name / "code.json"
        entry["code_path"] = str(path)
        archive[name] = entry
        write(path, entry)
        write(output / "archive.json", list(archive.values()))
        experiment.emit("support_variant", name=name, layer_count=entry["A_target"],
                        pruning_errors=entry["pruning_errors"])
        return entry

    feedback = directory(n, raw["profile"], raw["columns"], target, length, 100000000)
    proof = final_check(n, raw["profile"], raw["columns"], target, 600000000, True)
    base = export(raw["profile"], raw["columns"], feedback,
                  dict(mode="same_run_seed", source=str(source)), proof, 0)
    joint.refine_support(base, dict(N=n, K=k, target=target, length=length, refine_seconds=14400),
        SimpleNamespace(refine_trials=trials, work=100000000, verify_work=600000000,
                        catalogue_variants=catalogue_variants),
        llr, export, output)
    return [Path(e["code_path"]) for e in archive.values()]

def explore_profiles(experiment, sources, *, rounds=3, per_round=24):
    """Explore a freshly generated N64 profile graph, retaining all variants."""
    from run_joint_scl_profile_design import mixed_proposals, representatives
    from run_diverse_manuscript_profiles import necessary_solver, membership_check
    from polar_code.rp_funcs import rp_GA
    n, k, target = experiment.n, experiment.k, experiment.target
    costs = [-math.log1p(-float(v)) for v in rp_GA(n, k, 2.25).pe]
    ceiling = 1.3 * max(sum(costs[i] for i in read(s)["profile"]) for s in sources)
    archive, seen, output_paths = [], set(), []
    for index, source in enumerate(sources):
        variants = experiment.feedback(source, f"seed_feedback_{index:02d}", n,
                                       2026100571000000, 4096, 1.75)
        archive.extend(read(path) for path in variants)
        output_paths.extend(variants)
        seen.add(tuple(read(source)["profile"]))
    solver, variables, _ = necessary_solver(n, k, target, 8)
    for round_index in range(1, rounds + 1):
        parents = representatives(archive, 5, n, target)
        for ordinal, proposal in enumerate(mixed_proposals(parents, costs, ceiling, seen, 8, target)):
            if ordinal >= per_round:
                break
            profile = proposal["profile"]
            seen.add(tuple(profile))
            name = f"exchange_{round_index}_{ordinal:03d}"
            write(experiment.root / "proposals" / f"{name}.json", proposal)
            if membership_check(solver, variables, profile, 1000) != "sat":
                continue
            folder = experiment.root / name
            source = folder / "profile.json"
            write(source, dict(N=n, K=k, target=target, profile=profile))
            experiment.call(name, "run_joint_scl_profile_design.py", [
                "--worker", "joint", "--worker-input", source, "--output", folder,
                "--work", 100000000, "--verify-work", 600000000,
                "--solve-seconds", 30, "--trial-seconds", 90,
                "--cache", experiment.root / "exchange_cache"])
            result = read(folder / "worker_result.json")
            if result["status"] != "verified":
                continue
            constructed = folder / "support.json"
            write(constructed, dict(N=n, K=k, target=target, profile=profile,
                  columns=result["columns"], verification=result["verification"]))
            variants = experiment.feedback(constructed, f"feedback_{name}", n,
                                           2026100571000000, 4096, 1.75)
            for path in variants:
                entry = read(path)
                entry["depth"] = round_index
                archive.append(entry)
            output_paths.extend(variants)
    return output_paths
