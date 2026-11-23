"""Shared reconstruction stages for the two length-1024 Table II variants."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from independent_reproduction import Experiment, read, write

BASE_STAGE = Path("combined_02_floor16/code.json")

PREFIX_SEED = 2026100661000000

WALL_GUARD = 14400

def configure_parser(parser):
    parser.add_argument(
        "--base-archive", type=Path,
        help="Fresh main N=1024 run's frozen_archive.json or its output directory",
    )
    parser.set_defaults(defer_acceptance=True)

def prepare_inputs(run, args):
    archive = args.base_archive or run.root.parent / "n1024_k512" / "frozen_archive.json"
    if archive.is_dir():
        archive /= "frozen_archive.json"
    archive = archive.resolve()
    root = archive.parent
    with run.stage("import_main_checkpoint"):
        entries = read(archive)
        # The suffix identifies the stage even when an archived run has been
        # moved or generated on another OS. Read its output under this run root.
        selected = [e for e in entries
                    if Path(e["source"].replace("\\", "/")).parts[-len(BASE_STAGE.parts):] == BASE_STAGE.parts]
        if len(selected) != 1:
            raise ValueError(f"Expected the frozen {BASE_STAGE} candidate in {archive}")
        source = (root / "designs" / selected[0]["name"] / "code.json").resolve()
        if root / "designs" not in source.parents:
            raise ValueError(f"Base checkpoint is outside the generated archive: {source}")
        data = source.read_bytes()
        parent = json.loads(data)
        if (parent["N"], len(parent["profile"]), len(parent["columns"])) != (1024, 512, 1024):
            raise ValueError("The base archive must contain the length-1024, rate-1/2 design")
        run.base_checkpoint = run.root / "base_checkpoint.json"
        run.base_checkpoint.write_bytes(data)
    return dict(
        initial_candidate_count=1,
        base_archive=str(archive),
        base_recipe_stage=BASE_STAGE.as_posix(),
        base_checkpoint_sha256=hashlib.sha256(data).hexdigest(),
        base_checkpoint_preparation_included_in_this_run=False,
        base_checkpoint_import_included_in_this_run=True,
        search_limits="Recorded trial, query and descent limits with a nonbinding wall guard",
    )

def stage(run, source, name, script, arguments):
    output = run.root / name
    run.call(name, script, ["--source", source, "--output", output, *arguments])
    code = output / "code.json"
    if not code.is_file():
        raise RuntimeError(f"{name} did not produce a code. See {output}")
    return code

def initialize(target, description):
    return Experiment(
        1024, 512, target, description,
        configure_parser=configure_parser, prepare_inputs=prepare_inputs,
    )

def joint_refinement(run, source, name, queries, exchange=0):
    return stage(run, source, name, "refine_recursive_suffix_joint_spectrum.py", [
        "--target", 24, "--suffix-length", 256, "--design-snr", 2.25,
        "--exchange-count", exchange, "--steps", 4, "--queries", queries,
        "--seconds", WALL_GUARD, "--work", 100000000,
    ])

def prefix_completion(run, source, name, target, cutoff):
    return stage(run, source, name, "design_scl_protected_prefix.py", [
        "--target", target, "--cutoff", cutoff, "--row-floor", 16, "--max-flips", 8,
        "--frames", 512, "--snr", 1.0, "--list-size", 32, "--seed", PREFIX_SEED,
        "--work", 100000000,
    ])

def refine_baseline(run):
    continuation = stage(
        run, run.base_checkpoint, "suffix_continuation", "refine_recursive_suffix_spectrum.py",
        ["--target", 24, "--suffix-length", 256, "--trials", 1200,
         "--seconds", WALL_GUARD, "--work", 100000000],
    )
    return joint_refinement(run, continuation, "fixed_profile", 80)

def low_multiplicity(run):
    # Proposed-LM keeps the distance target 24. The information-position
    # exchange and subsequent expression choices seek a smaller A_24.
    base = refine_baseline(run)
    exchange = joint_refinement(run, base, "exchange_three_positions", 110, exchange=3)
    return prefix_completion(run, exchange, "prefix_completion", 24, 24)

def protected_exchange(run, source, target, removed):
    return stage(
        run, source, f"exchange_to_d{target}", "probe_distance_protected_profile_exchange.py",
        ["--target", target, "--snr", 2.25, "--work", 200000000, "--remove", *removed],
    )

def distance_32(run):
    base = refine_baseline(run)
    exchange = joint_refinement(run, base, "exchange_four_positions", 110, exchange=4)
    d28 = protected_exchange(run, exchange, 28, [688])
    # Proposed-D32 raises the distance target in two stages. At the final
    # target, protecting weights through 31 excludes every positive weight
    # below 32 while leaving the weight-32 multiplicity free to change.
    d32 = protected_exchange(run, d28, 32, [708, 712, 720, 736])
    prior = prefix_completion(run, d32, "initial_prefix_completion", 32, 31)
    rows = stage(run, prior, "suffix_row_refinement", "refine_suffix_layer_relative.py", [
        "--target", 32, "--suffix-length", 256, "--steps", 8, "--seconds", WALL_GUARD,
        "--work", 600000000, "--rows", 72, 80, 96, 144, 160, 192,
    ])
    cross = stage(run, rows, "suffix_cross_refinement", "refine_exact_cross_fibers.py", [
        "--target", 32, "--suffix-length", 256, "--steps", 100, "--pairs",
        "--work", 600000000, "--remove", 802, 792, 804, 808, 816,
    ])
    projected = stage(run, cross, "suffix_projection_refinement", "refine_exact_cross_fibers.py", [
        "--target", 32, "--suffix-length", 256, "--steps", 150, "--pairs",
        "--work", 600000000, "--left-rows", 24, 33, 34, 36, 40, 48, 65, 66, 68, 72, 80, 96,
    ])
    ranked = run.root / "ga_profile_completion"
    run.call("ga_profile_completion", "search_ga_profile_fixed_suffix.py", [
        "--original", prior, "--suffix-source", projected, "--output", ranked,
        "--design-snr", 2.25, "--pool-size", 8, "--candidates", 4,
        "--seconds", 120, "--a32-goal", 105000, "--work", 600000000,
    ])
    best = read(ranked / "summary.json")["best"]
    if best is None:
        raise RuntimeError(f"No ranked GA candidate completed: {ranked}")
    origin = prior.parent / "protocol.json"
    return stage(run, Path(best["code_path"]), "final_prefix_completion",
                 "run_ga_priority_d32_origin_prefix.py", [
                     "--origin-protocol", origin, "--work", 600000000,
                     "--defer-spectrum-verification",
                 ])
