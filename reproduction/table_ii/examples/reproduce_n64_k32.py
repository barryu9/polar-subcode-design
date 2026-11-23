"""Reproduce structural feedback, GA exchanges, and spectral/prefix profile feedback."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from independent_reproduction import Experiment, read

def main():
    run = Experiment(64, 32, 12, __doc__)
    run.ga()
    if run.smoke:
        return
    # Start with dynamic frozen expressions for the distance-12 target.
    # The next stage uses their information set as the starting point for
    # exchanges ranked with the GA channel reliabilities.
    suite = run.root / "structural_start"
    run.call("structural_start", "run_manuscript_n64_suite.py", [
        "--output", suite, "--ks", 32, "--snr", 2.0, "--seconds", 10,
        "--construction-trials", 256, "--refinement-rounds", 3,
        "--refinement-trials", 200, "--work", 300000000])
    seed = suite / "n64_k32_d12/final_support.json"
    exchanges = run.root / "ga_two_exchanges"
    run.call("ga_two_exchanges", "refine_profile_two_swaps.py", [
        "--seed", seed, "--reliability", suite / "n64_k32_d12/ga_reliability.json",
        "--target", 12, "--output", exchanges, "--candidates", 1000,
        "--outputs", 1, "--seconds", 30, "--work", 300000000])
    winner = Path(read(exchanges / "summary.json")["best"]["support"])
    spectral = run.root / "spectral_profile_feedback"
    run.call("spectral_profile_feedback", "run_n64_spectral_profile_feedback.py", [
        "--seed", winner, "--reliability", run.root / "ga_2.25.json",
        "--output", spectral, "--rounds", 2, "--per-round", 24,
        "--anchors", 8, "--seconds", 3600, "--support-seconds", 30,
        "--refine-trials", 96, "--refine-seconds", 30, "--work", 30000000,
        "--skip-polish"])
    archive = read(spectral / "archive.json")
    # GA_cost scores the information set. A_target counts codewords at weight
    # 12. Among profiles within the cost budget, continue from the best on
    # each measure (the two choices can be the same code).
    early = [entry for entry in archive if entry["depth"] <= 2]
    lowest_cost = min(entry["GA_cost"] for entry in early)
    admissible = [entry for entry in early if entry["GA_cost"] <= 1.3 * lowest_cost]
    parents = list({entry["name"]: entry for entry in (
        min(admissible, key=lambda e: (e["GA_cost"], e["A_target"])),
        min(admissible, key=lambda e: (e["A_target"], e["GA_cost"])))}.values())
    sources = [Path(entry["code_path"]) for entry in parents]
    from cold_case_feedback import explore_profiles
    sources += explore_profiles(run, sources, rounds=1, per_round=24)
    run.freeze(sources)

if __name__ == "__main__":
    main()
