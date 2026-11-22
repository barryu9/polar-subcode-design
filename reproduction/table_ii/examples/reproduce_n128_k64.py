"""Reproduce class alternatives, core release, and distance-preserving refinement."""

from itertools import islice
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from independent_reproduction import Experiment, read, write

def main():
    run = Experiment(128, 64, 16, __doc__)
    ga = run.ga()
    if run.smoke:
        return
    from ranked_class_profiles import ranked_class_profiles
    from run_diverse_manuscript_profiles import necessary_solver, membership_check
    costs = read(ga)["costs"]
    solver, variables, _ = necessary_solver(128, 64, 16, 8)
    proposals = []
    with run.stage("ranked_class_profile_prechecks"):
        # Only these floor-layer sizes enter the construction queue.
        counts = (1, 2)
        for count in counts:
            for bucket in range(8):
                for ordinal, row in enumerate(islice(ranked_class_profiles(
                        128, 64, 8, count, bucket, costs), 4), 1):
                    if row["cost"] > 0.32:
                        break
                    status = membership_check(solver, variables, row["profile"], 3000)
                    proposals.append(dict(profile=row["profile"], cost=row["cost"],
                        count=count, bucket=bucket, ordinal=ordinal, status=status))
        write(run.root / "all_profile_prechecks.json", proposals)
    eligible = [row for row in proposals if row["status"] == "sat"]
    if not eligible:
        raise RuntimeError("No profile passed the declared necessary checks")
    # Explore the two smallest floor-layer sizes, including all class alternatives.
    candidates = sorted(eligible,
                        key=lambda row: (row["cost"], row["bucket"], row["ordinal"]))
    write(run.root / "construction_candidates.json", candidates)
    sources = []
    for index, row in enumerate(candidates):
        current = run.construct(row["profile"], f"construct_{index:03d}", joint=256)
        if current is None:
            continue
        sources.append(current)
        # Fixed query counts remove dependence on machine-specific time cutoffs.
        for round_index, queries in enumerate((128, 294, 2000), 1):
            current = run.ordered(current, f"core_{index:03d}_{round_index}", "core", queries)
            sources.append(current)
            if read(current.parent / "summary.json")["resource_stopped"]:
                break
        else:
            current = run.ordered(current, f"floor_{index:03d}", "floor", 5000)
            sources.append(current)
    run.freeze(sources)

if __name__ == "__main__":
    main()
