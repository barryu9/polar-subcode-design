"""Attribute Table II replay wall time to measured construction operations."""

import argparse
import csv
import json
from pathlib import Path
import re
from collections import defaultdict

CATEGORIES = {
    "design_input": "V-C | GA inputs and separately timed candidate selection",
    "profile_construction": "V-A/B/E | Profile search and feasible construction",
    "spectral_refinement": "V-D | Fixed-profile spectral refinement",
    "prefix_protection": "V-D | Prefix protection",
    "joint_refinement": "V-D | Joint spectrum and decoding feedback",
    "composition": "V-D | Composition and construction-time verification",
    "overhead": "Checkpoint import and process time outside timed stages",
}

REFINEMENT = {"spectral_refinement", "prefix_protection", "joint_refinement"}

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def split_time(seconds, measured, measured_category, remaining_category):
    if not 0 <= measured <= seconds:
        raise ValueError(f"Nested time {measured} s exceeds stage time {seconds} s")
    return [(measured_category, measured), (remaining_category, seconds - measured)]

def classify(case_path, stage, seconds):
    if stage.startswith("channel_input_") or stage == "design_sample_scl_selection":
        return [("design_input", seconds)]
    if stage == "spectral_profile_feedback":
        trials = case_path / stage / "trials"
        records = list(trials.glob("*/refinement.json"))
        if not records:
            raise ValueError(f"No nested refinement timings in {trials}")
        refinement = sum(read(path)["seconds"] for path in records)
        # This stage includes both profile search and spectral refinement.
        # The nested timers isolate refinement within the parent interval.
        return split_time(seconds, refinement, "spectral_refinement", "profile_construction")
    if stage in ("exchange_three_positions", "exchange_four_positions"):
        summary = read(case_path / stage / "summary.json")
        descent = summary["seconds"] - summary["initial_phase_seconds"]
        return split_time(seconds, descent, "spectral_refinement", "profile_construction")
    if stage.startswith(("joint_feedback_", "seed_feedback_", "feedback_exchange_")):
        return [("joint_refinement", seconds)]
    if stage.startswith(("core_", "floor_", "whole_rows_", "suffix_refinement_")) or stage in {
        "suffix_continuation", "fixed_profile", "suffix_row_refinement",
        "suffix_cross_refinement", "suffix_projection_refinement",
    }:
        return [("spectral_refinement", seconds)]
    if stage.startswith("prefix_") or stage in {"initial_prefix_completion", "final_prefix_completion"}:
        return [("prefix_protection", seconds)]
    if stage.startswith("compose_"):
        return [("composition", seconds)]
    if stage == "import_main_checkpoint":
        return [("overhead", seconds)]
    if stage in {
        "structural_start", "structural_exploration", "ga_two_exchanges",
        "ranked_class_profile_prechecks", "profile_and_support_exploration",
        "exchange_to_d28", "exchange_to_d32", "ga_profile_completion",
    } or re.fullmatch(r"(?:exchange_[0-9]+|construct)_[0-9]+", stage):
        return [("profile_construction", seconds)]
    raise ValueError(f"Unclassified stage: {case_path.name}/{stage}")

def rows_for_case(run, result):
    case = result["case"]
    case_path = run / case
    wall = result["construction_wall_seconds"]
    stages = read(case_path / "summary.json")["stages"]
    if result["status"] != "reproduced" or not all(s["status"] == "completed" for s in stages):
        raise ValueError(f"Use a completed reproduction for timing attribution: {case}")
    outside = wall - sum(s["seconds"] for s in stages)
    # Startup and work between stages still count toward elapsed wall time.
    # Keeping this remainder makes the per-case percentages add up to 100.
    if outside < 0:
        raise ValueError(f"Logged stages exceed process wall time: {case}")
    rows = [
        (case, s["stage"], category, seconds)
        for s in stages for category, seconds in classify(case_path, s["stage"], s["seconds"])
    ]
    rows.append((case, "unassigned_process_time", "overhead", outside))
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    results = read(args.run / "benchmark.json")["results"]
    if not results:
        raise ValueError("No completed results")
    by_case = {r["case"]: r["construction_wall_seconds"] for r in results}
    detail = [row for result in results for row in rows_for_case(args.run, result)]
    with (args.run / "stage_breakdown.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "log_stage", "paper_operation", "wall_seconds", "pct_case_wall"))
        for case, stage, category, seconds in detail:
            writer.writerow((case, stage, CATEGORIES[category], seconds, 100 * seconds / by_case[case]))
    totals = defaultdict(float)
    for case, _, category, seconds in detail:
        totals[case, category] += seconds
        totals["total", category] += seconds
    by_case["total"] = sum(by_case.values())
    with (args.run / "paper_step_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "paper_operation", "wall_seconds", "pct_case_wall"))
        for case, wall in by_case.items():
            for category, label in CATEGORIES.items():
                seconds = totals[case, category]
                writer.writerow((case, label, seconds, 100 * seconds / wall))
            refinement = sum(totals[case, category] for category in REFINEMENT)
            writer.writerow((case, "V-D | Total refinement (excludes composition)",
                             refinement, 100 * refinement / wall))

if __name__ == "__main__":
    main()
