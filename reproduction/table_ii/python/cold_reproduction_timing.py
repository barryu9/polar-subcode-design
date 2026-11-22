"""Conservative, non-overlapping deductions from measured exploration times."""

import csv
import json
from pathlib import Path

def bank_breakdown(run):
    """Separate bank generation and support trials using its own event clock."""
    run = Path(run)
    result = []
    bank = run / "profile_exploration"
    if not bank.exists():
        return result
    for path in bank.rglob("events.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        seed = next((row for row in rows if row.get("kind") == "seeds_ready"), None)
        end = next((row for row in reversed(rows) if row.get("kind") == "case_complete"), None)
        trials = path.parent / "trials.json"
        if seed is None or end is None or not trials.exists():
            continue
        trial_rows = json.loads(trials.read_text(encoding="utf-8"))
        support = sum(float(row.get("trial_seconds", 0)) for row in trial_rows)
        elapsed = float(end["seconds"])
        generation = float(seed["seconds"])
        result.append(dict(events=str(path), profile_seed_generation_seconds=generation,
            support_trial_seconds=support, case_seconds=elapsed,
            interleaved_proposals_checks_and_bookkeeping_seconds=max(0, elapsed-generation-support),
            profile_trials=len(trial_rows), actual_cache_scope="This run only",
            caveat="Generation includes channel and necessary-check setup. Interleaved neighbor checks are not attributed to support trial time."))
    return result

def measured_exclusions(run):
    run = Path(run)
    result = []
    for path in run.rglob("trials.json"):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            if row.get("status") in ("wall_limit", "enumeration_limit", "representation_limit"):
                result.append(dict(record=str(path), index=index,
                    seconds=float(row.get("trial_seconds", 0)), status=row["status"]))
    # These directories are separate sequential stages, outside bank trial timers.
    for folder in run.iterdir():
        if not folder.is_dir() or not folder.name.startswith(("core_", "floor_", "row_refinement_", "suffix_")):
            continue
        queries = folder / "queries.json"
        events = folder / "events.jsonl"
        if queries.exists():
            rows = json.loads(queries.read_text(encoding="utf-8"))
            path = queries
        elif events.exists():
            rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
            path = events
        else:
            continue
        previous = 0.0
        for index, row in enumerate(rows):
            current = float(row.get("seconds", previous))
            duration = max(0.0, current-previous)
            previous = current
            if row.get("status") == "unknown":
                result.append(dict(record=str(path), index=index, seconds=duration,
                    status="enumeration_or_representation_limit",
                    timing_method="Difference of consecutive cumulative query times"))
    for path in run.rglob("refinement_trace.json"):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            if row.get("status") == "oracle_work_limit" and "query_seconds" in row:
                result.append(dict(record=str(path), index=index,
                    seconds=float(row["query_seconds"]), status="enumeration_limit"))
    for path in run.rglob("records.json"):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            if row.get("status") in ("unknown", "unresolved") and "seconds" in row:
                result.append(dict(record=str(path), index=index,
                    seconds=float(row["seconds"]), status=row["status"]))
    for folder in run.iterdir():
        path = folder / "worker_result.json"
        if folder.name.startswith("exchange_") and path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") == "unresolved" or (
                    row.get("status") == "retained" and row.get("reason") == "bounded_joint_completion"):
                result.append(dict(record=str(path), seconds=float(row.get("seconds", 0)),
                    status="bounded_completion_incomplete"))
    return result

def update_summary(run):
    run = Path(run)
    path = run / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    exclusions = measured_exclusions(run)
    excluded = sum(row["seconds"] for row in exclusions)
    total = summary["total_wall_seconds"]
    if excluded > total+1e-6:
        raise AssertionError("Measured deductions exceed the complete wall time")
    summary.update(excluded_attempts=exclusions, excluded_attempt_seconds=excluded,
        effective_seconds=total-excluded,
        bank_phase_breakdown=bank_breakdown(run),
        timing_caveat="Effective time is not ordinary wall time. Only individually timed resource-limited attempts are deducted. Completed unsuccessful trials and unmeasured overhead remain counted.")
    path.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    with (run / "stage_timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stage", "wall_seconds", "excluded_seconds", "effective_seconds"))
        writer.writeheader()
        for stage in summary.get("stages", []):
            command = stage.get("command", [])
            folder = run / stage["stage"]
            if "--output" in command:
                folder = Path(command[command.index("--output")+1]).resolve()
            deduction = sum(item["seconds"] for item in exclusions
                            if folder == Path(item["record"]).parent or folder in Path(item["record"]).parents)
            writer.writerow(dict(stage=stage["stage"], wall_seconds=stage["seconds"],
                excluded_seconds=deduction, effective_seconds=max(0, stage["seconds"]-deduction)))
    return summary
