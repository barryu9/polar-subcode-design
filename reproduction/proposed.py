"""Reproduce the Proposed curves of Figs. 5–10, or inspect their published JSONs.

From the repository root:
    python reproduction/proposed.py check
    python reproduction/proposed.py cold --cases n256_k64 --output runs/short
    python reproduction/proposed.py cold --output runs/table-ii
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from polar_subcodes.code import load_code

CONFIG = ROOT / "configs/proposed"
REPLAY = ROOT / "reproduction/table_ii"
MANIFEST = json.loads((CONFIG / "manifest.json").read_text(encoding="utf-8"))
BASE = "n1024_k512"


def digest(value):
    """Use the canonical JSON encoding of the original curve identity audit."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def identities(code):
    profile = digest({"N": code.n, "profile": list(code.profile)})
    expressions = digest({"N": code.n, "columns_hex": [hex(mask) for mask in code.columns]})
    return profile, expressions


def verify(code, row):
    if code.n != row["N"] or code.k != row["payload_K"]:
        raise ValueError(f"Unexpected dimensions for {row['label']}")
    if identities(code) != (row["profile_hash"], row["support_hash"]):
        raise ValueError(f"Information positions or frozen expressions differ for {row['label']}")


def check(cases):
    for case in cases:
        row = MANIFEST[case]
        seed = load_code(CONFIG / row["profile_file"])
        code = load_code(CONFIG / row["code_file"])
        if seed.profile != code.profile or any(seed.columns):
            raise ValueError(f"{case}: the pre-construction file must specify only the profile")
        verify(code, row)
        print(f"{case}: Fig. {row['figure']}, {row['label']}, [{code.n},{code.k}], profile and expressions verified")


def collect(run, cases):
    """Pick the published curve only after the search has frozen its candidate archive."""
    destination = run / "selected"
    destination.mkdir(exist_ok=True)
    for case in cases:
        row = MANIFEST[case]
        parent = run / (BASE if case == BASE + "_np" else case)
        entries = json.loads((parent / "frozen_archive.json").read_text(encoding="utf-8"))
        matches = [item for item in entries if (item["profile_hash"], item["support_hash"]) ==
                   (row["profile_hash"], row["support_hash"])]
        if len(matches) != 1:
            raise ValueError(f"{case}: expected one matching profile and expression set; found {len(matches)}")
        # Relative paths keep a completed run readable after moving it to another computer.
        code = load_code(parent / "designs" / matches[0]["name"] / "code.json")
        verify(code, row)
        profile = {"N": code.n, "K": code.k, "profile": list(code.profile)}
        final = {**profile, "dynamic_expressions": {str(j): list(bits) for j, bits in enumerate(code.supports) if bits}}
        for filename, data in ((row["profile_file"], profile), (row["code_file"], final)):
            (destination / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Constructed {row['label']} [{code.n},{code.k}] -> {destination / row['code_file']}")


def write_paper_times(run):
    """Combine per-process measurements using the two cumulative columns of Table II."""
    report = json.loads((run / "benchmark.json").read_text(encoding="utf-8"))
    with (run / "paper_step_summary.csv").open(encoding="utf-8", newline="") as stream:
        refinement = {r["case"]: float(r["wall_seconds"]) for r in csv.DictReader(stream)
                      if r["paper_operation"] == "V-D | Total refinement (excludes composition)"}
    values = {}
    with (run / "table_ii_times.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "w_o_refinement_seconds", "w_refinement_seconds"])
        # Give the CSV the same row order as Table II, independently of run order.
        def table_order(row):
            key = row["case"]
            code = MANIFEST[key]
            variant = {"Proposed": 0, "Proposed-LM": 1, "Proposed-D32": 2}
            return code["N"], code["payload_K"], variant[code["label"]]

        for row in sorted(report["results"], key=table_order):
            key = row["case"]
            full = row["construction_wall_seconds"]
            subtotal = full - refinement[key]
            parent = BASE if key in (BASE + "_lm", BASE + "_d32") else None
            if parent:
                base_subtotal, base_full = values[parent]
                subtotal += base_subtotal
                full += base_full
            values[key] = (subtotal, full)
            writer.writerow([key, f"{subtotal:.2f}", f"{full:.2f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    for action, description in [
        ("check", "Validate the distributed profile/code pairs"),
        ("cold", "Build native tools, run the recorded search and match the curve identities"),
        ("collect", "Export matching codes from a previously completed cold run"),
    ]:
        option = sub.add_parser(action, help=description)
        option.add_argument("--cases", nargs="+", choices=tuple(MANIFEST), default=list(MANIFEST))
        if action != "check":
            option.add_argument("--output", type=Path, required=True, help="Run directory (fresh for cold)")
    args = parser.parse_args()
    if args.action == "check":
        check(args.cases)
        return
    run = args.output.resolve()
    if args.action == "cold":
        if run.exists():
            parser.error("Use a new directory for a cold search; use collect to inspect a completed run.")
        # NP is the measured intermediate before prefix protection. Both length-1024
        # variants start from the main candidate regenerated in this same run.
        table_cases = list(dict.fromkeys(BASE if key == BASE + "_np" else key for key in args.cases))
        subprocess.run([sys.executable, str(REPLAY / "build_native.py")], check=True)
        subprocess.run([sys.executable, str(REPLAY / "examples/reproduce_table_ii.py"),
                        "--cases", *table_cases, "--output", str(run)], check=True)
        subprocess.run([sys.executable, str(REPLAY / "examples/report_table_ii_stage_timing.py"),
                        "--run", str(run)], check=True)
        write_paper_times(run)
    collect(run, args.cases)


if __name__ == "__main__":
    main()
