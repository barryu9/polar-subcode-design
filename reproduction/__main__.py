"""List, validate or reconstruct the code curves in Section VI (Figs. 5–10)."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

from polar_subcodes.code import load_profile
from polar_subcodes.codec import load_codec
from .references import CATALOG, construct, export, identity, read_json

ROOT = Path(__file__).resolve().parents[1]


def catalogue():
    """Join the two manifests in figure order. A case ID identifies one code curve."""
    cases = []
    for key, row in read_json(ROOT / "configs/proposed/manifest.json").items():
        cases.append((key, "proposed", row["figure"], row["label"], row["L"], row["simulator_code_hash"]))
    for row in read_json(CATALOG):
        cases.append((row["id"], "reference", row["figure"], row["label"], row["L"], row["code_hash"]))
    return sorted(cases, key=lambda row: (row[2], row[1], row[0]))


def check():
    """Tie every plotted series to both config files and to the saved FER measurements."""
    cases = catalogue()
    references = {row["id"]: row for row in read_json(CATALOG)}
    expected = {}
    for key, kind, figure, label, size, digest in cases:
        directory = ROOT / "configs" / kind
        # The profile records the information set I. The final configuration adds
        # the dynamic frozen expressions and, where applicable, the payload mapping.
        profile = load_profile(directory / (key + ".profile.json"))
        codec = load_codec(directory / (key + ".code.json"))
        if (profile.n, profile.profile) != (codec.mother.n, codec.mother.profile) or identity(codec) != digest:
            raise ValueError(f"Configuration mismatch in Fig. {figure}: {label}")
        if kind == "reference" and codec != construct(references[key]):
            raise ValueError(f"Payload mapping differs from the construction for {key}")
        if key in expected:
            raise ValueError(f"Repeated case {key}")
        expected[key] = (codec.mother.n, codec.payload, size, digest)

    # Each SNR point belongs to a particular code and SCL list size L. Use
    # payload K here: a CRC-aided code has additional mother-code information bits.
    # Reading the stored frame/error counts also checks how the plotted FER was obtained.
    displayed = set()
    with (ROOT / "results/fer.csv").open(encoding="utf-8", newline="") as stream:
        for point in csv.DictReader(stream):
            key = point["case"]
            if (int(point["N"]), int(point["K"]), int(point["L"]), point["code_hash"]) != expected[key]:
                raise ValueError(f"FER/configuration mismatch for {key}")
            frames, errors = int(point["frames"]), int(point["errors"])
            if not 0 <= errors <= frames or abs(float(point["FER"]) - errors/frames) > 1e-10:
                raise ValueError(f"Inconsistent FER counts for {key}")
            if int(point["displayed"]):
                displayed.add(key)
    if displayed != set(expected):
        raise ValueError("The FER figures and configuration catalogue cover different codes")
    print(f"Verified {len(cases)} profile/code pairs and their Fig. 5–10 measurements.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show the figure, legend, decoder list size and case ID")
    sub.add_parser("check", help="Check the archived curve configurations and measurements")
    build = sub.add_parser("construct", help="Run construction for all codes or selected case IDs")
    build.add_argument("--cases", nargs="+", choices=[c[0] for c in catalogue()])
    build.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "list":
        for case, _, fig, name, size, _ in catalogue():
            print(f"Fig. {fig:2d}  {name:23s}  L={size:3d}  {case}")
    else:
        base = args.output.resolve()
        selected = args.cases or [c[0] for c in catalogue()]
        base.mkdir(parents=True, exist_ok=False)
        refs = [r for r in read_json(CATALOG) if r["id"] in selected]
        # The reference manifest specifies each comparison code. Proposed cases
        # use the Section V construction and are collected after its search finishes.
        if refs:
            directory = base / "reference"
            directory.mkdir()
            for row in refs:
                export(row, directory)
                print(f"Verified {row['id']}", flush=True)
        proposed = [c[0] for c in catalogue() if c[1]=="proposed" and c[0] in selected]
        if proposed:
            subprocess.run([sys.executable, str(ROOT/"reproduction/proposed.py"), "cold",
                            "--cases", *proposed, "--output", str(base/"proposed")], check=True)


if __name__ == "__main__":
    main()
