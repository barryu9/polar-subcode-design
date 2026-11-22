"""Reconstruct every comparison code in the FER figures and verify its identity.

Run from the checkout: python -m reproduction.references --output runs/references
The seven published Z matrices are the reference authors' specified codes. For
these, row elimination constructs the causal equations used in the simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path

from polar_subcodes.code import PolarSubcode
from polar_subcodes.codec import PayloadCodec
from polar_subcodes.construction.ebch import ebch_subcode
from polar_subcodes.construction.profiles import make_profile
from polar_subcodes.precoding import from_input_basis, PAC_TAPS, pac_code

HERE = Path(__file__).resolve().parent
CATALOG = HERE.parent / "configs" / "reference" / "manifest.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def construct(row):
    """Use only construction parameters, the specified shared profile or published Z."""
    n, payload, method = row["N"], row["payload"], row["method"]
    if method == "published_Z":
        source = read_json(HERE / "reference_inputs" / row["input"])
        basis = [int(s, 16) for s in source["Z_rows_hex"]]
        if len(basis) != payload:
            raise ValueError("Unexpected precoding matrix dimension")
        return PayloadCodec(from_input_basis(n, basis))
    if method == "ebch":
        return PayloadCodec(ebch_subcode(n, payload, row["design_distance"],
                                        int(row["primitive_polynomial"], 16), row["ebn0_db"]))
    if method == "same_profile":
        # Fig. 10 holds the information set fixed across this PAC comparison.
        profile = read_json(HERE / "reference_inputs" / row["input"])["profile"]
    else:
        profile = make_profile(n, row["K"], method, payload=payload,
                               ebn0_db=row.get("ebn0_db", 2.0))
    if row["label"].startswith("PAC"):
        return PayloadCodec(pac_code(n, profile, PAC_TAPS), taps=PAC_TAPS)
    return PayloadCodec(PolarSubcode(n, tuple(profile), (0,)*n), int(row["polynomial"],16))


def identity(codec):
    """Same hash input as the simulation library, including payload and outer CRC."""
    data = codec.as_dict()
    canonical = dict(N=codec.mother.n, profile=list(codec.mother.profile),
                     columns_hex=[hex(c) for c in codec.mother.columns],
                     payload=codec.payload, crc=data.get("crc"))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def export(row, directory):
    codec = construct(row)
    if identity(codec) != row["code_hash"]:
        raise ValueError(f"{row['id']}: regenerated code differs from the FER code")
    prefix = directory / row["id"]
    profile = dict(N=codec.mother.n, K=codec.mother.k, profile=list(codec.mother.profile))
    final = codec.as_dict()
    # Publish each coefficient once, in the physical-index notation used in Section II.
    final.pop("columns")
    final["dynamic_expressions"] = {str(j): list(s) for j,s in enumerate(codec.mother.supports) if s}
    for suffix, data in [(".profile.json", profile), (".code.json", final)]:
        path = prefix.with_name(prefix.name + suffix)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="One ID from configs/reference/manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    cases = [r for r in read_json(CATALOG) if args.case in (None, r["id"])]
    if not cases:
        parser.error("Unknown case")
    args.output.mkdir(parents=True, exist_ok=False)
    for row in cases:
        export(row, args.output)
        print(f"{row['id']}  {row['code_hash'][:16]}  verified")


if __name__ == "__main__":
    main()
