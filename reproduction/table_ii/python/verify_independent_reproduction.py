"""Post-freeze full-digest acceptance checks; never a design input."""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    (64, 32): ("2e978b67a820b65bd67a448b385c8721a06d5a7c29eae47d75efe04a1d3f5b18",
               "809cb7f367d6789da32eb02a9dec7f8a7b72aa15b7517e9d6e928415f994870d"),
    (128, 64): ("4e0ce7b639c43e64074c7af4cc9948d4b997220e3cc1840c3037288f56857aa7",
                "5fa6305afd338240f2e16287134818009bae864c83168e50ef9e401cca399212"),
    (256, 64): ("746f1f7b74f3c653b083fe9b82cc7ca4c7555f60c9dc9fe8606dce6722c5268c",
                "f7b67c50d1199eee8d3ff0e1ca82d46d9d5ffc553497e26ce767055e272ef8c3"),
    (256, 128): ("46be57e2bf2ecd9be29c24191188434769f033bf1ad7f46e80a43ff7b7cfa3ff",
                 "f95bdd1697f84f50c56d4d81daf95a8f58060ebfd505199c3827cecd4bd7c95f"),
    (256, 171): ("b8dffd9c8f297ce390a049058a60f8e6eb11b920859764d29529f17da64c1ffe",
                 "3995dceb8d007293e14f28c29eb68950d1379724fd78353defb217a681a33840"),
    (1024, 512): ("91d744fa0600af16c057fd7dcf44f1a2658484c8d34eb6cee255ca72eebde9b7",
                  "7600123772b0d7609afab06025b5d2c5ed55cf0c1a7cbd646ff6db82afd6cb56"),
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode("ascii")).hexdigest()

def check(run, *, expected=None, analyze_deductions=True):
    run = Path(run).resolve()
    if analyze_deductions:
        from cold_reproduction_timing import update_summary
        summary = update_summary(run)
    else:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    archive_path = run / "frozen_archive.json"
    archive = json.loads(archive_path.read_text())
    if expected is None:
        expected = EXPECTED[(summary["N"], summary["K"])]
    matched, same_profile, checked = [], [], []
    for row in archive:
        path = Path(row["code_path"]).resolve()
        if run not in path.parents:
            raise ValueError("Frozen archive contains an external design path")
        raw = json.loads(path.read_text())
        n, profile, columns = raw["N"], raw["profile"], raw["columns"]
        if (n != summary["N"] or len(columns) != n or len(profile) != summary["K"]
                or len(set(profile)) != len(profile) or any(not 0 <= i < n for i in profile)
                or any(type(mask) is not int or mask < 0 or mask >> n for mask in columns)):
            raise AssertionError("Invalid dimensions, indices, or coefficient masks")
        ph = digest(dict(N=n, profile=sorted(profile)))
        sh = digest(dict(N=n, columns_hex=[hex(c) for c in columns]))
        if (ph, sh) != (row["profile_hash"], row["support_hash"]):
            raise AssertionError("Frozen metadata disagrees with actual vectors")
        info = set(profile)
        causal = all((j not in info or not mask) and all(
            i in info and i < j for i in range(n) if mask >> i & 1)
            for j, mask in enumerate(columns))
        if not causal or len(info) != summary["K"]:
            raise AssertionError("Invalid causal expression or profile")
        proof = raw.get("verification", {})
        bound = raw.get("distance_lower_bound", proof.get("distance_lower_bound"))
        if bound is None or bound < summary["target"]:
            raise AssertionError("Construction has no recorded target-distance verification")
        result = dict(name=row["name"], code_path=str(path), causal=True,
                      profile_hash=ph, support_hash=sh)
        checked.append(result)
        if ph == expected[0]:
            same_profile.append(result)
            if sh == expected[1]:
                matched.append(result)
    summary.update(exact_historical_identity_not_yet_checked=False, acceptance_passed=bool(matched))
    (run / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    result = dict(N=summary["N"], K=summary["K"], exact_pair_found=bool(matched),
        matching_pairs=matched, same_profile_count=len(same_profile), checked=len(checked),
        frozen_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        full_SHA256_comparison=True, historical_vectors_read=False,
        acceptance_after_freeze=True, timing=summary)
    matching_ph, matching_sh = expected
    result["expected_profile_hash"] = matching_ph
    result["expected_support_hash"] = matching_sh
    (run / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.run)
    print(json.dumps({k: result[k] for k in
        ("N", "K", "exact_pair_found", "same_profile_count", "checked")}), flush=True)
    raise SystemExit(0 if result["exact_pair_found"] else 2)

if __name__ == "__main__":
    main()
