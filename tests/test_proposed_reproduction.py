"""Regression checks for simulation-code identity and Table II's shared base."""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from polar_subcodes.code import load_code
from reproduction import proposed


class ProposedCurveReproduction(unittest.TestCase):
    def test_all_pre_and_post_configurations_match_the_published_designs(self):
        self.assertEqual(len(proposed.MANIFEST), 9)
        with redirect_stdout(io.StringIO()):
            proposed.check(proposed.MANIFEST)
        proposed_code = load_code(proposed.CONFIG / "n1024_k512.code.json")
        np_code = load_code(proposed.CONFIG / "n1024_k512_np.code.json")
        self.assertEqual(proposed_code.profile, np_code.profile)
        # Equation (9): all changed frozen expressions precede the first
        # information row whose polar row has weight at most 24 (here 688).
        boundary = min(i for i in proposed_code.profile if 1 << i.bit_count() <= 24)
        edits = [j for j in range(np_code.n) if np_code.columns[j] != proposed_code.columns[j]]
        self.assertEqual(boundary, 688)
        self.assertTrue(edits and max(edits) < boundary)

    def test_conditional_export_recomputes_identity_after_loading_the_archive(self):
        case = "n64_k32"
        row = proposed.MANIFEST[case]
        code = load_code(proposed.CONFIG / row["code_file"])
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            candidate = run / case / "designs/first/code.json"
            candidate.parent.mkdir(parents=True)
            # Use a snapshot's portable mask representation, independent of the
            # sorted position-to-source lists in the distributed curve file.
            raw = {"N": code.n, "profile": code.profile, "columns": list(code.columns)}
            candidate.write_text(json.dumps(raw), encoding="utf-8")
            archive = [{**{k: row[k] for k in ("profile_hash", "support_hash")}, "name": "first"}]
            (run / case / "frozen_archive.json").write_text(json.dumps(archive), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                proposed.collect(run, [case])
            self.assertEqual(load_code(run / "selected" / row["code_file"]), code)
            # Merely labelling a different code with the two hashes must fail.
            changed = next(j for j, mask in enumerate(code.columns) if mask)
            raw["columns"][changed] = 0
            candidate.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ"):
                proposed.collect(run, [case])

    def test_variant_timing_includes_one_shared_base_run(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cases = [(proposed.BASE, 100, 40), (proposed.BASE + "_lm", 25, 10),
                     (proposed.BASE + "_d32", 50, 20)]
            bench = {"results": [{"case": name, "construction_wall_seconds": seconds}
                                 for name, seconds, _ in cases]}
            (run / "benchmark.json").write_text(json.dumps(bench), encoding="utf-8")
            with (run / "paper_step_summary.csv").open("w", newline="", encoding="utf-8") as stream:
                rows = csv.writer(stream)
                rows.writerow(["case", "paper_operation", "wall_seconds"])
                for name, _, refinement in cases:
                    rows.writerow([name, "V-D | Total refinement (excludes composition)", refinement])
            proposed.write_paper_times(run)
            with (run / "table_ii_times.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(r["w_o_refinement_seconds"], r["w_refinement_seconds"]) for r in rows],
                             [("60.00", "100.00"), ("75.00", "125.00"), ("90.00", "150.00")])
