"""Reproduce layered profiles and whole-row/prefix refinement at rate 171/256."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from independent_reproduction import Experiment, read, write

def main():
    run = Experiment(256, 171, 12, __doc__)
    if run.smoke:
        run.ga()
        return
    codes = run.bank(4, profiles=8, layered=True, depth=4, seeds=256, gap=True,
                     full_necessary=True)
    sources = list(codes)
    # The historical branch was retained after SCL comparison. Re-evaluate a
    # small fixed design sample instead of loading that selected branch.
    from polar_code.SCL_full_decoder import simulate_awgn_llr_batch
    from run_joint_scl_profile_design import prefix_assessment
    with run.stage("design_sample_scl_selection"):
        llr = simulate_awgn_llr_batch(256, 171, 2.0,
            range(2026100611000000, 2026100611004096))
        assessments = [dict(source=str(source), **prefix_assessment(read(source), llr))
                       for source in codes]
        chosen = sorted(assessments, key=lambda e: (e["errors"], e["pruning_errors"], e["source"]))[:2]
        write(run.root / "design_sample_selection.json", dict(frames=4096,
            EbN0_dB=2.0, L=32, assessments=assessments, selected=chosen,
            historical_profile_input=False, holdout_FER_selection=False))
    for index, selection in enumerate(chosen):
        source = Path(selection["source"])
        folder = run.root / f"row_refinement_{index:02d}"
        run.call(f"whole_rows_{index:02d}", "refine_whole_expression_spectrum.py", [
            "--source", source, "--output", folder, "--target", 12,
            "--seconds", 14400, "--work", 3000000, "--max-rank", 22])
        refined = folder / "code.json"
        sources.append(refined)
        sources += run.feedback(refined, f"joint_feedback_{index:02d}", 256,
                                2026100611000000, 4096, 2.0, catalogue_variants=False)
    run.freeze(sources)

if __name__ == "__main__":
    main()
