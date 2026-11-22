"""Reproduce an RM-GA anchor, exact outer completion, and prefix/spectrum refinement."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from independent_reproduction import Experiment

def main():
    run = Experiment(256, 128, 24, __doc__)
    if run.smoke:
        run.ga()
        return
    codes = run.bank(3, profiles=1, layered=False, anchor=True, joint=0, focus=64)
    sources = list(codes)
    for index, source in enumerate(codes):
        sources += run.feedback(source, f"joint_feedback_{index:02d}", 128,
                                2026100801000000, 2048, 1.5, catalogue_variants=False)
    run.freeze(sources)

if __name__ == "__main__":
    main()
