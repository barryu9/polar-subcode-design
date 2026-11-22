"""Reproduce the length-256, dimension-64 structural-feedback construction."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from independent_reproduction import Experiment, read, write

def main():
    run = Experiment(256, 64, 48, __doc__)
    reliability = run.ga(2.0)
    if run.smoke:
        return
    output = run.root / "structural_exploration"
    run.call("structural_exploration", "structure_feedback_profile_design.py", [
        "--n", 256, "--k", 64, "--target", 48, "--reliability", reliability,
        "--order", "rm-ga", "--output", output, "--support-seconds", 120,
        "--nodes", 50000, "--work", 300000000, "--trials", 256,
        "--focus-batch", 64, "--outputs", 1])
    paths = []
    for index, code in enumerate(read(output / "summary.json")["outputs"]):
        path = run.root / "constructed" / f"{index:03d}.json"
        write(path, code)
        paths.append(path)
    run.freeze(paths)

if __name__ == "__main__":
    main()
