"""Reproduce diverse profiles and independent suffix/prefix completion."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from independent_reproduction import Experiment

def main():
    run = Experiment(1024, 512, 24, __doc__)
    if run.smoke:
        run.ga()
        return
    codes = run.bank(5, profiles=3, layered=True, depth=1, focus=2)
    sources = list(codes)
    for index, source in enumerate(codes):
        suffix = run.root / f"suffix_{index:02d}"
        run.call(f"suffix_refinement_{index:02d}", "refine_recursive_suffix_spectrum.py", [
            "--source", source, "--output", suffix, "--target", 24,
            "--suffix-length", 256, "--trials", 200, "--seconds", 14400, "--work", 100000000])
        sources.append(suffix / "code.json")
        for floor in (1, 16):
            prefix = run.root / f"prefix_{index:02d}_floor{floor}"
            run.call(f"prefix_{index:02d}_{floor}", "design_scl_protected_prefix.py", [
                "--source", source, "--output", prefix, "--target", 24, "--cutoff", 24,
                "--row-floor", floor, "--frames", 512, "--snr", 1.0,
                "--list-size", 32, "--seed", 2026100461000000])
            final = run.root / f"combined_{index:02d}_floor{floor}"
            run.call(f"compose_{index:02d}_{floor}", "compose_scl_prefix_completion.py", [
                "--prefix", prefix / "code.json", "--suffix", suffix / "code.json",
                "--output", final, "--cutoff", 24])
            sources.append(final / "code.json")
    run.freeze(sources)

if __name__ == "__main__":
    main()
