"""Inspect, enumerate and simulate the polar subcodes accompanying the paper."""
import argparse
from pathlib import Path
import sys
import unittest

from polar_subcodes.codec import load_codec
from polar_subcodes.io import ROOT


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("examples", help="Show where to run the numbered algorithms, examples and constructions")

    inspect = commands.add_parser("inspect", help="Print the information set and frozen expressions")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--include-static", action="store_true")
    inspect.add_argument("--output", type=Path)

    enumerate_ = commands.add_parser("enumerate", help="Count or list low-weight codewords")
    enumerate_.add_argument("--support", type=Path, required=True)
    enumerate_.add_argument("--cutoff", type=int, required=True)
    enumerate_.add_argument("--method", choices=["decomposition", "words", "native", "python", "quotient", "direct"],
                            default="decomposition")
    enumerate_.add_argument("--work", type=int, default=100_000_000)
    enumerate_.add_argument("--output", type=Path, required=True)

    simulate = commands.add_parser("simulate", help="Run BPSK-AWGN Monte Carlo with an SCL decoder")
    simulate.add_argument("--support", type=Path, required=True)
    simulate.add_argument("--snr", type=float, default=3.0, help="Eb/N0 in dB")
    simulate.add_argument("--frames", type=int, default=1000)
    simulate.add_argument("--list-size", type=int, default=64)
    simulate.add_argument("--batch-size", type=int, default=32)
    simulate.add_argument("--seed", type=int, default=2026)
    simulate.add_argument("--backend", choices=["auto", "numpy", "cpp"], default="auto")
    simulate.add_argument("--messages", choices=["zero", "random"], default="zero")
    simulate.add_argument("--output", type=Path, required=True)

    commands.add_parser("verify-theory", help="Run finite-case and implementation regression tests")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "examples":
        print("""Run the following commands from the repository root.

Numbered examples and proof checks
  python -m paper_examples all
  python -m theory.check_results

Fixed-profile construction (Algorithm 3) and profile selection (Algorithm 4)
  python -m algorithms.algorithm3_fixed_profile
  python -m algorithms.algorithm4_profiles

Simulation codes in Figs. 5–10
  python -m reproduction list
  python -m reproduction check
  python -m reproduction construct --cases n64_k32 --output runs/fig5""")
    elif args.command == "inspect":
        codec = load_codec(args.input)
        description = codec.mother.describe(args.include_static)
        if codec.polynomial is not None:
            description += f"\nCRC polynomial {codec.polynomial:#x}, payload {codec.payload} bits."
        if codec.taps:
            description += f"\nPAC taps in increasing delay order: {list(codec.taps)}."
        print(description)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(description + "\n", encoding="utf-8")
    elif args.command == "enumerate":
        from polar_subcodes.spectrum.api import enumerate_code

        enumerate_code(load_codec(args.support).as_subcode(), args.cutoff, args.output, args.method, args.work)
    elif args.command == "simulate":
        from polar_subcodes.decoding.simulation import simulate

        simulate(load_codec(args.support), args.output, args.snr, args.frames, args.list_size,
                 args.batch_size, args.seed, args.backend, args.messages)
    elif args.command == "verify-theory":
        tests = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():
            return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
