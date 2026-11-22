"""Run a Python child with the same cold-data boundary as its parent."""

import argparse
from pathlib import Path
import runpy
import sys
from independent_reproduction import install_guard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    install_guard(args.run_root)
    argv = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(args.script)] + argv
    runpy.run_path(str(args.script), run_name="__main__")

if __name__ == "__main__":
    main()
