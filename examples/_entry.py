"""Allow one-file examples to run from any current working directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(command):
    from main import main
    raise SystemExit(main(command+sys.argv[1:]))
