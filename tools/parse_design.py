"""Print a profile and its dynamic frozen expressions in human-readable form."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main import main
if __name__=='__main__':
    raise SystemExit(main(['inspect']+sys.argv[1:]))
