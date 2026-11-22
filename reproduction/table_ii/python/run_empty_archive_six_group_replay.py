"""Build six distance-qualified banks with audited empty-archive feedback.

Only channel ordering, polar profiles, and zero expressions initialize the run.
Previously feasible designs and reference files are denied by a filesystem audit
hook. A frozen archive can subsequently be compared with historical designs.
All accepted variants, including dominated ones, remain in the archive.
Distance qualification alone does not recover a historical locked design.
Exact profile/support recovery is a separately reported acceptance test.
"""

import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')

os.environ.setdefault('MKL_NUM_THREADS', '1')

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

ROOT = Path(__file__).resolve().parents[1]

def is_within(path, root):
    return path == root or root in path.parents

def guard_history(output):
    output = output.resolve()
    forbidden = [(ROOT/'results').resolve(), (ROOT/'reference').resolve()]

    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if any(is_within(path, p) for p in forbidden) and not is_within(path, output):
            raise PermissionError(f'Cold replay denies historical design access: {path}')
    sys.addaudithook(audit)
