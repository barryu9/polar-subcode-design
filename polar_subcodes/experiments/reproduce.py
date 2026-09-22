"""The five fixed-profile distance-improvement experiments."""
from pathlib import Path
import time

from polar_subcodes.code import load_code
from polar_subcodes.construction.pipeline import run, Limits
from polar_subcodes.io import ROOT, write_json

CASES = {
    'n64_k32': (12,'word',True),
    'n128_k42': (28,'nested',True),
    'n256_k64': (48,'gap',False),
    'n256_k128': (24,'word',True),
    'n256_k171': (12,'word',True),
}


def reproduce(name,output=None,limits=None):
    target,recipe,counted = CASES[name]
    folder = Path(output) if output else ROOT/'runs'/f'{name}_{time.time_ns()}'
    code,report = run(ROOT/'profile'/f'{name}.json',target,folder,limits or Limits(),recipe,counted)
    # The archived answer is opened only AFTER construction and verification.
    # It is a regression oracle, never a search seed or a source of coefficients.
    expected = load_code(ROOT/'results'/name/'support.json')
    same = code==expected
    write_json(folder/'reproduction.json',dict(support_matches_archive=same,
               archived_support_read_after_search=True,initial_support='zero'))
    if not same:
        raise AssertionError('Verified code differs from the archived deterministic support')
    print(f'Reproduced {name}: dmin={report["dmin"]}, Admin={report["Admin"]}. Output: {folder}',flush=True)
    return report
