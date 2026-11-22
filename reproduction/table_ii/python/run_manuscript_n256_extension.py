"""Exact affine-row optimization on nested N128/N256 extensions of a solved tail.

The tail checkpoint is produced separately by the documented joint search.
N64 uses the manuscript affine enumerator for verification; N128/N256 row
catalogues use the existing complete word enumerator, as singleton families.
"""

import json

def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def cutoff_source_start(profile, cutoff, n):
    """Proposition 1: earliest possible leader through the fixed cutoff."""
    return min((i for i in profile if 1 << i.bit_count() <= cutoff), default=n)
