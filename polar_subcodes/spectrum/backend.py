"""Complete native word generation and exact bounded-weight counting."""
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import sys

from polar_subcodes.io import ROOT

VENDOR = ROOT/'vendor/polar-low-weight-spectrum'
sys.path.insert(0, str(VENDOR/'python'))
from polar_spectrum import algorithm1, WorkLimitError

_LIB = None


def library_path():
    suffix = '.dll' if os.name == 'nt' else '.dylib' if sys.platform == 'darwin' else '.so'
    path = ROOT/'build'/('polar_subcodes'+suffix)
    if not path.exists():
        raise FileNotFoundError('Native spectrum library missing. Run: python tools/build_native.py')
    return path


def count(code, cutoff, max_work=100_000_000, backend='cpp'):
    """Return exact coefficients for 0 <= w <= cutoff, or raise on exhaustion."""
    if type(cutoff) is not int or not 0 <= cutoff <= code.n or max_work < 1:
        raise ValueError('Invalid cutoff or work limit')
    if backend == 'cpp':
        os.environ['POLAR_SPECTRUM_LIBRARY'] = str(library_path())
    answer = algorithm1(code.n, code.checks, cutoff, backend=backend, max_work=max_work)
    if (not answer['exact'] or answer['spectrum'].get(0) != 1 or
            answer['transmitted_dimension'] != code.k or answer['kernel_dimension'] != 0):
        raise RuntimeError('Spectrum dimension or zero-word audit failed')
    return answer


def words(code, cutoff, max_work=100_000_000):
    """List every nonzero polar input u with wt(uG) <= cutoff exactly once.

    This is a complete bounded-weight directory, not a list-decoder estimate.
    Native limits are 4 <= N <= 256. Tiny codes use independent message traversal.
    """
    global _LIB
    if type(cutoff) is not int or not 0 <= cutoff <= code.n or max_work < 1:
        raise ValueError('Invalid cutoff or work limit')
    if cutoff == 0:
        return [], dict(exact=True, spectrum={0: 1})
    if code.n < 4:
        values = [(code.input_word(m), code.encode(m).bit_count()) for m in range(1, 1 << code.k)]
        values = [(u,w) for u,w in values if w <= cutoff]
        return values, dict(exact=True, spectrum={0:1, **dict(Counter(w for _,w in values))})
    if code.n > 256:
        raise ValueError('The native complete-word backend currently supports N <= 256')
    if _LIB is None:
        _LIB = ctypes.CDLL(str(library_path()))
        pointer = ctypes.POINTER(ctypes.c_uint8)
        _LIB.block_neighborhood.argtypes = [ctypes.c_int,pointer,pointer,pointer,ctypes.c_int,ctypes.c_uint64]
        _LIB.block_neighborhood.restype = ctypes.c_void_p
        _LIB.polar_spectrum_free.argtypes = [ctypes.c_void_p]
        _LIB.polar_spectrum_free.restype = None
    n = code.n
    info = set(code.profile)
    array = (ctypes.c_uint8*(n*n))(*(c >> i & 1 for c in code.columns for i in range(n)))
    ptr = _LIB.block_neighborhood(n, (ctypes.c_uint8*n)(*(int(i in info) for i in range(n))),
                                 array, (ctypes.c_uint8*n)(), cutoff, max_work)
    if not ptr:
        raise MemoryError('Native result allocation failed')
    try:
        result = json.loads(ctypes.string_at(ptr))
    finally:
        _LIB.polar_spectrum_free(ptr)
    if not result['exact']:
        raise (WorkLimitError if 'work limit' in result['error'] else RuntimeError)(result['error'])
    values = sorted((int(u,2),w) for u,w in result.pop('words'))
    if len({u for u,_ in values}) != len(values):
        raise RuntimeError('Duplicate native inputs')
    result['spectrum'] = {0:1, **dict(sorted(Counter(w for _,w in values).items()))}
    return values, result
