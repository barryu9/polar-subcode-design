"""Isolated wider-bitset build of the existing complete Count/Gen enumerator.

The default vendor kernel and short-code adapter are unchanged. A separate
library uses 1,024-bit storage, the same recursion, and bounded work accounting.
An exhausted query raises WorkLimitError and supplies no spectrum certificate.
"""

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
if __package__:
    from .external_low_weight_spectrum import VENDOR, WorkLimitError, algorithm1
else:
    from external_low_weight_spectrum import VENDOR, WorkLimitError, algorithm1
from polar_spectrum.algorithm import validate

CAPACITY = 1024

LIBRARY = Path(__file__).with_name(
    '_extended_exact_spectrum.dll' if sys.platform == 'win32'
    else '_extended_exact_spectrum.so')

def build():
    source = VENDOR / 'cpp/src/kernel.cpp'
    if LIBRARY.exists() and LIBRARY.stat().st_mtime >= source.stat().st_mtime:
        return LIBRARY
    compiler = os.environ.get('CXX') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        compiler_path = Path(os.environ.get('MW_MINGW64_LOC', 'C:/MinGW64')) / 'bin/g++.exe'
        if compiler_path.exists():
            compiler = str(compiler_path)
    if not compiler:
        raise FileNotFoundError('A C++17 compiler is required for the extended kernel')
    command = [compiler, '-O3', '-DNDEBUG', '-std=c++17', '-shared',
               f'-DPOLAR_SPECTRUM_CAPACITY={CAPACITY}', str(source), '-o', str(LIBRARY)]
    command += (['-static', '-static-libgcc', '-static-libstdc++'] if sys.platform == 'win32'
                else ['-fPIC'])
    subprocess.run(command, check=True)
    return LIBRARY

def compute(n, checks, cutoff, max_work, policy='enumerate'):
    checks, punctured, shortened = validate(n, checks, cutoff, (), (), policy, max_work)
    if n > CAPACITY:
        raise ValueError('The extended exact kernel supports N <= 1,024')
    if n <= 256:
        return algorithm1(n, checks, cutoff, backend='cpp', policy=policy, max_work=max_work)
    tick = time.perf_counter()
    lib = ctypes.CDLL(str(build()))
    ptr = ctypes.POINTER(ctypes.c_uint8)
    lib.polar_spectrum_compute.argtypes = [ctypes.c_int, ctypes.c_int, ptr, ctypes.c_int,
                                          ptr, ptr, ctypes.c_int, ctypes.c_uint64]
    lib.polar_spectrum_compute.restype = ctypes.c_void_p
    lib.polar_spectrum_free.argtypes = [ctypes.c_void_p]
    lib.polar_spectrum_free.restype = None
    h = (ctypes.c_uint8 * (len(checks) * n))(
        *(row >> j & 1 for row in checks for j in range(n)))
    zero = (ctypes.c_uint8 * n)()
    pointer = lib.polar_spectrum_compute(n, len(checks), h, min(cutoff, n), zero, zero,
                                       {'auto': 0, 'enumerate': 1, 'dp': 2}[policy], max_work)
    if not pointer:
        raise MemoryError('Extended exact kernel result allocation failed')
    try:
        result = json.loads(ctypes.string_at(pointer))
    finally:
        lib.polar_spectrum_free(pointer)
    if 'error' in result:
        cls = WorkLimitError if 'work limit' in result['error'] else RuntimeError
        raise cls(result['error'])
    result['spectrum'] = {int(w): int(a) for w, a in result['spectrum'].items()}
    result.update(cutoff=cutoff, storage_capacity=CAPACITY,
                  binding_elapsed_seconds=time.perf_counter()-tick)
    return result

def canonical_checks(n, profile, columns):
    profile = tuple(profile)
    if n < 1 or n & (n-1) or n > CAPACITY or len(columns) != n:
        raise ValueError('Invalid length or expression array')
    if len(set(profile)) != len(profile) or any(not 0 <= i < n for i in profile):
        raise ValueError('Invalid profile')
    information = set(profile)
    mask = sum(1 << i for i in information)
    for j, value in enumerate(columns):
        if value < 0 or value >> j or value & ~mask or (j in information and value):
            raise ValueError('Noncanonical or noncausal dynamic frozen expression')
    return tuple((1 << j) ^ value for j, value in enumerate(columns) if j not in information)

def measure(n, profile, columns, cutoff, max_work=3000000, policy='enumerate',
            backend='cpp', reduce_repetitions=True):
    """Count all layers through cutoff, optionally using an exact repetition reduction.

For a static left projection with distance above the cutoff, a low-weight
codeword has zero left projection. Setting its information symbols to zero
leaves (b,b), including any cross-half constraints. Thus only the shortened
right constituent is needed, and every counted weight is doubled.
    """
    checks = canonical_checks(n, profile, columns)
    original_n, original_k, original_cutoff = n, len(profile), cutoff
    repetition, stages = 1, []
    if reduce_repetitions:
        while n > 1:
            half = n // 2
            left = [i for i in profile if i < half]
            left_static = all(not columns[j] for j in range(half))
            left_distance = min((1 << i.bit_count() for i in left), default=n+1)
            if not left_static or left_distance <= cutoff:
                break
            stages.append(dict(N=n, left_static_distance=left_distance,
                               cutoff=cutoff, proof='Plotkin weight lower bound'))
            profile = tuple(i-half for i in profile if i >= half)
            columns = [value >> half for value in columns[half:]]
            n, cutoff, repetition = half, cutoff//2, 2*repetition
        checks = canonical_checks(n, profile, columns)
    answer = (compute(n, checks, cutoff, max_work, policy) if backend == 'cpp'
              else algorithm1(n, checks, cutoff, backend='python', policy=policy,
                              max_work=max_work))
    if (not answer['exact'] or answer['spectrum'].get(0) != 1
            or answer['transmitted_dimension'] != len(profile)
            or answer['kernel_dimension'] != 0):
        raise AssertionError('Extended spectrum rank or zero-codeword audit failed')
    return dict(N=original_n, K=original_k, cutoff=original_cutoff,
                spectrum={repetition*w: a for w, a in answer['spectrum'].items()},
                exact=True, backend=backend, repetition_factor=repetition,
                reduced_N=n, reduced_K=len(profile), repetition_reduction=stages,
                counting=answer, stats=answer['stats'])
