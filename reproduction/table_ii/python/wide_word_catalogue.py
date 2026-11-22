"""Isolated 1,024-bit build of the complete actual-code word catalogue.

The existing short-code library is unchanged. Work exhaustion returns no
partial catalogue. The wide build retains the same causal leaders and Gen
recursion as the short-code backend.
"""

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
if __package__:
    from .extended_exact_spectrum import canonical_checks, VENDOR, WorkLimitError
else:
    from extended_exact_spectrum import canonical_checks, VENDOR, WorkLimitError

HERE = Path(__file__).resolve().parent

LIBRARY = HERE / ("_wide_word_catalogue.dll" if sys.platform == "win32"
                  else "_wide_word_catalogue.so")

_LIB = None

def build():
    source = HERE.parents[2] / "cpp/candidate_block_neighborhood_kernel.cpp"
    dependencies = [source, source.with_name("block_neighborhood_kernel.cpp"),
                    source.with_name("row_neighborhood_kernel.cpp"), VENDOR / "cpp/src/kernel.cpp"]
    if LIBRARY.exists() and LIBRARY.stat().st_mtime >= max(p.stat().st_mtime for p in dependencies):
        return LIBRARY
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        compiler = str(Path(os.environ.get("MW_MINGW64_LOC", "C:/MinGW64")) / "bin/g++.exe")
    command = [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-shared",
               "-DPOLAR_SPECTRUM_CAPACITY=1024", str(source), "-o", str(LIBRARY)]
    command += (["-static", "-static-libgcc", "-static-libstdc++"] if sys.platform == "win32" else ["-fPIC"])
    subprocess.run(command, check=True)
    return LIBRARY

def enumerate_words(n, profile, columns, cutoff, max_work=100_000_000):
    global _LIB
    canonical_checks(n, profile, columns)
    if not 4 <= n <= 1024 or not 1 <= cutoff <= n or max_work < 1:
        raise ValueError("Invalid wide catalogue request")
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build()))
        ptr = ctypes.POINTER(ctypes.c_uint8)
        _LIB.block_neighborhood.argtypes = [ctypes.c_int, ptr, ptr, ptr,
                                            ctypes.c_int, ctypes.c_uint64]
        _LIB.block_neighborhood.restype = ctypes.c_void_p
        _LIB.polar_spectrum_free.argtypes = [ctypes.c_void_p]
        _LIB.polar_spectrum_free.restype = None
    information = set(profile)
    raw = (ctypes.c_uint8 * (n * n))(
        *(columns[j] >> i & 1 for j in range(n) for i in range(n)))
    pointer = _LIB.block_neighborhood(n,
        (ctypes.c_uint8 * n)(*(int(i in information) for i in range(n))),
        raw, (ctypes.c_uint8 * n)(), cutoff, max_work)
    if not pointer:
        raise MemoryError("Wide catalogue allocation failed")
    try:
        result = json.loads(ctypes.string_at(pointer))
    finally:
        _LIB.polar_spectrum_free(pointer)
    if not result["exact"]:
        error = result["error"]
        raise (WorkLimitError if "work limit" in error else RuntimeError)(error)
    words = sorted((int(u, 2), weight) for u, weight in result.pop("words"))
    if len({u for u, _ in words}) != len(words):
        raise AssertionError("Duplicate wide catalogue input")
    return words, result
