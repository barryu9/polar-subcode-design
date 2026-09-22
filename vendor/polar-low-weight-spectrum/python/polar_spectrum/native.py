"""ctypes binding; all preprocessing and counting execute inside C++."""
import ctypes
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def library_path():
    override = os.environ.get("POLAR_SPECTRUM_LIBRARY")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parents[2]
    names = ("polar_spectrum.dll", "libpolar_spectrum.dll") if sys.platform == "win32" else (
        "libpolar_spectrum.dylib" if sys.platform == "darwin" else "libpolar_spectrum.so",)
    for directory in (root / "build", root / "build/Release", root / "lib"):
        for name in names:
            path = directory / name
            if path.exists():
                return path
    raise FileNotFoundError("C++ kernel not found. Run python tools/build_cpp.py or set POLAR_SPECTRUM_LIBRARY.")


def compute(n, checks, D, punctured, shortened, policy, max_work):
    if n > 256:
        raise ValueError("The C++ kernel supports n <= 256.")
    start = perf_counter()
    lib = ctypes.CDLL(str(library_path()))
    byte_ptr = ctypes.POINTER(ctypes.c_uint8)
    lib.polar_spectrum_compute.argtypes = [ctypes.c_int, ctypes.c_int, byte_ptr, ctypes.c_int,
                                          byte_ptr, byte_ptr, ctypes.c_int, ctypes.c_uint64]
    lib.polar_spectrum_compute.restype = ctypes.c_void_p
    lib.polar_spectrum_free.argtypes = [ctypes.c_void_p]
    lib.polar_spectrum_free.restype = None
    H = (ctypes.c_uint8 * (len(checks) * n))(*(row >> j & 1 for row in checks for j in range(n)))
    P = (ctypes.c_uint8 * n)(*(int(j in punctured) for j in range(n)))
    S = (ctypes.c_uint8 * n)(*(int(j in shortened) for j in range(n)))
    # Only the bounded cutoff enters the int32 ABI; retain the caller's D.
    pointer = lib.polar_spectrum_compute(n, len(checks), H, min(D, n), P, S,
                                       {"auto": 0, "enumerate": 1, "dp": 2}[policy], max_work)
    if not pointer:
        raise MemoryError("Native result allocation failed.")
    try:
        result = json.loads(ctypes.string_at(pointer))
    finally:
        lib.polar_spectrum_free(pointer)
    if "error" in result:
        from .algorithm import WorkLimitError
        raise (WorkLimitError if "work limit" in result["error"] else RuntimeError)(result["error"])
    result["spectrum"] = {int(k): int(v) for k, v in result["spectrum"].items()}
    result["cutoff"] = D
    result["binding_elapsed_seconds"] = perf_counter() - start
    return result
