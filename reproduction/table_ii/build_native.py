"""Build the native counting and design-sample decoding kernels from repo sources."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import importlib.machinery
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
KERNELS = HERE / "python"
CPP = REPOSITORY / "cpp"
sys.path.insert(0, str(KERNELS))
from native_paths import library_name


def build(compiler=None):
    compiler = compiler or os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler and os.name == "nt":
        candidate = Path(os.environ.get("MW_MINGW64_LOC", "C:/MinGW64")) / "bin/g++.exe"
        if candidate.exists():
            compiler = str(candidate)
    if not compiler:
        raise RuntimeError("Install a C++17 compiler (GCC or Clang) and set CXX to its path.")
    # Distinct host library extensions allow a shared checkout to be used on
    # Windows and WSL without accidentally loading another host's binary.
    vendor_kernel = REPOSITORY / "vendor/polar-low-weight-spectrum/cpp/src/kernel.cpp"
    suffix = ".dll" if os.name == "nt" else ".so"
    jobs = [
        (library_name("_row_neighborhood_kernel"), CPP / "row_neighborhood_kernel.cpp", []),
        (library_name("_block_neighborhood_kernel"), CPP / "block_neighborhood_kernel.cpp", []),
        (library_name("_candidate_block_neighborhood_kernel"), CPP / "candidate_block_neighborhood_kernel.cpp", []),
        ("_extended_exact_spectrum" + suffix, vendor_kernel, ["-DPOLAR_SPECTRUM_CAPACITY=1024"]),
        ("_wide_word_catalogue" + suffix, CPP / "candidate_block_neighborhood_kernel.cpp", ["-DPOLAR_SPECTRUM_CAPACITY=1024"]),
        (library_name("_relative_subset_counter_v2"), HERE / "cpp/relative_subset_counter.cpp", []),
    ]
    dependencies = [vendor_kernel, *CPP.glob("*neighborhood_kernel.cpp")]
    def compile_one(job):
        filename, source, flags = job
        destination = KERNELS / filename
        if destination.exists() and all(destination.stat().st_mtime >= f.stat().st_mtime
                                        for f in [source, *dependencies, Path(__file__)]):
            return
        shared = ["-static", "-static-libgcc", "-static-libstdc++"] if os.name == "nt" else ["-fPIC"]
        print("Building", filename, flush=True)
        subprocess.run([compiler, "-std=c++17", "-O3", "-DNDEBUG", "-shared", *flags,
                        str(source), *shared, "-o", str(destination)], check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(compile_one, jobs))
    # setuptools chooses the compiler appropriate for the running CPython ABI.
    extension = CPP / "scl_step_decoder_cpp.cpp"
    candidates = [KERNELS / "polar_code" / ("_scl_step_decoder_cpp" + ext)
                  for ext in importlib.machinery.EXTENSION_SUFFIXES]
    setup = HERE / "setup_decoder.py"
    if not any(f.exists() and f.stat().st_mtime >= max(extension.stat().st_mtime, setup.stat().st_mtime)
               for f in candidates):
        subprocess.run([sys.executable, str(setup), "build_ext", "--inplace"], cwd=KERNELS, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler")
    build(parser.parse_args().compiler)
