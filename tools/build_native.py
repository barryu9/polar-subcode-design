"""Build the standard C++ spectrum library and optional NumPy SCL extension."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default=os.environ.get('CXX'))
    parser.add_argument('--scl', action='store_true', help='Also compile the optional SCL backend')
    args = parser.parse_args()
    compiler = args.compiler or shutil.which('g++') or shutil.which('clang++')
    if not compiler and os.name == 'nt':
        candidate = Path(os.environ.get('MW_MINGW64_LOC','C:/MinGW64'))/'bin/g++.exe'
        if candidate.exists():
            compiler = str(candidate)
    if not compiler:
        parser.error('Install a C++17 compiler and pass --compiler. See README for MSVC/CMake.')
    folder = ROOT/'build'
    folder.mkdir(exist_ok=True)
    suffix = '.dll' if os.name == 'nt' else '.dylib' if sys.platform == 'darwin' else '.so'
    command = [compiler,'-O3','-DNDEBUG','-std=c++17','-shared',
               str(ROOT/'cpp/candidate_block_neighborhood_kernel.cpp'),
               '-o',str(folder/('polar_subcodes'+suffix))]
    # MinGW thread runtimes must not become undeclared DLL dependencies.
    command += ['-static','-static-libgcc','-static-libstdc++'] if os.name == 'nt' else ['-fPIC']
    print(f'Building spectrum library with {compiler}', flush=True)
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, '-c',
        'import ctypes, sys; library = ctypes.CDLL(sys.argv[1]); '
        'assert hasattr(library, "polar_spectrum_compute"); '
        'assert hasattr(library, "block_neighborhood")',
        str(folder/('polar_subcodes'+suffix))], check=True)
    print('Built exact spectrum and complete-word kernels.')
    if args.scl:
        subprocess.run([sys.executable,str(ROOT/'tools/setup_scl.py'),'build_ext','--inplace'],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
