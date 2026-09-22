# Polar Low-Weight Spectrum

Exact, deterministic computation of the low-weight spectrum of binary linear
polar subcodes, including PAC, punctured, and shortened codes. This repository
implements **Algorithm 1** of the research manuscript
*Deterministic Low-Weight Spectrum Computation of Polar Subcodes via Affine
Coset Counting*, by Junhua You, Shaohua Wu, Yajing Deng, Nan Cheng, and Qinyu Zhang.

[中文说明](README_zh.md) · [Algorithm and complexity](docs/algorithm.md) ·
[Conventions and API](docs/conventions.md) · [Validation](docs/validation.md)

The Python prototype, pure MATLAB prototype, and C++17 kernel each implement
the complete flow: shortening and projection normalization, first-one coset
decomposition, deterministic threshold pruning, truncated coset WEF recursion,
and weighted affine counting by bounded deviations or syndrome DP.
No channel samples or SCL list size are needed.

Coefficients are exact integers. The result counts **distinct transmitted
codewords** of weight at most D, including the zero word. A work-limit exception
returns no incomplete spectrum.

## Python examples

Python 3.10+; the reference implementation uses only the standard library.
Run from the repository root:

~~~shell
python examples/python/run_spectrum.py
python examples/python/run_spectrum.py --n 128 --kind pac --mode puncture -D 8
python examples/python/run_spectrum.py --n 256 --kind polar --mode shorten -D 8 --json results/example.json
~~~

The default is the (64,32) Polar mother code, no rate matching, D=8:

~~~text
weight   multiplicity
     0   1
     8   664
~~~

Build the optional native backend with g++ or clang++:

~~~shell
python tools/build_cpp.py
python examples/python/run_spectrum.py --n 128 --kind pac --backend cpp --mode mixed -D 8 --repeat 7
~~~

Alternatively, CMake supports GCC, Clang, and MSVC:

~~~shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
~~~

Use --compiler with the build script if the compiler is outside PATH.
POLAR_SPECTRUM_LIBRARY can point to a custom shared-library location.
The C++ kernel supports power-of-two lengths up to 256; it has no external
library dependency and is called through ctypes, not a subprocess.
See [cpp/include/polar_spectrum.h](cpp/include/polar_spectrum.h) for its C ABI.

Optional Python package installation:

~~~shell
python -m pip install -e .
~~~

## MATLAB examples

Run these commands from the repository root in MATLAB:

~~~matlab
addpath('examples/matlab');
r = run_spectrum();                         % (64,32) Polar, D=8
r = run_spectrum(128,'pac','puncture',8);
r = run_spectrum(256,'polar','shorten',8);
run_all;                                    % all 24 fixed cases
~~~

The example invokes only MATLAB functions. No MEX, Python, Java, or Symbolic
Math Toolbox is used. Counts are decimal strings backed by exact integer
limb arithmetic. Tested with MATLAB R2025a.

## Fixed profiles and matrices

All six profiles are hard-coded: Polar and PAC at (64,32), (128,64), (256,128).
Both code types share the GA information set at **Eb/N0=2.0 dB and mother
rate R=1/2**. PAC uses causal taps **[1,0,1,1,0,1,1]**, the agreed printed
octal-133 convention. The actual taps, orientation, and GA approximation are
fully specified in [conventions](docs/conventions.md).

| Mode | Punctured coordinates, zero-based | Shortened coordinates, zero-based |
|---|---|---|
| none | empty | empty |
| puncture | 0 through N/16−1 | empty |
| shorten | empty | N−N/16 through N−1 |
| mixed | 0 through N/32−1 | N−N/32 through N−1 |

These are explicit demonstration patterns, not optimized rate-matching
constructions. JSON fixtures and dense binary selector CSVs are committed under
[profiles](profiles). MATLAB and Python load their own literal fixtures;
neither example runs GA. MATLAB indices are one-based.

Shortening imposes x[S]=0 before deleting coordinates. Puncturing deletes
coordinates without imposing their values. Effective dimension is computed
and reported; the mother-code (N,K) label does not imply the same dimension
after rate matching.

## Reproduce the checks

~~~shell
python -m pip install pytest
python tools/build_cpp.py
python -m pytest -q
python tools/benchmark.py
~~~

In MATLAB:

~~~matlab
addpath('tests');
test_matlab;
~~~

Then rerun the Python benchmark to compare its results with the saved MATLAB
report. Committed validation includes 24 three-language profile comparisons,
independent exhaustive small-code checks, and coefficients exceeding 64 bits.
See [validation](docs/validation.md) for timing scope and limitations.

## Repository layout

| Directory | Contents |
|---|---|
| python/polar_spectrum | Python Algorithm 1, GF(2) algebra, fixed profiles, ctypes binding |
| matlab/+polar_spectrum | Pure MATLAB Algorithm 1 and exact integer arithmetic |
| cpp | Standalone C++ kernel and public C header |
| examples | Python and MATLAB spectrum/timing examples |
| profiles | Fixed GA profiles, coordinate patterns, explicit selector matrices |
| tests | Independent exhaustive oracle, profile and large-integer checks |
| tools | Native build, fixture audit, benchmark and source packaging scripts |
| docs | Paper mapping, formulas, conventions, complexity, validation |
| results | Reproducible validation and timing records |

MIT licensed. The associated paper remains a research manuscript; no
publication acceptance or new DOI is implied. Citation metadata is supplied
in [CITATION.cff](CITATION.cff).
