# Earlier reader release · 历史教学示例

The following guide documents the earlier five-case release. For the r112 manuscript, figures and Table II, start at the [current README](../README.md). Commands below run from the repository root.


Reader-oriented code for **Deterministic and Efficient Design of Dynamic Frozen
Expressions for Polar Subcodes via Exact Low-Weight Spectrum Characterization**.

Authors: Junhua You, Shaohua Wu, Aolin Liu, Yajing Deng, and Qinyu Zhang.

This repository separates reusable numerical algorithms from experiment drivers.
The five principal constructions start from **zero support**, using only the
information set. Archived answers are checked only after the search finishes.
No trained model, reference precoder, GPU, or original research workspace is needed.

## Quick Start

Python 3.10 or newer, NumPy, and a C++17 compiler are required for the fast
construction examples. Python 3.12 is the tested interpreter. Matplotlib is
used only for plots. Setuptools builds the optional native SCL decoder.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/build_native.py --scl
python main.py verify-theory
python examples/n64_k32.py
python main.py examples
```

On Linux/macOS, use GCC or Clang. On Windows the spectrum build uses MinGW
(`g++` on PATH, `CXX`, `--compiler`, or `C:/MinGW64/bin/g++.exe`). The optional
SCL extension uses Microsoft C++ Build Tools. Alternatively build the spectrum
library with MSVC:

```bash
cmake -S . -B build/cmake
cmake --build build/cmake --config Release
python tools/setup_scl.py build_ext --inplace
```

`python tools/build_native.py` without `--scl` is enough for construction and
enumeration. SCL then uses its slower NumPy fallback. Build binaries locally:
no DLLs, executables, virtual environments, or platform-specific caches are tracked.
Use a source checkout, not an installed wheel, for the data-driven examples.

## Reproduce the Designs

| Example | Zero-support `d_P` | Final `dmin` | Exact first coefficient |
|---|---:|---:|---:|
| `examples/n64_k32.py` | 8 | 12 | `A12 = 3,280` |
| `examples/n128_k42.py` | 16 | 28 | `A28 = 16,128` |
| `examples/n256_k64.py` | 32 | 48 | Not counted |
| `examples/n256_k128.py` | 16 | 24 | `A24 = 344,220` |
| `examples/n256_k171.py` | 8 | 12 | `A12 = 19,456` |

```bash
python examples/n128_k42.py --output runs/n128
python examples/n256_k128.py --output runs/n256 --seconds 600
python main.py reproduce all --output runs/all
python examples/repeated_designs.py --output runs/repeated --repetitions 3
```

Each output directory must be new. Without `--output`, a timestamped directory
under `runs/` is created. The log explains domain selection, the complete mother
directory, shared source conditions, quotient exclusions, conditional means,
and final verification. It also writes structured `events.jsonl`, intermediate
models, search frontiers, timings, `support.json`, and `equations.txt`.

`results/paper_summary.json` records the historical measurements from the author's
computer, not a hardware-independent runtime guarantee. The cleaned drivers add
some checks and do not have precisely the same timing boundaries. Their fresh
wall times appear in each run's report. The largest examples take longer than the
small examples and can use appreciable memory.

The initial reader release passed **61 tests** with both native backends built.
All five sequential zero-support replays matched the archived supports exactly.
See [validation notes](../docs/VALIDATION.md) and the compact logs, equations and
source hashes in [release evidence](../results/release_validation/manifest.json).
To export an independent replay and a fresh test transcript:

```bash
python tools/export_evidence.py --runs runs/all --output runs/evidence
```

## Repository Map

```text
main.py                         all public commands
examples/                       standalone experiment entry points
profile/                        fixed information sets, with no support seeds
results/                        archived supports and compact reference evidence
polar_subcodes/
  code.py                       validated code model, encoding, JSON, equations
  algebra.py                    binary linear algebra and polar transforms
  construction/                 domains, source patterns, outer completion
  spectrum/                     exact counts and complete word directories
  decoding/                     NumPy/C++ SCL and AWGN simulation
  theory/                       compatibility, bounds, complete tiny-code reference
  experiments/                  reproducible experiment orchestration and figures
cpp/                            native word-directory and SCL sources
vendor/polar-low-weight-spectrum/ upstream exact counter, unchanged MIT license
tests/                          exhaustive small-code and numerical regressions
tools/                          builds, parser, and evidence export
docs/                           conventions, algorithm map, provenance and scope
```

## Inspect or Supply a Code

```bash
python tools/parse_design.py results/n256_k128/support.json --output runs/equations.txt
python main.py inspect profile/n64_k32.json --include-static
python main.py construct --profile profile/n64_k32.json --distance 12 --output runs/my_design
```

A profile JSON contains `N`, `K`, and a sorted `profile` array. A support JSON adds
`columns`, whose entry `j` encodes the earlier information bits in the expression
for `u[j]`. Decimal strings preserve integers larger than 64 bits. Alternatively,
give `dynamic_expressions`, e.g. `{"4": [1, 3]}` means `u[4] = u[1] XOR u[3]`.
The parser rejects noncausal indices, frozen-bit references, conflicting formats,
and information positions with frozen expressions. See [conventions](../docs/CONVENTIONS.md).

The custom-profile driver is a **bounded first-domain construction**, not a promise
to solve every feasible profile. `--recipe word`, `nested`, and `gap` expose the
actual alternatives used by the examples. A failure or timeout is not global
infeasibility. The complete all-domain specification is executable on tiny codes
in `theory/exhaustive.py` and checked against exhaustive support enumeration.

## Exact Spectra and Complete Words

```bash
python main.py enumerate --support results/n256_k128/support.json --cutoff 24 --method decomposition --output runs/a24
python main.py enumerate --support results/n64_k32/support.json --cutoff 12 --method words --output runs/word_list
python main.py enumerate --support results/n64_k32/support.json --cutoff 12 --method native --output runs/native
python examples/enumeration_comparison.py --output runs/benchmark
```

`decomposition` counts disjoint left cosets and their exact right-shortening
multiplicities without listing all full-code words. `words` lists every nonzero
input/codeword through the chosen cutoff in `words.jsonl`. Both are exact only
on completion. `native` and `python` expose the upstream independent counters.
`quotient` and `direct` expose the readable complete-directory implementations
used in the matched benchmark. The zero word is counted but omitted from word lists.

Do not confuse a cutoff spectrum with the entire weight enumerator. Absence of a
positive coefficient through `D` establishes `dmin >= D+1`, not an exact distance.
No list-decoder spectrum estimate is labeled exact here. `--work` bounds native
work per invocation. An exhausted invocation returns no claimed exact spectrum.

## SCL and Monte Carlo

```bash
python examples/monte_carlo.py --support results/n64_k32/support.json --snr 3 --list-size 64 --frames 10000 --messages random --output runs/fer
python main.py simulate --support results/n256_k128/support.json --snr 3 --list-size 64 --frames 1000 --backend cpp --output runs/fer256
```

The decoder accepts arbitrary valid causal supports and arbitrary transmitted
messages. It is **min-sum/max-log SCL without CRC**, not exact log-domain SCL.
Every path enforces the full frozen expression, with no limit of three sources.
`--snr` is **Eb/N0**, BPSK maps 0 to +1, and `sigma^2 = 1/(2 R Eb/N0)`.
The simulator compares decoded information bits, reports FER, BER and a 95%
Wilson interval, and uses a fixed frame count with a reproducible noise seed.
Small smoke runs are functionality checks, not publication-quality FER estimates.

Equal-metric pruning has been corrected in this reader release: exactly `L`
branches are retained, using `(metric, bit, parent-path index)` for ties.
The original research files and historical FER logs were not modified or rerun.
See [decoder notes](../docs/DECODER.md) before comparing old simulation outputs.

## Mechanisms, Verification and Scope

```bash
python examples/toy_mechanisms.py --output runs/toy
python examples/verify_theory.py
python examples/intermediate_distance.py --output runs/intermediate
```

The N32 experiments exhaustively compare complete mother constraints, replacement
of old low-weight words, overlapping conditions, and conditional-mean completion.
They emit JSON, PNG figures and editable PGFPlots files. The enumeration benchmark
checks exact input-set equality before timing matched direct/quotient traversals.

The historical distance-20 example verifies a stored, previously seeded code on
the same N256/K128 profile. It is **not** a new zero-support construction or a
controlled search comparison. This distinction is preserved in its metadata.

The library is parameterized by profile and support, but the native spectrum
representation currently supports **N <= 256**. The SCL decoder is not tied to
these five profiles. Large lengths, dense released-row patterns, high projected
rank and large error radius can exceed practical limits. See
[algorithm scope](../docs/ALGORITHMS.md) and [experiment map](../docs/EXPERIMENTS.md).

## Attribution

The exact spectrum dependency is retained with its own MIT license and citation
metadata. Binary elimination, MacWilliams identities, standard polar recursion,
RM weight gaps, and Moore checks are attributed mathematical ingredients, not
claims of new general results. [Provenance](../docs/PROVENANCE.md) distinguishes reused
kernels, packaging changes, and the decoder correction. A public license for the newly organized author code has not been selected.
