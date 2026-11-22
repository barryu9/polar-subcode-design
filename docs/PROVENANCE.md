# Provenance and Changes During Extraction

The repository was extracted from the authors' research workspace. The machine-
readable `source_manifest.json` maps reused kernels and datasets to their original
relative filenames and source hashes. It contains no absolute workspace paths.
That manifest records the source at extraction, not the hash of the subsequently
cleaned implementation.

## Reused Numerical Components

- Source-pattern propagation and deterministic product ordering.
- Exact projected coset conditions, conditional-mean prefix backtracking and
  binary image-coordinate maps.
- Exact relative spectra, subset-join counting and quotient input directories.
- Exhaustive N32 toy experiments and independent numerical theorem checks.
- The original vectorized and native SCL numerical kernels.
- `polar-low-weight-spectrum` v0.1.0, retained under its MIT license with its own
  citation metadata. The complete word wrapper uses that kernel's generation
  interface. This dependency is not silently relabeled as a new contribution.

## Reader-Repository Adaptations

- Removed imports of learning environments, historical search drivers and
  manuscript formatting tests. No PyTorch or Z3 is needed by these examples.
- Replaced machine-specific DLL loading with local portable builds.
- Linked MinGW runtimes statically and checked actual library loading during builds.
- Added a validated lossless profile/support interchange format.
- Unified zero-support drivers, progress logs, exception handling and custom inputs.
- Generalized nested cross-only domains by split structure instead of a fixed length.
- Retained a sparse exact error-ball path when projected rank is too large for
  the dense Boolean table. This is essential for the N256/K64 design.
- Added checked enumeration resource limits and independent spectrum comparisons.
- Corrected equal-metric SCL pruning in both backends, documented separately.
- Made native SCL error cleanup conform to standard C++ declaration rules for
  GCC as well as MSVC.
- Retained the original five canonical supports unchanged and compared fresh
  outputs with them after construction.

## Attribution Boundaries

Standard binary elimination, polar recursion, MacWilliams transforms, conditional
expectation, RM minimum-weight gaps and Moore determinants remain standard or
attributed ingredients. The files implement their use in this support-design
framework. See the accompanying manuscript and the dependency's documentation
for the corresponding mathematical references.

The original author code and the vendor dependency have separate terms.
See the current [notice](../NOTICE.md). The source manifest in this directory
describes the earlier five-case release. The r112 curve manifests are under
[`configs/`](../configs/README.md).
