# Reader Release Validation

The five examples were run sequentially from zero support in the independent
reader repository. No answer was opened until each construction and its exact
verification had finished. Every resulting support matched the canonical one.

| Profile | Initial d_P | Verified distance | Exact first coefficient | Fresh wall time (s) |
|---|---:|---:|---:|---:|
| N64/K32 | 8 | 12 | 3,280 | 0.32 |
| N128/K42 | 16 | 28 | 16,128 | 9.56 |
| N256/K64 | 32 | 48 | Not counted | 57.23 |
| N256/K128 | 16 | 24 | 344,220 | 85.94 |
| N256/K171 | 8 | 12 | 19,456 | 3.70 |

These are single fresh runs on the author's Windows computer, not speed
guarantees or replacements for repeated manuscript measurements. Reader-driver
timings include additional checks and output I/O. Hardware identification and
package versions are in `results/release_validation/manifest.json`.

The N256/K64 distance follows from complete exclusions, verified weight-gap
membership, generator rank and a legal weight-48 witness. Its A48 remains unknown
in this release. No absent coefficient is represented as a known zero.

## Independent Checks

- 61 tests passed, with no failures, errors or skipped native checks.
- Small-code exhaustive messages independently check the native and Python
  counters, direct and quotient word directories, and decomposed spectra.
- The complete tiny-code construction is compared with all legal supports.
- Enumeration work-limit exhaustion is tested as an error, never a zero spectrum.
- NumPy and C++ SCL agree on noisy frames and recover nonzero noiseless messages.
  A full-list tiny decoder is compared with exhaustive maximum likelihood.
- The complete historical distance-20 count reproduces A20 = 4,042.
- N32 mechanism experiments enumerate their whole finite support spaces.
- The matched direct/quotient benchmark checks complete input-set equality,
  warms both methods, alternates their timing order, and measures allocations
  separately from wall times.
- A 1,000-frame random-message N64/K32 native SCL run at 3 dB and L = 64
  completed with four frame errors. This is a functionality check only.

Compact auxiliary records are under `results/validation_tools/`. Full fresh
working files are under the ignored `runs/` directory and can be regenerated
using the entry points in `docs/EXPERIMENTS.md`.

## Interpretation

The deterministic first-domain driver is the tested production path. It is not
a complete all-domain implementation for arbitrary large profiles. The complete
specification is exercised on tiny codes. A failed bounded search does not prove
global infeasibility.

The reader release fixes equal-metric SCL pruning. Historical research FER
outputs were not rerun and are not revalidated by these tests. Construction and
exact spectrum checks do not depend on that decoder.
