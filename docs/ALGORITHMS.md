# Algorithms and Implementation Scope

## Common-Mother Rejection

`construction/source_patterns.py` retains shared source variables. For each
complete mother word it records the source pattern and frozen-check defect.
The same support assignment must reject all forbidden defects. Fixed-coordinate
and leading-position filtering preserve the entire set of possible candidates.
Multiplicity is retained when words share a condition.

`construction/domains.py` selects released rows of weight at least the baseline
row minimum and retains high-weight information rows as a pure core.
`theory/exhaustive.py` implements disjoint domain shells, complete source
alternatives, constituent pairs, and outer backtracking on tiny instances.
It is intentionally exponential and is not the production driver for N256.

## Outer Completion

`spectrum/relative.py` uses
`wt(a+b,b) = wt(a) + 2 wt(b outside supp(a))`.
It projects parity checks modulo the weight-free coordinates. Each admissible
outside assignment has exactly `2^shortening_dimension` legal lifts.

`construction/completion.py` keeps every unvisited bit-prefix sibling. Exact
conditional means order branches. A mean below one is sufficient for a
completion of that group. A mean at least one is not a contradiction.
`pipeline.complete_outer` returns to an earlier source if its selected column
blocks a later source. Right-constituent changes are outside this fixed-pair call.

Restricted outer systems use `construction/bounded.py`. Projected coset-event
multiplicity counts violated events, not codewords. This distinction is recorded
in the experiment data. A weight gap is used only after checking actual mother
membership and the admissible row positions.

## Exact Enumeration

`spectrum/quotient.py` is a readable complete-word implementation. It factors the
inside-coordinate map, tests outside assignments in the quotient, and emits all
kernel lifts. The direct variant tests all inside assignments. Both return the
same complete input set. The upstream C++ counter can terminate affine families
without expanding them and is retained as a separate implementation.

`spectrum/decomposition.py` combines complete left directories with exact right
spectra. It uses integer MacWilliams transforms when the dual dimension is at
most 16, and exact syndrome subset counts for small punctured-coset radii.

## Deliberate Limits

| Component | Current practical scope |
|---|---|
| Native counter / complete word kernel | `N <= 256` |
| Dense source-pattern domains | At most 12 released rows |
| Unrestricted outer prefix solver | Radius <= 3, projected rank < 63 |
| Dense projected error ball | Rank <= 22, radius <= 6 |
| Sparse projected error ball | Radius <= 3, Python integer labels |
| Nested projected directories | Each explicitly enumerated constituent has dimension <= 20 |
| Relative histogram counts | Optimized radius <= 4, subset joins <= 6 |
| SCL | General power-of-two lengths, limited by time and memory |

The constructor's time/node limits govern the search loops, not preprocessing
or an entire process deadline. `--work` is a per-native-call work allowance,
not a hard wall-clock or byte limit. Preprocessing and word materialization can
still be expensive. The complete-word API currently materializes its bounded
directory before export. It is not a constant-memory streaming enumerator.

All success reports require completed exact checks. A resource-limited return,
an exhausted first domain, or a failed constituent pair does not prove that
the profile admits no good support. No universal polynomial-time guarantee is made.
