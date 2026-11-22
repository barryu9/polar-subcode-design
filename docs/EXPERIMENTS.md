# Earlier Five-Case Experiment and Verification Map

For the codes used in r112 Figs. 5–10, see the [curve configurations](../configs/README.md).

| Task | Entry point | Required inputs | Output / meaning |
|---|---|---|---|
| Five zero-support designs | `examples/n*_k*.py` | Matching `profile/` JSON | New support, exact distance, available first coefficient |
| Full-design timing replays | `examples/repeated_designs.py` | All five profiles | Sequential times and identical-support checks |
| Replacement of old words | `examples/toy_mechanisms.py` | Built-in N32 domain | Complete 14-bit support-space counts |
| Overlapping constraints | Same toy entry | Built-in N32 domain | Complete 21-bit support-space compatibility counts |
| Conditional-mean completion | Same toy entry | Same complete event system | Exact prefix means and successful assignment |
| Direct versus quotient lists | `examples/enumeration_comparison.py` | N64 profile and archived N256 fixed constituents | Complete input-set equality, alternating timings, traced allocations |
| Intermediate distance 20 | `examples/intermediate_distance.py` | Archived historical seeded support | Exact `1 + 4042 X^20`, not a construction comparison |
| Outer conditional-mean trace | `main.py plot-run <run>` | A fresh unrestricted outer-completion log | Editable JSON trace and PNG |
| FER/BER | `examples/monte_carlo.py` | Any valid support JSON | Measured error counts and confidence interval |
| Principal theoretical claims | `examples/verify_theory.py` | Independent tiny profiles | Exhaustive finite checks, not substitutes for proofs |

The pure-right/mixed decomposition for the N256/K128 construction is retained in
`spectrum/stages` outputs. Complete A24 counting includes weight-24 left words,
not just the weight-16 left words sufficient for the distance proof.

## Theory Coverage

- Shared support rejection, full mother versus old-word-only constraints,
  affine support-image multiplicities and conditional means.
- Disjoint relaxation shells, every legal source assignment, earlier-source
  repair, and complete tiny-code left/right/outer construction.
- Weight-free coordinate projection and restoration of every kernel lift.
- Relative-coset spectra, repeated syndrome columns, and exact subset-join counts.
- Joint interface compatibility and failures invisible to separate conditions.
- Retained-core shortening ranks and conditional outer-extension lower bounds.
- Causal auxiliary-check realization and the algebraic positive-rate subfamily.
- Canonical JSON, encoder consistency, decoder ties and nonzero-message decoding.

Some tests use fixed pseudorandom seeds to generate additional test inputs.
The five construction examples themselves use no random sampling.

## What Is Not Included

The repository does not bundle unrelated A2C training, exploratory dead ends,
downloaded literature PDFs, account data, or multi-million-frame historical FER
jobs. Such jobs were not part of the current deterministic-construction results.
The SCL/Monte Carlo utilities allow a reader to run new simulations explicitly.
The historical distance-20 illustration is retained as a verifiable artifact,
not presented as a deterministic zero-start reproduction.
