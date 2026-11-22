# Reconstructing the Proposed codes in Figs. 5–10

English · [简体中文](README.zh-CN.md)

## Run a construction

Run these commands from the repository root in an activated CPython 3.12 environment.

```sh
python -m pip install -r reproduction/table_ii/requirements.txt
python reproduction/proposed.py check
python reproduction/proposed.py cold --cases n256_k64 --output runs/short
python reproduction/proposed.py cold --output runs/table-ii
```

The first command installs the numerical and solver versions used for the replay. For cold construction, put a C++17 compiler (GCC or Clang) on your PATH or point `CXX` to it. On Windows the CPython SCL extension also uses the MSVC build tools. The driver builds the native kernels from the included sources before it starts the construction timer.

`check` verifies the identities and causality of the nine supplied profile/code pairs. `cold` starts a new search in a fresh output directory. Start with `n256_k64` to follow Example 9. Without a case filter, `cold` runs the eight constructions in Table II and also extracts the Proposed-NP ablation from the main Fig. 10 run.

## Find the driver for a figure

Each Python driver specifies the stages, channel choices, candidate budgets and seeds for its case. The functions under [python/](python/) perform the searches, spectrum tests and decoding-prefix scoring. The six main cases each start with empty expression and candidate archives. Cases of the same length may use different information-set searches. Example 8 and the Fig. 8 Proposed code both attain distance 24 at [256,128] using different information sets. Example 9 constructs the [256,64] Proposed code plotted in Fig. 7.

| Figure | Curve and dimensions | Python driver |
|---|---|---|
| 5 | Proposed [64,32] | [Structural feedback, exchanges and refinement](examples/reproduce_n64_k32.py) |
| 6 | Proposed [128,64] | [Ranked profiles and core spectrum refinement](examples/reproduce_n128_k64.py) |
| 7 | Proposed [256,64] | [Reliability-guided profile exchanges and coefficient construction](examples/reproduce_n256_k64.py) |
| 8 | Proposed [256,128] | [Profile search and joint refinement](examples/reproduce_n256_k128.py) |
| 9 | Proposed [256,171] | [Profile selection, whole-row and local refinement](examples/reproduce_n256_k171.py) |
| 10 | Proposed and Proposed-NP [1024,512] | [Profile search, suffix refinement and prefix protection](examples/reproduce_n1024_k512.py) |
| 10 | Proposed-LM [1024,512] | [Low-multiplicity continuation](examples/reproduce_n1024_k512_lm.py) |
| 10 | Proposed-D32 [1024,512] | [Distance-32 continuation](examples/reproduce_n1024_k512_d32.py) |

## Read the generated configurations

Once the driver has frozen its candidate archive, the verifier compares the SHA-256 hashes of each complete information set and its coefficient columns with those of the plotted code. It exports each matching pair under `runs/table-ii/selected/`. To export the pairs again from that run, use `python reproduction/proposed.py collect --output runs/table-ii`.

A `*.profile.json` file holds the chosen information positions at the start of fixed-profile coefficient design. The paired `*.code.json` adds a list of earlier information positions for each dynamic frozen position. Their bits are XORed to obtain its value. Unlisted frozen positions carry zero. All indices are zero-based. Use `polar_subcodes.code.load_code` to load either file, including the [distributed configurations](../../configs/proposed).

## Follow the length-1024 variants

Proposed-NP is the suffix-refined code before prefix protection. Proposed combines this suffix with a spectrum-preserving prefix. Proposed-LM and Proposed-D32 both start from the newly computed main-code candidate `combined_02_floor16`. Selecting either variant with `--cases` automatically runs the shared construction. The [variant driver](examples/reproduce_n1024_variants.py) records the exchanged positions, design-noise seeds and search budgets.

The final Proposed-D32 prefix step preserves the spectrum through weight 31. The weight-32 multiplicity in Table II comes from an independent complete count on the matched code. To repeat it, take the logged command for the [final stage](python/run_ga_priority_d32_origin_prefix.py), choose a fresh output folder and omit `--defer-spectrum-verification`.

## Interpret the timings

`table_ii_times.csv` reports the two cumulative times used in Table II. **w/o refinement** includes the reliability-guided information-set search and coefficient construction, which already produce a code meeting the distance target. It also counts separately timed assembly and verification. **w/ refinement** adds spectrum refinement and prefix protection. Each length-1024 variant includes the shared main-code stage once.

`stage_breakdown.csv` gives seconds and percentages by stage. `paper_step_summary.csv` groups them by the operations described in the paper. The [published timings](published_timings/table_ii_times.csv) come from the original sequential run. The protocols take earlier design decisions as inputs and measure automated search, decoding on design samples and in-run verification. The FER curves use independent simulations. Wall-time limits and numerical ties can affect which candidates a machine reaches. The replay reports an error if the completed archive lacks the plotted code.
