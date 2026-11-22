# Section VI measurements

English · [简体中文](README.zh-CN.md)

| File | What the rows contain |
|---|---|
| [fer.csv](fer.csv) | Independent FER measurements for Figs. 5–10. Use `case` to match each point to its code configuration. Each row gives the decoder list size, SNR, frame count, error count and recorded confidence sequence. |
| [bounds.csv](bounds.csv) | The normal approximation and meta-converse for each plotted binary-input AWGN channel and payload rate. Both are channel benchmarks and can be computed from the channel and rate alone. |
| [table_ii.csv](table_ii.csv) | Distances, minimum-weight multiplicities, GA profile statistics at 2.25 dB and the two construction times printed in Table II of r112. |
| [prefix_diagnostics.csv](prefix_diagnostics.csv) | The paired noise frames in Table III. Divide each event count by `common_frames` to get the rate of correct-path pruning or incorrect final selection. Fig. 10 uses separately sampled frames for its FER curves. |

## Reproducing the figure inputs

Of the 274 measured points in `fer.csv`, the 269 with `displayed=1` fall within the published plotting ranges. Fig. 9 ends at 4.25 dB. For lengths 64–256, the code, information-set and frozen-expression hashes were checked when the measurements were imported. At length 1024, the original curve label and decoder list size identify the entry in the archived code library.

From the repository root, run `python -m reproduction check` to recompute the simulator's code hash from each final configuration. Run `python -m reproduction.table_ii_values` to recompute the unprecoded minimum distance and both GA profile statistics for all eight rows. These statistics use a common comparison SNR of 2.25 dB.

## Reading the construction times in Table II

**w/o refinement** subtracts the measured refinement stages from the same run. By that point the reliability-guided information-set search and coefficient construction have already produced a code meeting the distance target. **w/ refinement** also includes the subsequent spectrum refinement and prefix protection. Each length-1024 variant includes the shared main-code stage once. The table reports the original sequential measurements. Local timings depend on the machine and software environment.
