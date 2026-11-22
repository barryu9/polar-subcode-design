# Worked Examples 1–9

**English** · [简体中文](README.zh-CN.md)

Run the commands from the repository root. Numbering follows the manuscript, including the extension to cutoff 4 in Example 4. Examples 1–6 share the same length-16 code and two frozen expressions. Each example's main calculation is in one Python file.

```shell
python -m paper_examples all
python -m paper_examples 4
python -m paper_examples.example07
python -m unittest paper_examples.test_examples
```

| Example | Script | What to inspect |
|---|---|---|
| 1 | [example01.py](example01.py) | Construct the input checks and generator, then apply the polar transform. |
| 2 | [example02.py](example02.py) | Toggle a later coefficient at cutoff 3 and recover a coefficient tied to it. |
| 3 | [example03.py](example03.py) | Enumerate all 64 supercode inputs, group nine weight-2 words and test all 16 designs to recover the four feasible configurations. |
| 4 | [example04.py](example04.py) | Embed the two released checks into the seven-check space and compute the two quotient ranks. Lift the accepted outside assignment into both weight-4 codewords. |
| 5 | [example05.py](example05.py) | Restrict the rank-11 joint matrix to four coordinates, obtain the rank-2 map and recover all designs represented by the feasible representative. |
| 6 | [example06.py](example06.py) | Compare the unconditional mean with the two source-group means and select the coefficients. |
| 7 | [example07.py](example07.py) | Count all 16,384 outer matrices, check both bounds and follow both paths in Fig. 3. |
| 8 | [example08.py](example08.py) | Complete the specified $[256,128]$ profile, compare the last-column means, bound the completion count and check the weight-24 witness. |
| 9 | [example09.py](example09.py) | Solve both length-32 domains and use the suffix feedback to make two exchanges. Solve the outer conditions to obtain the distance-48 code. |

## Rebuild the two length-256 examples

The default walkthroughs finish quickly. Example 8 reads the recorded critical-message directory and conditional means, recomputes the joint rank and completion-count bound, and encodes the distance witness. Example 9 reads the recorded GA ordering and internal expressions, solves the two length-32 domains, rebuilds the parent's forbidden sets, and checks the stored parent coefficients against all of them. The two code JSONs list the contributing information indices at each dynamic frozen position.

```shell
python tools/build_native.py
python -m paper_examples.example08 --construct --verify-spectrum
python -m paper_examples.example08 --recompute-means
python -m paper_examples.example09 --construct --verify-spectrum
```

- **Example 8.** `--construct` starts from the RM–GA information set and the seven exchanges given in Section V. It constructs the right code with all coefficients initially zero, generates the complete left directory, and computes the outer columns in source order. It checks every resulting coefficient against the saved code. `--recompute-means` independently counts forbidden values in each reachable affine image for the three reported prefixes. `--verify-spectrum` independently enumerates the left inputs and initial right supercode, checks the constructed right distance, and tests the 920 nonzero-left cosets through weight 23. The reported 354,383,872 completions refer to the last 33 bits after fixing the preceding 38 columns and six bits of the last column. The saved spectrum through weight 24 belongs to this fixed-profile example. Section VI uses a separately selected profile for its $[256,128]$ **Proposed** curve.
- **Example 9.** `--construct` computes RM–GA at design $E_b/N_0=2$ dB, reruns the initial suffix screen and both exchanges, rebuilds the length-64 and length-128 codes, and solves the 48 outer coefficients. It compares every reconstructed coefficient to the recorded one. The initial failed search certifies only the 20-coefficient domain on the length-32 suffix. `--verify-spectrum` checks RM membership and exactly counts the lower weight layer for each suffix and the full code. Together with the verified weight-48 word this gives the distance claimed in the paper.

See the inputs and counts for [Example 8](data/example08_trace.json) and [Example 9](data/example09_trace.json). Each `--construct` routine generates the coefficients before comparing them with the [Example 8 code](data/example08_code.json) or [Example 9 code](data/example09_code.json).
