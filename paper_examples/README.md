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
| 2 | [example02.py](example02.py) | At cutoff 3, changing $\alpha$ preserves the low-weight codeword set. Changing $b_1$ can bring a word below the cutoff. Recover $\alpha$ when the domain ties it to $b_1$. |
| 3 | [example03.py](example03.py) | Enumerate all 64 supercode inputs, group nine weight-2 words and test all 16 designs to recover the four feasible configurations. |
| 4 | [example04.py](example04.py) | Embed the two released checks into the seven-check space and compute the two quotient ranks. Lift the accepted outside assignment into both weight-4 codewords. |
| 5 | [example05.py](example05.py) | Restrict the rank-11 joint matrix to four coordinates, obtain the rank-2 map and recover all designs represented by the feasible representative. |
| 6 | [example06.py](example06.py) | Compare the unconditional mean with the two source-group means and select the coefficients. |
| 7 | [example07.py](example07.py) | Count all 16,384 outer matrices, check both bounds and follow both paths in Fig. 3. |
| 8 | [example08.py](example08.py) | Complete the specified $[256,128]$ profile, compare the last-column means, bound the completion count and check the weight-24 witness. |
| 9 | [example09.py](example09.py) | Follow the two exchanges in the paper, compare the length-32 domain outcomes, and check the parent exclusion conditions for the distance-48 code. |

## Follow one message through Examples 3–6

In Example 3, find the rows with source pattern `011` in `directory_(input,output,q,beta)`. The bit string lists $(u_3,u_4,u_5)$ from left to right. In Python it is the integer `6` because bit 0 stores $u_3$. The last field gives the two released-check residuals $(u_8,u_9)$. A residual matching $(b_0+b_1,c)$ leaves that word in the code.

For the same message, Example 4 reports `residual_counts = {"00": 1, "11": 1}`. Each of these residuals has one compatible inside assignment at cutoff 3. Avoiding both requires $b_0+b_1+c=1$. The other critical message, `010`, requires $c=1$. Example 5 takes these two observations and recovers exactly the four `feasible` assignments listed in Example 3. Example 6 then constructs a representative by choosing $c$ at source 4 and $b_1$ at source 5. Its `violations_for_children` lists violation counts for candidate bit values 0 and 1, in that order.

## Rebuild the two length-256 examples

The default walkthroughs finish quickly. Example 8 reads the recorded critical-message directory and conditional means, recomputes the joint rank and completion-count bound, and encodes the distance witness. Example 9 reads the recorded GA ordering and internal expressions, solves the two length-32 domains, rebuilds the parent's forbidden sets, and checks the stored parent coefficients against all of them. The two code JSONs list the contributing information indices at each dynamic frozen position.

```shell
python tools/build_native.py
python -m paper_examples.example08 --construct --verify-spectrum
python -m paper_examples.example08 --recompute-means
python -m paper_examples.example09 --construct --verify-spectrum
```

- **Example 8.** `--construct` starts from the RM–GA information set and the seven exchanges given in Section V. It constructs the right code with all coefficients initially zero, generates the complete left directory, and computes the outer columns in source order. It checks every resulting coefficient against the saved code. `--recompute-means` independently counts forbidden values in each reachable affine image for the three reported prefixes. `--verify-spectrum` independently enumerates the left inputs and initial right supercode, checks the constructed right distance, and tests the 920 nonzero-left cosets through weight 23. The reported 354,383,872 completions refer to the last 33 bits after fixing the preceding 38 columns and six bits of the last column. The saved spectrum through weight 24 belongs to this fixed-profile example. Section VI uses a separately selected profile for its $[256,128]$ **Proposed** curve.
- **Example 9.** `--construct` computes the RM–GA order at design $E_b/N_0=2$ dB, screens the initial profile against the suffix bounds and applies the two exchanges specified in the example. It solves the length-64 and length-128 domains and all 48 outer coefficients, then checks the reconstructed expressions. The first search excludes the 20-coefficient domain on the length-32 suffix. `--verify-spectrum` checks RM membership and counts the lower weight layer of each suffix and the full code. Together with the weight-48 witness this establishes the minimum distance of the $[256,64]$ **Proposed** code in Fig. 7.

See the inputs and counts for [Example 8](data/example08_trace.json) and [Example 9](data/example09_trace.json). Each `--construct` routine generates the coefficients before comparing them with the [Example 8 code](data/example08_code.json) or [Example 9 code](data/example09_code.json).
