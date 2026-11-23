# Finite checks accompanying the proofs

**English** · [简体中文](README.zh-CN.md)

Run the commands below from the repository root. The first runs all eight checks in paper order. The second follows one constituent pair from its joint observation through coefficient recovery to the distance test on an affine domain. The script prints a summary for each result once its comparisons pass. [Theory checks](check_results.py)

```bash
python -m theory.check_results
python -m theory.check_results theorem2 proposition3 corollary1
```

**Following one outer matrix.** Start at `theta_data()` in the script. The left and right constituents each have length four. Each of the two left sources contributes one column of two bits to the outer matrix $T$, giving 16 possible matrices. For each assignment $t=\operatorname{vec}(T)$, `observations[t]` contains the syndrome observations $Q_a Tm$ of the critical messages. A separate calculation encodes every parent-code message directly and stores the counts $(A_0,A_1,A_2)$ in `spectra[t]`. This lets you trace what each function checks against the codewords themselves.

Theorem 2 reports rank 2 at cutoff $D=2$. Thus four joint observations are attainable. Proposition 3 retains one representative per observation. Adding the four members of the joint kernel to a representative recovers all four matrices with that observation. Corollary 1 uses the same 16 matrices to check which affine domains meet the distance target at **every** point.

| Manuscript result | What the code compares |
|---|---|
| Proposition 1, (8) | Group length-eight configurations by the retained coefficients and compare the sets of low-weight words within each group. |
| Theorem 1, (13)–(14) | Vary four coefficients around a nonzero baseline. Compare the per-pattern residual counts with directly encoded words in each forbidden layer to check both joint exclusion and weight-by-weight counting. |
| Theorem 2, (23)–(25) | For all 16 outer matrices of a small fixed split, compare the complete weight enumerators and the spectrum-preserving translations. Enumerate all possible linear kernels to check the minimum rank. |
| Proposition 2, (30) | Compare Algorithm 1's outputs with sorted lists of directly enumerated inputs for 502 code/cutoff instances. Sorting preserves repeated entries so duplicates are detected. |
| Proposition 3, (33)–(34) | Reconstruct the full feasible set from representatives and kernel vectors. Compare joint choices per source, affine fibers with linked columns, and restricted columnwise images. |
| Corollary 1, (35) | Enumerate all 307 affine domains of a four-parameter system. Compare exact word counts at every point with the condition that every message's reachable set avoids its forbidden set. |
| Theorem 3, (36), (39)–(40) | Compare conditional means with observed violations at every prefix under both the fixed-constituent and common-supercode conditions. Where the strict bound applies, follow the child with smaller mean. The next check also counts continuations for each accepted assignment of earlier columns. |
| Corollary 2, (42)–(46) | Verify the containment and rank hypotheses of Example 7, then count how many columns extend each accepted partial assignment under the projected checks. |

**Reading the Example 7 count.** Run `python -m theory.check_results corollary2`. The output `Lambda=[8, 16], lower bound 13440, exact 13472` refers to outer matrices. Each source has 128 candidate columns. At most 8 choices can be excluded at the first source, and at most 16 at the second for each accepted first column. Their product bound is $(128-8)(128-16)=13{,}440$. Applying the actual forbidden sets counts 13,472 matrices attaining $d_\star=6$. For the construction that uses these same source groups, see [Example 7](../paper_examples/example07.py).
