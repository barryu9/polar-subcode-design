# Finite checks accompanying the proofs

**English** · [简体中文](README.zh-CN.md)

From the repository root, run all checks or select a result. Each function in [check_results.py](check_results.py) is named after its numbered result. The checks use Python's standard library and the repository's binary algebra. They compare the predictions in the paper with the actual words, assignments, subspaces and preimage counts of small codes.

```bash
python -m theory.check_results
python -m theory.check_results theorem2 proposition3 corollary1
```

| Manuscript result | Independent comparison |
|---|---|
| Proposition 1, (8) | Group length-eight configurations by the retained coefficients and compare the sets of low-weight words within each group. |
| Theorem 1, (13)–(14) | Vary four coefficients around a nonzero baseline. Compare the per-pattern residual counts with directly encoded words in each forbidden layer to check both joint exclusion and weight-by-weight counting. |
| Theorem 2, (23)–(25) | For all 16 outer matrices of a small fixed split, compare the complete weight enumerators and the spectrum-preserving translations. Enumerate all possible linear kernels to check the minimum rank. |
| Proposition 2, (30) | Compare Algorithm 1's outputs with sorted lists of directly enumerated inputs for 502 code/cutoff instances. Sorting preserves repeated entries so duplicates are detected. |
| Proposition 3, (33)–(34) | Reconstruct the full feasible set from representatives and kernel vectors. Compare joint choices per source, affine fibers with linked columns, and restricted columnwise images. |
| Corollary 1, (35) | Enumerate all 307 affine domains of a four-parameter system. Compare exact word counts at every point with the condition that every message's reachable set avoids its forbidden set. |
| Theorem 3, (36), (39)–(40) | Compare conditional means with observed violations at every prefix under both the fixed-constituent and common-supercode conditions. Where the strict bound applies, follow the child with smaller mean. The next check also counts continuations for each accepted assignment of earlier columns. |
| Corollary 2, (45)–(46) | Rebuild Example 7's right-code containment, inside nullities, source bounds and completion counts. Of the 16,384 outer matrices, 13,472 attain the distance target, compared with the uniform lower bound of 13,440. |

For the independent word counts, the checker computes the polar matrix entry by entry and encodes every message directly. This gives a comparison for Algorithm 1's quotient and lifting operations. In the length-32 count, the checker uses quotient checks after verifying the containment and rank hypotheses.
