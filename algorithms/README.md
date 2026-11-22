# The four algorithms in manuscript order

**English** · [简体中文](README.zh-CN.md)

Run these commands from the repository root. Each module contains a callable function and a small example. The implementations use the paper's row-vector polar transform and zero-based indexing. In binary terminal output the rightmost bit has index zero.

| Paper | Run | What to inspect |
|---|---|---|
| Alg. 1, §III.C, (30) | `python -m algorithms.algorithm1_directory` | `list_inputs(code, cutoff)` handles the repeated right codeword, enumerates the outside bits and lifts each accepted assignment. It returns inputs $u$ whose transformed words $x$ meet the weight cutoff. Use `from_causal_checks(N, rows)` to substitute earlier frozen expressions into the check rows. |
| Alg. 2, §IV.B, (24), (34), (41) | `python -m algorithms.algorithm2_outer` | `complete_outer(left, right, cutoff, domains)` forms the forbidden sets, selects a basis in reverse source order and searches forward by conditional mean. Each column domain supplies a base column and its independent directions of variation. The result is a feasible outer matrix or `None` after exhaustive failure. |
| Alg. 3, §V.B, (48), (52)–(53) | `python -m algorithms.algorithm3_fixed_profile` | `construct(N, profile, target)` checks the initial domain, enlarges it after proved failure, and verifies the distance of any code returned. The `direct` method uses the common-supercode conditions. The `factored` method enumerates constituent pairs and calls Algorithm 2. |
| Alg. 4, §V.C–D, (59), (61)–(62) | `python -m algorithms.algorithm4_profiles` | `design_profiles(...)` ranks single exchanges using the caller's reliability order, costs and candidate family. It groups candidates by suffix, proposes up to $Q$ distinct information sets, and retains verified stage endpoints. |

## Follow the feedback

Algorithm 3 accepts optional `suffix_lengths` and a work callback `tick`. Raising `Deferred` stops the current construction and returns the suffix reports computed so far. Exhausting a coordinate domain allows the algorithm to release another coefficient. Exhausting the whole profile gives `infeasible`. Algorithm 4 uses the smallest unresolved suffix and the largest verified shorter suffix to prioritize exchanges. It can exclude other profiles with the same suffix when the obstruction has been proved independent of the choice of coefficients.

## Refinement and design observations

The teaching implementation of Algorithm 4 uses the whole-frame rule in (62) when supplied with an SCL loss measurement. Create one with `frame_loss_measure(llrs, list_size)` using design-frame LLRs for transmitted all-zero words. Omitting the measurement runs spectral refinement and returns that stage's endpoint. For the **Proposed** code reported in Section VI.C, the construction instead selects each editable expression by its local prefix score, then measures whole-frame loss. See the [reproduction commands](../README.md).

The direct solvers build the low-weight directory and may traverse every free parameter. Start with a small code to inspect the search tree. The tests include every length-four profile, affine outer-column domains, a deferred search with a partially resolved suffix tree, and an empty initial proposal.

```bash
python -m unittest discover -s tests -p test_paper_algorithms.py -v
```
