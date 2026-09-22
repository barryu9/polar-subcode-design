# Algorithm 1: formulas, implementation map, and complexity

The reference is the accompanying research manuscript, *Deterministic
Low-Weight Spectrum Computation of Polar Subcodes via Affine Coset Counting*,
Algorithm 1, in the version with the “Overall Algorithm and Complexity” section.
Equation labels below are its LaTeX labels, to avoid unstable equation numbers.

## Code model and normalization

All algebra in constraints and transforms is over GF(2):

$$
x=uG_N,\quad G_N=F^{\otimes\log_2N},\quad
F=\begin{bmatrix}1&0\\1&1\end{bmatrix},\quad Hu^\mathsf T=0.
$$

Highest-column pivot elimination turns H into causal checks: each pivot
input is a linear combination of earlier inputs. For shortened coordinates S,
append $E_SG_N^\mathsf T$ to H and eliminate again. Generate a basis for this
shortened code and project its words onto T. Select independent projected
rows, considering free input indices in descending order, and freeze the
unselected free inputs. The resulting subcode $C^\star$ projects bijectively
onto the transmitted code. This prevents counting puncturing-kernel copies.

Use $D'=\min(D,|T|)$ and

$$
q_j=\begin{cases}1&j\in T\\0&j\in P\\D'+1&j\in S.\end{cases}
$$

The extra shortening cost agrees with the threshold restriction; shortening
constraints have already been imposed. The zero word is counted once.
Every other representative word has a unique first nonzero free input:

$$W_{\le D}=1+\sum_{i\in I^\star}\operatorname{Count}(N,i,q).$$

## Count and Gen

For an interval of length $M=2m$, split $q=(\alpha,\beta)$ and
$u=(a,b)$. With $s=aG_m,t=bG_m$,

$$x=(s+t,t).$$

The universal lower bound ignores future frozen constraints:

$$
\underline d_{1,0}(q)=q_0,\qquad
\underline d_{2m,i}(\alpha,\beta)=
\begin{cases}
\underline d_{m,i}(\min(\alpha,\beta))&i<m,\\
\underline d_{m,i-m}(\alpha+\beta)&i\ge m.
\end{cases}
$$

Prune only when this bound exceeds D'. An empty result is then certified.
For $i\ge m$, the left input is zero and recursion continues on the right
with summed costs. For $i<m$, Gen streams valid left pairs under minimum costs.
Count aggregates their right extensions using the next identity.

Gen uses the same decomposition but emits actual input/codeword pairs.
It solves each feasible affine family, transforms its kernel basis once,
and traverses the basis with Gray updates. **A multiplicity cannot replace
these pairs inside Gen**, because a parent's equations depend on the
particular left input and output. No list-size truncation is used.

## Weighted affine closure

Right frozen equations are $H_Aa^\mathsf T+H_Bb^\mathsf T=0$. Set
$R=H_BG_m^\mathsf T$ and $c_a=H_Aa^\mathsf T$. For fixed $(a,s)$,

$$
\ell=\sum_{j:s_j=1}\min(\alpha_j,\beta_j),\qquad
\eta_j=\mathbf1\{s_j=1,\alpha_j>\beta_j\},
$$

$$
\delta_j=
\begin{cases}\alpha_j+\beta_j&s_j=0\\
|\alpha_j-\beta_j|&s_j=1,\end{cases}
\qquad
w_q(s+t,t)=\ell+\sum_j\delta_j(t_j\oplus\eta_j).
$$

Let F be the zero-delta coordinates and J its complement.
Write $t=\eta+e_J+z_F$. Eliminate columns in F first:

$$
R_Fz^\mathsf T+R_Je^\mathsf T=c_a+R\eta^\mathsf T,\quad
\rho=\operatorname{rank}R_F,\quad Be^\mathsf T=\bar c_a.
$$

Here the remaining independent equations define the quotient syndrome
system; $\kappa=\operatorname{rank}B$. Each feasible positive-cost deviation
has exactly $2^{|F|-\rho}$ zero-cost completions. Hence

$$
W_{a,s}^{\le D'}=
2^{|F|-\rho}X^\ell
\sum_{\substack{Be^\mathsf T=\bar c_a\\
\sum_{j\in J}\delta_je_j\le E}}
X^{\sum_{j\in J}\delta_je_j},\qquad E=D'-\ell.
$$

The implementation never assumes that infeasibility of e=0 excludes
nonzero deviations. Bounded enumeration uses sorted positive costs and
suffix-syndrome span pruning. Alternatively, sparse syndrome DP uses

$$
F_{r+1}(s,w)=F_r(s,w)+
\mathbf1\{w\ge\delta_r\}F_r(s\oplus B_r,w-\delta_r),
\quad F_0(0,0)=1.
$$

Only weights through E are stored. The auto policy compares the upper
bounds below; enumerate and dp force either exact method.

## WEF terminal recursion

If no local frozen index exceeds i, all remaining inputs are unrestricted.
Count returns the truncated polar coset WEF of prefix $0^i1$.
This terminal shortcut is the coset WEF component of the paper, not a
replacement by brute-force codeword enumeration.

For a prefix p of even length, let e and o be its even and odd subsequences.
Obtain pairs $f=(f_0,f_1)$ at costs $q_{\rm even}$ and prefix $e+o$,
and $g=(g_0,g_1)$ at costs $q_{\rm odd}$ and prefix o. Then

$$
\operatorname{Pair}(q,p)=
(f_0g_0+f_1g_1,\ f_0g_1+f_1g_0).
$$

For odd p, remove its final bit b before obtaining f,g and return
$(f_bg_0,f_{1-b}g_1)$. At a leaf the pair is $(1,X^{q_0})$.
Each product retains only terms whose combined degree is at most D'.
Even/odd costs here are deliberate: they follow the natural-order transform
in this WEF identity; the Count/Gen Plotkin split instead uses contiguous halves.

This weighted form implements the established polar coset pair recurrence.
The paper builds on Yao, Fazeli, and Vardy,
[A Deterministic Algorithm for Computing the Weight Distribution of Polar Code](https://doi.org/10.1109/TIT.2023.3243191),
IEEE Transactions on Information Theory, 2024.

## Paper-to-code map

Paths in the Python column are under python/polar_spectrum.
MATLAB functions are under matlab/+polar_spectrum; C++ names are in cpp/src/kernel.cpp.

| Algorithm step / paper labels | Python | MATLAB | C++ |
|---|---|---|---|
| Line 1; shortchecks, ratecost | algebra.prepare | algorithm1 preprocessing | prepare, run |
| Line 2; first-one sum | algorithm1 | algorithm1 outer loop | run |
| Line 4; universalbound | lower_bound | lower | lower |
| Line 5; wef, truncate | Solver.wef_pair | wef, multiply | Solver.wef, multiply |
| Lines 6–9; repeat, left | Solver.count, generate | count, generate | Solver.count, generate |
| Line 10; costs, quotient | Family constructor | family | Family constructor |
| Line 11; closure, syndromedp | Family.histogram | histogram | Family.histogram |
| Gen; system, extensions | Family.vectors, kernel_directions | expand, emitfamily, gray | Family.anchor, directions; generate |
| Deviations; residual bound | Family.patterns | patterns, deviations | Family.patterns |

The noiseless metric interpretation motivates the threshold bound. A channel
simulator and floating-point SCL metric are intentionally unnecessary in
Algorithm 1 itself.

## Complexity and implementation scope

For a family with p positive-cost positions and $E\ge0$, define

$$
Q=\sum_{h=0}^{\min(p,\lfloor E/\delta_{\min}\rfloor)}\binom ph,\qquad
C=\min\{m^2Q,(p+1)(E+1)2^\kappa\}.
$$

Set Q=1 for p=0. The auto policy selects the smaller bound.
Let $J_\gamma\le Q_\gamma$ be the feasible deviation count and
$Z_\gamma=2^{|F_\gamma|-\rho_\gamma}J_\gamma$.
With visited recursion nodes V, WEF terminals W, generation families Gg,
and counting families Gc, the paper's arithmetic-operation bound is

$$
T=O\left(N^3+D'+1+\sum_{v\in V}M_v+\sum_{\omega\in W}B_\omega
+\sum_{\gamma\in Gg}(m_\gamma^3+m_\gamma^2Q_\gamma+m_\gamma Z_\gamma)
+\sum_{\gamma\in Gc}(m_\gamma^3+C_\gamma)\right).
$$

Direct truncated WEF recursion has
$B_\omega=O(M_\omega(D'+1)^2)$ for general costs and
$O(M_\omega(D'+1))$ for unit costs, using the available degrees at each level.
With streamed Gen and two DP layers, the algorithmic storage bound is

$$
O\left(N^2+(D'+1)\log(N+1)
+\max_{\gamma\in G_{\rm DP}}(E_\gamma+1)2^{\kappa_\gamma}\right).
$$

These bounds count abstract arithmetic and indexed state operations.
Actual prototypes add dictionary/tree lookup, object allocation, and exact
integer costs. Coefficients may require O(N) bits; MATLAB uses base-10^7
limbs and C++ base-10^9 limbs. C++ ordered maps have logarithmic lookup
overhead; fixed 256-bit vectors also cost more than a single scalar operation.
The rows cache stores only structural interval checks, not an unbounded
search cache.

The $2^{|F|-\rho}$ reduction applies to **counting families**. It does not
remove the $Z_\gamma$ emission cost inside Gen, nor guarantee polynomial
running time for arbitrary D or profiles. max_work bounds counted search,
DP, and polynomial operations; it is a fail-fast guard, not a timeout or
a bound on preprocessing time or total memory.
