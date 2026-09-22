# Construction conventions and APIs

## Polar transform and indices

All encoders use the row-vector convention $x=uF^{\otimes n}$ with N=2^n,
without a bit-reversal permutation. In Python, bit j of an integer represents
coordinate j; JSON and CSV fixtures use zero-based coordinate labels.
MATLAB profile.information, P, and S use one-based indices.
profile.information_zero_based retains the literal published fixture.

The polynomial variable X describes output Hamming weight, not input weight.

## GA at Eb/N0 = 2.0 dB

Construction uses the mother rate R=K/N=1/2 throughout, including examples
that later puncture or shorten the code. With unit-energy BPSK:

$$
\sigma^2=(2R\,10^{2/10})^{-1},\qquad
m_0=2/\sigma^2=4R\,10^{2/10}.
$$

Starting from m0, each polarization stage interleaves
$[\phi^{-1}(1-(1-\phi(m))^2),2m]$ for each preceding mean m.
The fixed Chung-type approximation is

$$
\phi(x)=
\begin{cases}
\min(1,\exp(-0.4527x^{0.86}+0.0218)),&0\le x\le10,\\
\sqrt{\pi/x}\exp(-x/4)(1-10/(7x)),&x>10.
\end{cases}
$$

The implementation evaluates log(phi) and uses 80 bisection steps on [0,m]
for the bad-channel update. Select the K largest means, breaking ties by
larger index, then sort the selected indices increasingly.
This precisely defines these GA fixtures; alternative GA approximations,
Es/N0 interpretations, or bit reversal may produce different profiles.

Run tools/freeze_profiles.py only to audit or regenerate fixtures.
It writes the Python/MATLAB literals, profiles/ga_2db.json (including means),
rate-matching JSON, and selector CSVs. Normal examples do not invoke it.

## PAC orientation

Polar and PAC share the same set of free **source** positions. PAC uses

$$
g=(1,0,1,1,0,1,1),\quad
g(D)=1+D^2+D^3+D^5+D^6,\quad
u_i=\sum_{\ell=0}^{6}g_\ell v_{i-\ell},\qquad v_j=0\ (j<0).
$$

The printed octal-133 convention names this agreed tap vector. It must not
be reinterpreted as the least-significant-bit-first binary expansion of the
integer 0o133; that would reverse the taps.
There is zero prehistory and no extra termination bits after the N source bits.
For Polar, g=(1).

To form causal checks, compute the first N coefficients h of 1/g in GF(2).
For each frozen source position i, impose
$\sum_{j=0}^{i}h_{i-j}u_j=0$. Thus PAC is handled through H by the same kernel.
The independent test includes direct causal convolution of all source words
of a small PAC code to check this convention.

## Rate-matching matrices

For disjoint P and S, let T be the remaining coordinates in increasing order.
$E_P,E_S,E_T$ are rows of the N-by-N identity matrix. They satisfy

$$
x_{\rm tx}=xE_T^\mathsf T,\qquad E_Sx^\mathsf T=0.
$$

Each CSV under profiles/matrices is a fully written binary E_P or E_S,
one selected coordinate per row, N columns, no header. An empty file denotes
the 0-by-N selector; N is provided in its filename and the JSON fixture.
Python selection_matrices and MATLAB rate_matching also return E_T.
The fixed coordinate arrays are the lossless sparse representation of these
same hard-coded matrices.

The profiles are **mother** codes. For example, shortening may reduce K,
and puncturing may reduce it further through a nontrivial projection kernel.
Return fields distinguish original_dimension, shortened_dimension,
transmitted_dimension, and kernel_dimension, with
kernel_dimension = shortened_dimension − transmitted_dimension.

## Python API

~~~python
from polar_spectrum import algorithm1, get_profile, checks_for_profile, get_rate_matching

cfg = get_profile(128, "pac")
H = checks_for_profile(cfg)  # tuple of integer masks representing H rows
matching = get_rate_matching(128, "mixed")
result = algorithm1(128, H, 8, **matching, backend="python",
                    policy="auto", max_work=10_000_000)
print(result["spectrum"])   # {integer weight: arbitrary-precision integer count}
~~~

For a custom code, supply any homogeneous binary H, even if not already
causal or independent. Encode a row as the sum of 1<<j over its nonzero
columns. Punctured and shortened arguments are output-coordinate lists.
Affine nonzero check right-hand sides are outside this API.

D is a nonnegative integer. D>|T| is equivalent to |T| and both requested
and evaluated cutoffs are reported. Python supports general power-of-two N;
the optional C++ backend is explicitly limited to N≤256.

backend selects python/cpp. policy selects auto/enumerate/dp for Count;
Gen always emits feasible actual pairs. Stats report mechanism-specific work
counts, not CPU cycles. max_work defaults to 10 million counted operations.

The examples run directly from source. For general imports outside the
repository, install with pip or add its python directory to PYTHONPATH.
An installed wheel contains the Python package only; build the C++ library
from the source repository and set POLAR_SPECTRUM_LIBRARY to use it.

## MATLAB API

~~~matlab
addpath('matlab');
cfg = polar_spectrum.profile(128,'pac');
H = polar_spectrum.checks_for_profile(cfg); % logical matrix
[P,S,ET,EP,ES] = polar_spectrum.rate_matching(128,'mixed');
r = polar_spectrum.algorithm1(H,8,P,S,'auto',1e7);
disp([string(r.weights(:)),string(r.counts(:))]);
~~~

H is an r-by-N binary matrix; use false(0,N) for no constraints.
The result has parallel weights and counts arrays; counts is a cell array
of exact decimal strings. Avoid str2double for counts beyond 2^53.
The default MATLAB work limit may be changed up to flintmax.

## C ABI and timing

See cpp/include/polar_spectrum.h. H is a contiguous row-major byte matrix,
P and S are binary masks, and policy is 0/1/2. The returned UTF-8 JSON must
be released by polar_spectrum_free; an error is represented explicitly.
No C++ exception escapes the ABI. JSON multiplicities are decimal strings.

elapsed_seconds measures Algorithm 1 normalization plus counting. It excludes
profile/H construction, native input marshalling, loading the shared library,
printing, and MATLAB/Python process startup. The Python native binding additionally
reports binding_elapsed_seconds covering loading, marshalling, kernel execution,
and result parsing. Native timing uses QueryPerformanceCounter on Windows.
