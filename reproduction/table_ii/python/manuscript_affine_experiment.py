"""Experimental N<=64 family enumeration and exact single-row optimization.

The finite query rule uses repeated-coordinate conditional weight bounds.
It is not the exhaustive syndrome-query rule used in small-code verification.
"""

import time
from verify_manuscript_affine_enumeration import solve, combine, parity, project

def rejection_equations(records, prior, row, cutoff):
    equations = []
    for u, basis, weight in records:
        if 0 < weight <= cutoff:
            equations.append((project(u,prior), 1 ^ ((u>>row)&1)))
            equations.extend((project(b,prior), (b>>row)&1) for b in basis)
    return equations

def optimize(records, prior, row, cutoff, rng=None, max_rank=22):
    """An FWHT evaluates every image residual's exact weight.

    With no RNG, use the first optimal image and a zero kernel lift.
    """
    import numpy as np
    start = time.perf_counter()
    equations = rejection_equations(records,prior,row,cutoff)
    complete = solve(equations,len(prior))
    stats = dict(equations=len(equations), word_equations=sum(1<<len(b) for _,b,w in records if 0<w<=cutoff),
                 completion=complete is not None, completion_seconds=time.perf_counter()-start)
    if complete is not None:
        origin,basis = complete
        a = origin if rng is None else origin ^ combine(basis,rng.getrandbits(len(basis)))
        return a,0,dict(stats, design_rank=len(prior)-len(basis), residual_patterns=0,
                        optimization_seconds=time.perf_counter()-start)
    protect = solve(rejection_equations(records,prior,row,cutoff-1),len(prior))
    if protect is None:
        raise RuntimeError('Reference violates lower-layer protection')
    a0,allowed = protect
    coeffs,signs = [],[]
    for u,basis,w in records:
        if w != cutoff:
            continue
        word = u
        for z in range(1<<len(basis)):
            if z:
                word ^= basis[(z&-z).bit_length()-1]
            q = project(word,prior)
            coeffs.append(sum(parity(q&b)<<k for k,b in enumerate(allowed)))
            signs.append(1 if (((word>>row)&1)^parity(q&a0))==0 else -1)
    from verify_causal_flag_theory import rref
    rows = rref(coeffs)
    pivots = [(r&-r).bit_length()-1 for r in rows]
    rank = len(pivots)
    if rank > max_rank:
        raise RuntimeError('residual_rank_limit')
    histogram = np.zeros(1<<rank,dtype=np.int64)
    for c,sign in zip(coeffs,signs):
        histogram[project(c,pivots)] += sign
    step=1
    while step < len(histogram):
        blocks=histogram.reshape(-1,2*step)
        left=blocks[:,:step].copy();right=blocks[:,step:].copy()
        blocks[:,:step]=left+right;blocks[:,step:]=left-right
        step*=2
    minimum=int(histogram.min())
    # Tie and kernel choices do not change the exact conditional objective.
    best=np.flatnonzero(histogram==minimum)
    y=int(best[0] if rng is None else best[rng.randrange(len(best))])
    z=sum(((y>>k)&1)<<p for k,p in enumerate(pivots))
    null=solve([(c,0) for c in coeffs],len(allowed))[1]
    if rng is not None:
        z ^= combine(null,rng.getrandbits(len(null)))
    a=a0^combine(allowed,z)
    count=(len(coeffs)+minimum)//2
    return a,count,dict(stats,design_rank=rank,residual_patterns=1<<rank,
                        optimization_seconds=time.perf_counter()-start)

def code_from_columns(n,profile,columns):
    supports=[[i for i in profile if (columns[j]>>i)&1] if j not in profile else [] for j in range(n)]
    return dict(actions=[0 if j in profile else 2 if supports[j] else 1 for j in range(n)],supports=supports)
