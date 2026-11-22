"""Bounded, reference-blind stratified profile proposals and exact support trials.

Layer cardinalities define optional profile families, not necessary conditions.
Scoped failures change proposal priority only. No algebraic domain is imported.
"""

import z3
from profile_linear_bounds import basic_bound, coset_capacities

def necessary_solver(n, k, target, floor):
    """A finite subset of table-free necessary checks, with no sufficiency claim."""
    s = z3.Solver()
    s.set(random_seed=0)
    v = [z3.Bool(f'information_{i}') for i in range(n)]
    s.add(z3.PbEq([(x, 1) for x in v], k))
    s.add(*[z3.Not(v[i]) for i in range(n) if (1 << i.bit_count()) < floor])
    bounds = []
    length = 1
    while length <= n:
        target_local = (target*length+n-1)//n
        local = v[n-length:]
        cap = basic_bound(length, target_local, False)
        s.add(z3.PbLe([(x, 1) for x in local], cap))
        even_cap = basic_bound(length, target_local+target_local % 2, False)
        s.add(z3.Implies(z3.Not(local[0]), z3.PbLe([(x, 1) for x in local], even_cap)))
        if target_local >= 2 and length > 1:
            s.add(z3.Implies(z3.PbEq([(x, 1) for x in local], length-1), z3.Not(local[0])))
        for i, x in enumerate(local):
            if (1 << i.bit_count()) < target_local:
                s.add(z3.Implies(x, z3.Or(*[z3.Not(y) for y in local[i+1:]])))
        if length <= 128:
            caps, complement = coset_capacities(length, target_local)
            for i in range(length-1):
                later = [(x, 1) for x in local[i+1:]]
                s.add(z3.Implies(local[i], z3.PbLe(later, caps[i])))
                s.add(z3.Implies(z3.And(local[i], local[-1]), z3.PbLe(later, complement[i])))
        bounds.append(dict(length=length, target=target_local, dimension_cap=cap,
                           even_dimension_cap=even_cap, coset_cap_checked=length <= 128))
        length *= 2
    return s, v, bounds

def membership_check(solver, variables, profile, milliseconds=3000):
    info = set(profile)
    solver.push()
    try:
        solver.set(timeout=milliseconds)
        solver.add(*[v if i in info else z3.Not(v) for i,v in enumerate(variables)])
        return str(solver.check())
    finally:
        solver.pop()
