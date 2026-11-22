"""Test reference information sets with exact, lazy shared-support exclusions.

Only the pivot positions are retained from each reference precoder. Counterexample
inputs impose necessary rejection clauses on shared causal coefficients. SAT is
accepted only after complete bounded-weight enumeration. UNSAT of even a partial
clause set proves infeasibility of the selected domain. Limits never prove UNSAT.
"""

import json
import z3
from select_structured_domain import initial_structure

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def domain(n, profile, target, mode):
    structure = initial_structure(n, profile, target)
    initial = {(structure['rows'][q], i) for q, i in structure['coordinates']}
    if mode == 'initial':
        coordinates = initial
    elif mode == 'core_relaxed':
        coordinates = {(j, i) for j in structure['rows'] for i in profile if i < j}
    elif mode == 'full':
        coordinates = {(j, i) for j in range(n) if j not in profile
                       for i in profile if i < j}
    else:
        raise ValueError(mode)
    return structure, tuple(sorted(coordinates))

def clause_for_input(u, variables):
    by_row = {}
    for (j, i), variable in variables.items():
        if u >> i & 1:
            by_row.setdefault(j, []).append(variable)
    errors = []
    for j in sorted({j for j, _ in variables}):
        parity = z3.BoolVal(False)
        for variable in by_row.get(j, []):
            parity = z3.Xor(parity, variable)
        errors.append(parity != z3.BoolVal(bool(u >> j & 1)))
    return z3.simplify(z3.Or(errors))
