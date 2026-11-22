"""Exact lazy cross completion after a profile exchange, with a fixed suffix.

Only the cross coefficients of the newly selected information coordinates are
variables. A complete critical-left catalogue and exact quotient-ball queries
separate feasible assignments. Counterexamples exclude a whole bad syndrome
fiber, rather than one assignment. The affine origin is fixed and reproducible.
"""

from functools import reduce
import hashlib
import json
from operator import xor
import time
import z3
from exact_relative_spectrum import complete_catalogue, count_recursive, subset_counts
from feasible_design_bank import validate_design
from plotkin_relative_structure import RelativePlotkinSpectrum
from polar_code.rp_funcs import rp_GA
from refine_recursive_suffix_spectrum import suffix_design
from run_diverse_profile_suite import identities, read, write
from run_n64_scl_comparison import normalize_json
from select_designs_scl import noiseless_check
from verify_causal_flag_theory import polar_transform

class CriticalFiber:
    def __init__(self, model, u, weight, cutoff):
        self.u, self.weight, self.basis = u, weight, {}
        self.radius = (cutoff-weight)//2
        x = polar_transform(u, model.half)
        for i in range(model.half):
            if not x >> i & 1:
                continue
            value = model.check_columns[i]
            while value:
                pivot = value.bit_length()-1
                if pivot in self.basis:
                    value ^= self.basis[pivot]
                else:
                    self.basis[pivot] = value
                    break
        self.pivots = sorted(self.basis, reverse=True)
        self.kernel = weight-len(self.basis)
        self.labels = [self.project(model.check_columns[i]) for i in range(model.half)
                       if not x >> i & 1]
        self.base = 0
        for q, j in enumerate(i for i in range(model.half) if i not in model.right_profile):
            if (model.columns[model.half+j] & u).bit_count() & 1:
                self.base ^= 1 << q

    def project(self, value):
        for pivot in self.pivots:
            if value >> pivot & 1:
                value ^= self.basis[pivot]
        return value

    def project_bv(self, value, width):
        for pivot in self.pivots:
            value = value ^ z3.If(z3.Extract(pivot, pivot, value) == z3.BitVecVal(1, 1),
                                 z3.BitVecVal(self.basis[pivot], width), z3.BitVecVal(0, width))
        return value

    def query(self, syndrome):
        projected = self.project(syndrome)
        counts = subset_counts(self.labels, projected, self.radius)
        return projected, {self.weight+2*i: int(count) << self.kernel
                           for i, count in enumerate(counts) if count}

def solve_new_cross(model, words, sources, cutoff, output, seconds=120, iterations=256):
    r = model.redundancy
    rows = tuple(i for i in range(model.half) if i not in model.right_profile)
    # Canonical frozen equations have unit check syndromes at frozen polar rows.
    for q, row in enumerate(rows):
        x, syndrome = polar_transform(1 << row, model.half), 0
        while x:
            bit = x & -x
            syndrome ^= model.check_columns[bit.bit_length()-1]
            x ^= bit
        if syndrome != 1 << q:
            raise AssertionError('Unexpected right-check coordinate convention')
    variables = {i: z3.BitVec(f'new_cross_delta_{i}', r) for i in sources}
    origins = {i: int.from_bytes(hashlib.sha256(f'fixed-cross-origin-v1:{i}:{r}'.encode()).digest(),
                                 'little') & ((1 << r)-1) for i in sources}
    solver = z3.SolverFor('QF_BV')
    solver.set(random_seed=0, threads=1)
    fibers = []
    for u, weight in words:
        item = CriticalFiber(model, u, weight, cutoff)
        selected = tuple(i for i in sources if u >> i & 1)
        if not selected:
            raise ValueError('Critical input is not controlled by a newly selected information coordinate')
        origin = item.base
        for i in selected:
            origin ^= origins[i]
        expression = z3.BitVecVal(origin, r) ^ reduce(xor, (variables[i] for i in selected),
                                                    z3.BitVecVal(0, r))
        item.selected = selected
        item.expression = item.project_bv(expression, r)
        fibers.append(item)
    started = time.perf_counter()
    trace, certificates, calls = [], [], 0
    cached = {}
    with (output/'solver_events.jsonl').open('w', encoding='utf-8') as stream:
        for iteration in range(iterations):
            remaining = seconds-(time.perf_counter()-started)
            if remaining <= 0:
                break
            solver.set(timeout=max(1, int(remaining*1000)))
            answer = solver.check()
            if answer != z3.sat:
                result = dict(status=str(answer), reason=solver.reason_unknown() if answer == z3.unknown else None,
                              iterations=iteration, seconds=time.perf_counter()-started)
                write(output/'solver_result.json', result)
                return result
            assignment = solver.model()
            values = {i: origins[i] ^ assignment.eval(v, model_completion=True).as_long()
                      for i, v in variables.items()}
            bad = []
            checked = 0
            for index, item in enumerate(fibers):
                syndrome = item.base
                for i in item.selected:
                    syndrome ^= values[i]
                projected = item.project(syndrome)
                key = index, projected
                if key not in cached:
                    cached[key] = item.query(syndrome)[1]
                    calls += 1
                spectrum = cached[key]
                checked += 1
                if spectrum:
                    bad.append((index, projected, spectrum))
                    if len(bad) == 16:
                        break
            event = dict(iteration=iteration, checked_fibers=checked, bad_fibers=len(bad),
                         exact_ball_queries=calls, clauses=len(certificates),
                         seconds=time.perf_counter()-started)
            trace.append(event)
            stream.write(json.dumps(event)+'\n')
            stream.flush()
            print(json.dumps(dict(stage='cross_completion', **event)), flush=True)
            if not bad:
                result = dict(status='sat', values={str(i):v for i,v in values.items()},
                              all_critical_fibers_checked=checked,
                              critical_catalogue_complete=True,
                              variables=r*len(sources), frozen_rows=rows,
                              iterations=iteration+1, exact_ball_queries=calls,
                              seconds=time.perf_counter()-started, trace=trace,
                              scope='New-information cross coefficients; both diagonal constituents fixed',
                              global_optimality_claim=False)
                write(output/'solver_result.json', result)
                write(output/'counterexample_fibers.json', certificates)
                return result
            for index, projected, spectrum in bad:
                item = fibers[index]
                solver.add(item.expression != z3.BitVecVal(projected, r))
                certificates.append(dict(fiber=index, left_input=item.u,
                                         forbidden_projected_syndrome=projected,
                                         forbidden_spectrum=spectrum))
            write(output/'progress.json', event)
    result = dict(status='unknown', reason='completion_budget', seconds=time.perf_counter()-started,
                  iterations=len(trace), exact_ball_queries=calls,
                  global_infeasibility_claim=False)
    write(output/'solver_result.json', result)
    write(output/'counterexample_fibers.json', certificates)
    return result

def run(args):
    raw = read(args.source)
    code = normalize_json(raw)
    n, p, c = code['N'], tuple(code['profile']), list(code['columns'])
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cutoff = args.target//2-1
    if any(i < n//2 or i >= 3*n//4 or i not in p for i in args.new_information):
        raise ValueError('New information must be in the left projection of the last-half code')
    if min((1 << i.bit_count() for i in p if i < n//2), default=n+1) < args.target:
        raise ValueError('Top-level left leading cosets do not protect the target distance')
    half_p, half_c = suffix_design(n, p, c, n//2)
    model = RelativePlotkinSpectrum(half_p, half_c)
    validate_design(n, p, c)
    write(args.output/'protocol.json', dict(source=str(args.source.resolve()),
          source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
          N=n, K=len(p), target=args.target, new_information=args.new_information,
          solver_seconds=args.seconds, max_work=args.work,
          right_suffix_fixed=True, reference_support_used=False,
          deterministic_affine_origin='SHA256 fixed-cross-origin-v1:{local_source}:{right_redundancy}',
          random_sampling=False, global_optimality_claim=False))
    words, statistics = complete_catalogue(n//4, model.left_profile, model.left_columns,
                                           cutoff, args.work)
    write(args.output/'critical_left.json', dict(words=words, complete=True, statistics=statistics))
    right = count_recursive(model.right_profile, model.right_columns, args.target//4,
                            args.output/'right_verification', args.work)
    if any(w and w < args.target//4 and a for w, a in right.items()):
        raise ValueError('Fixed right suffix violates its required distance')
    answer = solve_new_cross(model, words, tuple(i-n//2 for i in args.new_information),
                             cutoff, args.output, args.seconds)
    if answer['status'] != 'sat':
        write(args.output/'summary.json', dict(status=answer['status'], solver=answer,
              right_spectrum=right, seconds=time.perf_counter()-started,
              global_infeasibility_claim=False))
        return
    new_mask = sum(1 << i for i in args.new_information)
    for q, row in enumerate(answer['frozen_rows']):
        j = 3*n//4+row
        c[j] &= ~new_mask
        c[j] |= sum(((answer['values'][str(i-n//2)] >> q) & 1) << i
                    for i in args.new_information)
    validate_design(n, p, c)
    original_right_p, original_right_c = suffix_design(n, p, code['columns'], n//4)
    final_right_p, final_right_c = suffix_design(n, p, c, n//4)
    if (original_right_p, original_right_c) != (final_right_p, final_right_c):
        raise AssertionError('The shortened right suffix was modified')
    ph, sh = identities(n, p, c)
    ga = rp_GA(n, len(p), args.design_snr)
    import math
    proxy = -math.expm1(sum(math.log1p(-float(ga.pe[i])) for i in p))
    design = dict(N=n, K=len(p), profile=p, columns=c, name=ph[:12]+'_'+sh[:12],
                  profile_hash=ph, support_hash=sh, GA_proxy=proxy,
                  GA_design_EbN0_dB=args.design_snr, target=args.target,
                  distance_lower_bound=args.target, distance_proof='Complete critical-left fibre exclusion, fixed right distance, and top leading-coset floor',
                  reference_support_used=False, FER_verified=False, global_optimality_claim=False)
    write(args.output/'code.json', design)
    spectrum = count_recursive(p, c, args.target, args.output/'full_spectrum', args.work)
    if any(w and w < args.target and a for w, a in spectrum.items()):
        raise AssertionError('Complete spectrum disagrees with the cross-completion proof')
    noiseless_check(design, 32, 'cpp')
    design.update(dmin=args.target if spectrum.get(args.target) else None,
                  Admin=spectrum.get(args.target) or None, A_target=spectrum.get(args.target, 0),
                  complete_spectrum=spectrum, spectrum_exact=True, noiseless_L32_pass=True)
    write(args.output/'code.json', design)
    summary = dict(status='verified', name=design['name'], dmin=design['dmin'],
                   A_target=design['A_target'], GA_proxy=proxy,
                   critical_left_words=len(words), cross_coefficients=model.redundancy*len(args.new_information),
                   right_spectrum=right, spectrum=spectrum, right_suffix_preserved=True,
                   solver_seconds=answer['seconds'], seconds=time.perf_counter()-started,
                   code_path=str((args.output/'code.json').resolve()), FER_verified=False,
                   global_optimality_claim=False)
    write(args.output/'summary.json', summary)
    print(json.dumps(dict(stage='complete', **summary)), flush=True)
