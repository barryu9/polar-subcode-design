"""Exact cross-expression descent with shared Plotkin syndrome fibers.

For weight-t left codewords the contribution is a constant-weight fiber.
For lighter left codewords, small outside-error syndrome tables give both
the objective layer and every forbidden lower layer. No partial catalogue
or sampled codeword is used to score a candidate expression.
"""

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from exact_relative_spectrum import complete_catalogue, count_recursive
from plotkin_relative_structure import RelativePlotkinSpectrum
from polar_code.rp_funcs import rp_GA
from refine_recursive_suffix_joint_spectrum import protected_exchange
from refine_recursive_suffix_spectrum import suffix_design
from refine_suffix_layer_relative import one_bit_descent
from row_neighborhood import RowCatalogue
from run_diverse_profile_suite import identities, read, write
from run_n64_scl_comparison import normalize_json
from select_designs_scl import noiseless_check
from verify_causal_flag_theory import polar_transform

def pack(mask):
    return int.from_bytes(np.packbits(mask, bitorder='little').tobytes(), 'little')

class CrossFibers:
    def __init__(self, profile, columns, target, work):
        self.profile, self.columns = tuple(profile), list(columns)
        self.model = RelativePlotkinSpectrum(profile, columns)
        m = self.model
        if m.redundancy > 63:
            raise ValueError('Require redundancy <= 63')
        self.rows = tuple(m.half+j for j in range(m.half) if j not in m.right_profile)
        self.infos = tuple(m.left_profile)
        self.words, _ = complete_catalogue(m.half, m.left_profile, m.left_columns, target, work)
        if any(w < target-4 or (target-w) % 2 for _, w in self.words):
            raise ValueError('This implementation supports integral outside-error radii at most two')
        if any(w < target for _, w in self.words) and m.redundancy > 16:
            raise ValueError('Tabulated nonconstant fibers require redundancy <= 16')
        effects = []
        for j in self.rows:
            x = polar_transform(1 << (j-m.half), m.half)
            effects.append(self.syndrome(x))
        residues, syndromes, kernels, target_tables, low_tables = [], [], [], [], []
        for u, weight in self.words:
            x, basis = polar_transform(u, m.half), {}
            for i in range(m.half):
                if not x >> i & 1:
                    continue
                v = m.check_columns[i]
                while v:
                    pivot = v.bit_length()-1
                    if pivot in basis:
                        v ^= basis[pivot]
                    else:
                        basis[pivot] = v
                        break
            pivots = sorted(basis, reverse=True)

            def residue(value):
                for pivot in pivots:
                    if value >> pivot & 1:
                        value ^= basis[pivot]
                return value

            s = 0
            for j, effect in zip(self.rows, effects):
                if (columns[j] & u).bit_count() & 1:
                    s ^= effect
            residues.append([residue(effect) for effect in effects])
            syndromes.append(residue(s))
            kernel = weight-len(basis)
            kernels.append(kernel)
            if weight < target:
                radius = (target-weight)//2
                states = 1 << m.redundancy
                outside = np.asarray([residue(m.check_columns[i]) for i in range(m.half)
                                      if not x >> i & 1], dtype=np.int64)
                hist = [np.eye(1, states, dtype=np.int64)[0],
                        np.bincount(outside, minlength=states)]
                if radius == 2:
                    a, b = np.triu_indices(len(outside), 1)
                    hist.append(np.bincount(outside[a] ^ outside[b], minlength=states))
                target_tables.append(hist[radius] << kernel)
                low_tables.append(sum(hist[:radius]) << kernel)
        self.residues = np.asarray(residues, dtype=np.uint64).reshape(len(self.words), len(self.rows)).T
        self.syndromes = np.asarray(syndromes, dtype=np.uint64)
        self.kernels = np.asarray(kernels, dtype=np.int64)
        self.weights = np.left_shift(np.int64(1), self.kernels)
        self.main = np.asarray([w == target for _, w in self.words], dtype=bool)
        self.low_indices = np.flatnonzero(~self.main)
        self.table_target = np.asarray(target_tables, dtype=np.int64)
        self.table_low = np.asarray(low_tables, dtype=np.int64)
        self.incidence = [np.fromiter((bool(u >> i & 1) for u, _ in self.words),
                                     dtype=bool, count=len(self.words)) for i in self.infos]
        self.planes = [pack(mask) for mask in self.incidence]
        self.low_planes = [pack(mask[self.low_indices]) for mask in self.incidence]
        self.kernel_values = sorted(set(kernels))

    def syndrome(self, x):
        value = 0
        while x:
            bit = x & -x
            value ^= self.model.check_columns[bit.bit_length()-1]
            x ^= bit
        return value

    def contribution(self):
        value = int(self.weights[self.main & (self.syndromes == 0)].sum())
        if len(self.low_indices):
            value += int(self.table_target[np.arange(len(self.low_indices)),
                                         self.syndromes[self.low_indices]].sum())
        return value

    def below(self):
        if not len(self.low_indices):
            return 0
        return int(self.table_low[np.arange(len(self.low_indices)),
                                 self.syndromes[self.low_indices]].sum())

    def best_move(self, pairs=False):
        moves = [(1 << i, plane, self.low_planes[q], (q,))
                 for q, (i, plane) in enumerate(zip(self.infos, self.planes))]
        if pairs:
            moves += [((1 << self.infos[a]) | (1 << self.infos[b]),
                       self.planes[a] ^ self.planes[b], self.low_planes[a] ^ self.low_planes[b], (a, b))
                      for a, b in combinations(range(len(self.infos)), 2)]
        winner = None
        for q, row in enumerate(self.rows):
            v = self.residues[q]
            active = (v != 0) & self.main
            gains, losses = (self.syndromes == v) & active, (self.syndromes == 0) & active
            masks = [(1 << kernel, pack(gains & (self.kernels == kernel)),
                      pack(losses & (self.kernels == kernel))) for kernel in self.kernel_values]
            blocked, low_masks = 0, []
            if len(self.low_indices):
                ids = np.arange(len(self.low_indices))
                current = self.syndromes[self.low_indices]
                proposed = current ^ v[self.low_indices]
                blocked = pack(self.table_low[ids, proposed] != 0)
                difference = self.table_target[ids, proposed]-self.table_target[ids, current]
                low_masks = [(int(amount), pack(difference == amount))
                             for amount in np.unique(difference) if amount]
            for delta, plane, low_plane, indices in moves:
                if low_plane & blocked:
                    continue
                change = sum(weight*((gain & plane).bit_count()-(loss & plane).bit_count())
                             for weight, gain, loss in masks)
                change += sum(amount*(mask & low_plane).bit_count() for amount, mask in low_masks)
                if change < 0 and (winner is None or (change, row, delta) < winner[:3]):
                    winner = (change, row, delta, q, indices)
        return winner

    def apply(self, move):
        change, row, delta, q, indices = move
        selected = self.incidence[indices[0]].copy()
        for i in indices[1:]:
            selected ^= self.incidence[i]
        before = self.contribution()
        self.syndromes[selected] ^= self.residues[q, selected]
        self.columns[row] ^= delta
        if self.contribution() != before+change:
            raise AssertionError('Fiber update disagrees with the scored move')
        if self.below():
            raise AssertionError('A selected move introduced a lower-weight codeword')

def run(args):
    raw = read(args.source)
    code = normalize_json(raw)
    n, p, c = code['N'], tuple(code['profile']), list(code['columns'])
    ga = rp_GA(n, len(p), args.design_snr)
    costs = [-math.log1p(-float(v)) for v in ga.pe]
    exchange = None
    if args.remove:
        boundary = min((i for i in p if 1 << i.bit_count() < args.target), default=n)
        added = sorted((i for i in range(boundary) if i not in p and 1 << i.bit_count() >= args.target),
                       key=lambda i: (costs[i], i))[:len(args.remove)]
        p, c, exchange = protected_exchange(n, p, c, args.remove, added, args.target-1)
    if min((1 << i.bit_count() for i in p if i < n-args.suffix_length), default=n+1) < args.target:
        raise ValueError('The prefix causal-coset floor does not protect the full distance')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    projected_events = []
    if args.left_rows:
        length, start = args.suffix_length//2, n-args.suffix_length
        lp = tuple(i-start for i in p if start <= i < start+length)
        lc = [v >> start for v in c[start:start+length]]
        local_target = args.target//(n//args.suffix_length)
        if min((1 << i.bit_count() for i in lp), default=length+1) < local_target:
            raise ValueError('Projected refinement requires the local profile row floor to protect its distance')
        for row in args.left_rows:
            if row in lp:
                raise ValueError('Projected edited row is not frozen')
            catalogue = RowCatalogue(SimpleNamespace(n=length, profile=lp), lc, row,
                                     local_target, args.work)
            before, _ = catalogue.score(0)
            delta, after, _ = one_bit_descent(catalogue, 12)
            lc = catalogue.child(delta)
            c[start+row] ^= delta << start
            projected_events.append(dict(row=row, before=before, after=after, delta=delta))
            print(json.dumps(dict(stage='projected_row', **projected_events[-1])), flush=True)
            del catalogue
    profile_hash, _ = identities(n, p, c)
    write(args.output/'protocol.json', dict(source=str(args.source.resolve()), target=args.target,
          source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
          suffix_length=args.suffix_length, profile_exchange=exchange,
          steps=args.steps, pair_descent=args.pairs, work=args.work,
          projected_refinement=projected_events,
          profile_row_floor=min(1 << i.bit_count() for i in p),
          distance_argument='Prefix leading cosets have weight at least target. With prefix information zero, the suffix repeats and its complete lower spectrum is preserved.',
          random_sampling=False, reference_support_read=False, global_optimality_claim=False))
    initial = count_recursive(p, c, args.target, args.output/'initial', args.work)
    local_p, local_c = suffix_design(n, p, c, args.suffix_length)
    target = args.target//(n//args.suffix_length)
    fibers = CrossFibers(local_p, local_c, target, args.work)
    model = fibers.model
    right = count_recursive(model.right_profile, model.right_columns, target//2,
                            args.output/'right_check', args.work)
    if any(w and 2*w < target and a for w, a in right.items()):
        raise ValueError('The repeated right code violates the constituent target')
    floor = right.get(target//2, 0)
    local_initial = count_recursive(local_p, local_c, target, args.output/'initial', args.work)
    if floor+fibers.contribution() != local_initial.get(target, 0):
        raise AssertionError('Complete spectrum and exact cross-fiber model disagree')
    if fibers.below():
        raise ValueError('Initial cross fibers contain forbidden lower-weight codewords')
    events = []
    for step in range(args.steps):
        move = fibers.best_move(False)
        if move is None and args.pairs:
            move = fibers.best_move(True)
        if move is None:
            break
        fibers.apply(move)
        event = dict(step=step+1, local_row=move[1],
                     toggled_sources=[i for i in fibers.infos if move[2] >> i & 1],
                     suffix_A=floor+fibers.contribution(),
                     change=move[0], seconds=time.perf_counter()-started)
        events.append(event)
        if len(events) % 5 == 0 or step == 0:
            print(json.dumps(dict(stage='fiber_descent', **event)), flush=True)
        write(args.output/'progress.json', dict(profile_hash=profile_hash, floor=floor,
              current_suffix_A=floor+fibers.contribution(), events=events))
    start, prefix_mask = n-args.suffix_length, (1 << (n-args.suffix_length))-1
    for row in fibers.rows:
        c[start+row] = (c[start+row] & prefix_mask) | (fibers.columns[row] << start)
    final = count_recursive(p, c, args.target, args.output/'final', args.work)
    final_local = count_recursive(local_p, fibers.columns, target, args.output/'final', args.work)
    if final_local.get(target, 0) != floor+fibers.contribution():
        raise AssertionError('Final independent constituent count disagrees with fiber refinement')
    if any(w and w < args.target and a for w, a in final.items()):
        raise AssertionError('Full profile row-floor distance guarantee failed')
    noiseless_check(dict(N=n, K=len(p), profile=p, columns=c), 32, 'cpp')
    ph, sh = identities(n, p, c)
    candidate = dict(N=n, K=len(p), profile=p, columns=c, name=ph[:12]+'_'+sh[:12],
                     profile_hash=ph, support_hash=sh, target=args.target,
                     dmin=args.target if final.get(args.target) else None,
                     distance_lower_bound=args.target, Admin=final.get(args.target) or None,
                     A_target=final.get(args.target, 0), GA_proxy=-math.expm1(-sum(costs[i] for i in p)),
                     GA_design_EbN0_dB=args.design_snr, GA_proxy_definition='1 - product(1 - pe[i])',
                     complete_spectrum=final, spectrum_exact=True, profile_exchange=exchange,
                     noiseless_L32_pass=True, FER_verified=False, global_optimality_claim=False)
    write(args.output/'code.json', candidate)
    summary = dict(name=candidate['name'], initial_A_target=initial.get(args.target, 0),
                   final_A_target=final.get(args.target, 0), GA_proxy=candidate['GA_proxy'],
                   right_subcode_floor=floor, final_suffix_A=final_local.get(target, 0),
                   steps=len(events), seconds=time.perf_counter()-started,
                   projected_refinement=projected_events,
                   spectrum=final, profile_exchange=exchange, code_path=str((args.output/'code.json').resolve()),
                   global_optimality_claim=False, FER_verified=False)
    write(args.output/'events.json', events)
    write(args.output/'summary.json', summary)
    print(json.dumps(dict(stage='complete', **summary)), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=int, default=32)
    parser.add_argument('--suffix-length', type=int, default=256)
    parser.add_argument('--remove', type=int, nargs='*', default=[])
    parser.add_argument('--design-snr', type=float, default=2.25)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--pairs', action='store_true')
    parser.add_argument('--left-rows', type=int, nargs='*', default=[])
    parser.add_argument('--work', type=int, default=600000000)
    run(parser.parse_args())
