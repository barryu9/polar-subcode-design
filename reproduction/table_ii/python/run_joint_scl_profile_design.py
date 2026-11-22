"""Reliability, exact spectrum, and sampled prefix-risk feedback in one loop.

The Table II drivers call these routines on newly constructed profiles.
A proposed frozen expression is evaluated from the fixed design noise
stream. A complete distance check accepts or rejects the resulting code.
"""

import argparse
from collections import Counter, deque
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '1')

os.environ.setdefault('MKL_NUM_THREADS', '1')

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from cold_feedback_support import FeedbackTrial, ProvedSuffixInfeasible, trial_options
from design_scl_protected_prefix import design_prefix, protected_boundary
from extended_exact_spectrum import measure as full_measure, WorkLimitError
from polar_code.pc_funcs import pc_init
from polar_code.SCL_full_decoder import decode_full_path_batch
from rank_ordered_profile_design import solve_domain
from refine_protected_core_spectrum import measure, distance_pass
from refine_recursive_suffix_spectrum import suffix_design
from run_bounded_rm_ga_support import Trial
from run_diverse_profile_suite import read, write
from run_n64_spectral_profile_feedback import candidate_order, signature
from run_reference_domain_sweep import domain
from run_scaled_spectral_profile_feedback import directory
from select_designs_scl import construction
from structure_feedback_profile_design import protected_trial, final_check

ROOT = Path(__file__).resolve().parents[1]


class JointTrial(FeedbackTrial):
    """Permit a joint length-128 model instead of an eight-row dispatch guard."""

    def __init__(self, args):
        super().__init__(args)
        self.attempted = set()

    def dense(self, n, profile, distance, folder):
        key = (n, tuple(profile), distance)
        if n > 128 or key in self.attempted:
            return None
        self.attempted.add(key)
        _, edges = domain(n, profile, distance, 'initial')
        if not edges:
            return None
        cfg = SimpleNamespace(work=self.args.work, support_seconds=self.remaining(),
                              backend='z3', deadline=self.deadline)
        try:
            result = solve_domain(n, profile, distance, edges, folder, cfg,
                                  lambda *a, **k: None)
        except WorkLimitError as error:
            self.record(folder.name, dict(status='oracle_work_limit', reason=str(error)))
            return None
        self.record(folder.name, dict(status=result['status'], coefficients=len(edges),
            scope='Complete local retained-core domain, not all causal expressions'))
        if result['status'] == 'sat':
            return result['columns']
        if result['status'] == 'unsat' and self.remaining() > 0:
            return self.locked_search(n, profile, [0]*n, edges, distance,
                                      folder/'proved_lock_feedback')
        return None

    def build(self, n, profile, distance, folder):
        if not profile:
            return [0]*n
        folder.mkdir(parents=True, exist_ok=True)
        if n <= 128 and min(1 << i.bit_count() for i in profile) < distance:
            child = self.dense(n, profile, distance, folder/'joint_initial')
            if child is not None:
                return child
        # The lazy fallback uses proof-scoped locks in the full causal system.
        return Trial.build(self, n, profile, distance, folder)

def worker(args):
    data = read(args.worker_input)
    n, p, target = data['N'], data['profile'], data['target']
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        if args.worker == 'count':
            if n == 256 and target == 24:
                from count_constructed_a24 import count_code
                spectrum = count_code(tuple(p), data['columns'], target, args.output, args.work)
                result = dict(exact=True, spectrum=spectrum, method='disjoint Plotkin cosets')
            else:
                result = full_measure(n, p, data['columns'], target, args.work, policy='auto')
            result['status'] = 'complete'
        else:
            cfg = SimpleNamespace(work=args.work, verification_work=args.verify_work,
                support_seconds=args.solve_seconds, outer_seconds=args.solve_seconds,
                nodes=200000, direct_rows=8, radius_limit=3, pattern_outer=True,
                joint_coefficients=256, backend='z3', retry_incomplete=True)
            if args.worker == 'protected':
                result = protected_trial(n, tuple(p), target, args.cache, cfg,
                    lambda kind, **values: print(json.dumps(dict(kind=kind, **values)), flush=True))
            else:
                options = trial_options(args.output/'joint', args.solve_seconds,
                                        args.trial_seconds-10, args.work)
                options.expansions, options.release_batch = 32, 8
                options.output.mkdir(parents=True, exist_ok=True)
                trial = JointTrial(options)
                columns = trial.build(n, tuple(p), target, options.output/'recursive')
                if columns is None:
                    result = dict(status='retained', reason='bounded_joint_completion', stages=trial.stages)
                else:
                    proof = final_check(n, p, columns, target, args.verify_work, True)
                    result = dict(status='verified', columns=columns, verification=proof, stages=trial.stages)
    except ProvedSuffixInfeasible as error:
        result = dict(status='retained', reason='proved_necessary_suffix_obstruction',
                      suffix_certificate=error.certificate, full_profile_excluded=False)
    except (WorkLimitError, ValueError) as error:
        result = dict(status='unresolved', reason=str(error), exact=False,
                      full_profile_excluded=False)
    result['seconds'] = time.perf_counter()-start
    write(args.output/'worker_result.json', result)


def prefix_assessment(code, llr):
    n, k = code['N'], code['K']
    params = pc_init(n, k, 0, np.asarray(((1, 0), (1, 1)), dtype=int))
    actions, supports = construction(n, code['profile'], code['columns'])
    losses, failures = [], []
    for offset in range(0, len(llr), 128):
        _, loss, _, decoded = decode_full_path_batch(params, actions, supports, llr[offset:offset+128],
            32, early_stop=False, terminal_margin=0., max_support_size=None,
            backend='cpp', return_decoded_u=True)
        losses.extend(np.asarray(loss).astype(int).tolist())
        failures.extend(np.any(np.asarray(decoded) != 0, axis=1).tolist())
    losses = np.asarray(losses)
    hist = Counter(map(int, losses[losses >= 0]))
    return dict(frames=len(llr), pruning_errors=int(sum(hist.values())), errors=int(sum(failures)),
        terminal_errors=int(sum(failures))-int(sum(hist.values())),
        first_loss_histogram={str(i): v for i, v in sorted(hist.items())},
        sampled=True, FER_guarantee=False)

def prefix_proposals(parents, costs, ceiling, seen, floor):
    values = []
    for parent in parents:
        info = set(parent['profile'])
        hist = parent['prefix_feedback']['first_loss_histogram']
        for old in sorted(info):
            # Weak information positions near observed loss events receive priority.
            pressure = sum(v for key, v in hist.items() if old <= int(key) < old+len(costs)//4)
            if not pressure:
                continue
            for new in range(len(costs)):
                if new in info or 1 << new.bit_count() < floor or costs[new] >= costs[old]:
                    continue
                profile = sorted(info-{old}|{new})
                cost = sum(costs[i] for i in profile)
                if tuple(profile) in seen or cost > ceiling:
                    continue
                values.append(dict(profile=profile, removed=old, added=new, GA_cost=cost,
                    parent=parent['name'], parent_depth=parent['depth'], mode='prefix_pressure',
                    observed_loss_pressure=pressure, source_A_target=0))
    values.sort(key=lambda r: (-r['observed_loss_pressure'], r['GA_cost'],
                               r['removed'], r['added'], r['parent']))
    emitted = set()
    for value in values:
        key = tuple(value['profile'])
        if key not in emitted:
            emitted.add(key)
            yield value

def mixed_proposals(parents, costs, ceiling, seen, floor, target):
    base = iter(candidate_order(parents, costs, ceiling, seen, floor, target))
    prefix = iter(prefix_proposals(parents, costs, ceiling, seen, floor))
    active, emitted = [base, prefix], set()
    while active:
        for iterator in list(active):
            try:
                value = next(iterator)
            except StopIteration:
                active.remove(iterator)
                continue
            key = tuple(value['profile'])
            if key not in emitted:
                emitted.add(key)
                yield value

def representatives(archive, count, n, target):
    """Keep empirical risk and spectral endpoints plus structural diversity."""
    if not archive:
        return []
    endpoints = [min(archive, key=lambda e: (e[key], e['GA_cost'], e['name']))
                 for key in ('GA_cost', 'A_target', 'pruning_errors')]
    chosen, names = [], set()
    for entry in endpoints:
        if entry['name'] not in names:
            chosen.append(entry)
            names.add(entry['name'])
    used = {signature(e['profile'], n, target) for e in chosen}
    for entry in sorted(archive, key=lambda e: (e['pruning_errors'], e['GA_cost'], e['A_target'])):
        if len(chosen) >= count:
            break
        label = signature(entry['profile'], n, target)
        if label not in used:
            chosen.append(entry)
            names.add(entry['name'])
            used.add(label)
    for entry in sorted(archive, key=lambda e: (e['pruning_errors'], e['GA_cost'], e['A_target'])):
        if len(chosen) >= count:
            break
        if entry['name'] not in names:
            chosen.append(entry)
            names.add(entry['name'])
    return chosen[:count]

def support_front(variants, count=3):
    """An Admin improvement alone cannot discard a lower-prefix-risk support."""
    front = [v for v in variants if not any(
        q['A_target'] <= v['A_target'] and q['pruning_errors'] <= v['pruning_errors']
        and (q['A_target'] < v['A_target'] or q['pruning_errors'] < v['pruning_errors'])
        for q in variants)]
    if len(front) <= count:
        return front
    endpoints = [min(front, key=lambda v: (v['A_target'], v['pruning_errors'])),
                 min(front, key=lambda v: (v['pruning_errors'], v['A_target']))]
    chosen = {v['name']: v for v in endpoints}
    for v in sorted(front, key=lambda v: (v['prefix_feedback']['errors'], v['pruning_errors'])):
        if len(chosen) >= count:
            break
        chosen.setdefault(v['name'], v)
    return list(chosen.values())

def refine_support(base, case, args, llr, export, folder):
    n, p, target, length = case['N'], base['profile'], case['target'], case['length']
    start, deadline = time.perf_counter(), time.perf_counter()+case['refine_seconds']
    variants, trace = [base], []
    boundary = protected_boundary(p, target, n)
    early = [i for i in p if i < boundary]
    if early and any(j not in p and any(i < j for i in early) for j in range(boundary)):
        cols, proof = design_prefix(n, p, base['columns'], target, llr[:512], 32,
                                    row_floor=1, max_flips=8)
        if cols != base['columns']:
            feedback = dict(count=base['A_target'], source_A_target=base['source_A_target'],
                            length=length, local_target=base['feedback_target'])
            entry = export(p, cols, feedback, dict(base['proposal'], support_mode='neutral_prefix'),
                           dict(spectrum_invariance=proof['spectrum_invariance']), base['depth'])
            variants.append(entry)
            write(folder/'neutral_prefix.json', proof)
    offset, factor = n-length, n//length
    edges = [(j, i) for j in range(offset, n) if j not in p
             and 1 << (j-offset).bit_count() >= base['dp']*length//n
             for i in p if offset <= i < j]
    spectral = sorted(edges, key=lambda e: (-base['source_A_target'].get(str(e[1]), 0),
                                           bool(base['columns'][e[0]]), e))
    losses = base['prefix_feedback']['first_loss_histogram']
    prefix = sorted(edges, key=lambda e: (-sum(v for key, v in losses.items() if e[1] <= int(key)),
                                         abs(e[0]-int(max(losses, key=losses.get))) if losses else e[0], e))
    queues, tried, evaluations = [deque(spectral), deque(prefix)], set(), 0
    for attempt in range(args.refine_trials):
        if time.perf_counter() >= deadline or evaluations >= 12:
            break
        mode = attempt % 2
        parent = min(variants, key=lambda v: (v['A_target'], v['pruning_errors']) if mode == 0
                     else (v['pruning_errors'], v['A_target']))
        if not queues[mode]:
            continue
        j, i = queues[mode].popleft()
        cols = list(parent['columns'])
        cols[j] ^= 1 << i
        if tuple(cols) in tried:
            continue
        tried.add(tuple(cols))
        query_started = time.perf_counter()
        event = dict(edge=[j, i], mode=('spectral' if mode == 0 else 'prefix'),
                     parent=parent['name'], accepted=False)
        try:
            lp, lc = suffix_design(n, p, cols, length)
            counted = measure(length, lp, lc, target//factor, args.work)
            count = counted['spectrum'].get(target//factor, 0)
            event.update(status='exact_suffix', count=count)
            if not distance_pass(counted['spectrum'], target//factor):
                event['status'] = 'distance_rejection'
            elif mode == 0 and count >= parent['A_target']:
                event['status'] = 'no_spectral_improvement'
            else:
                proof = counted if n == length else final_check(n, p, cols, target, args.verify_work, True)
                # Only profile exchanges need per-source codeword multiplicities.
                # The fixed-profile decisions use the base histogram and this
                # already complete count, not a new catalogue for each variant.
                if getattr(args, 'catalogue_variants', True):
                    feedback = directory(n, p, cols, target, length, args.work)
                else:
                    feedback = dict(count=count, source_A_target=None, length=length,
                        local_target=target//factor, exact=True, full_layer=length==n,
                        oracle=counted, source_catalogue_computed=False)
                entry = export(p, cols, feedback,
                    dict(base['proposal'], support_mode=event['mode'], edge=[j, i]), proof, base['depth'])
                evaluations += 1
                event.update(pruning_errors=entry['pruning_errors'], name=entry['name'],
                             full_distance_verified=True)
                new_front = support_front(variants+[entry])
                event['accepted'] = entry in new_front
                # The original construction remains stored even if empirically dominated.
                variants = [base]+[v for v in new_front if v['name'] != base['name']]
        except WorkLimitError:
            event['status'] = 'oracle_work_limit'
        except AssertionError as error:
            if 'distance' not in str(error).lower():
                raise
            event.update(status='full_distance_rejection', reason=str(error))
        event['query_seconds'] = time.perf_counter()-query_started
        trace.append(event)
        write(folder/'refinement_trace.json', trace)
    result = support_front(variants)
    if base not in result:
        result.append(base)
    write(folder/'refinement_summary.json', dict(seconds=time.perf_counter()-start,
        attempts=len(trace), prefix_evaluations=evaluations, boundary=boundary,
        kept=[v['name'] for v in result], global_support_optimality=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete a coefficient or counting task for the current same-run profile.")
    parser.add_argument("--worker", choices=("joint", "protected", "count"), required=True)
    parser.add_argument("--worker-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--work", type=int, default=100000000)
    parser.add_argument("--verify-work", type=int, default=300000000)
    parser.add_argument("--solve-seconds", type=float, default=60)
    parser.add_argument("--trial-seconds", type=float, default=240)
    worker(parser.parse_args())
