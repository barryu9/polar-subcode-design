"""Deterministic coefficient descent beyond a retained information core.

Every accepted edit has an exact truncated spectrum and preserves the requested
distance. A finite prefix or a work-limited query is not an optimality proof.
"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time
from external_low_weight_spectrum import spectrum, WorkLimitError, checks_from_construction, algorithm1
from run_diverse_profile_suite import read, write as atomic_write, identities
from run_n64_scl_comparison import normalize_json
from select_designs_scl import construction

def write(path, value):
    for attempt in range(8):
        try:
            atomic_write(path, value)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt+1))

def measure(n, profile, columns, target, work, backend='cpp', policy='auto'):
    actions, supports = construction(n, profile, columns)
    code = dict(actions=list(map(int, actions)),
                supports=[list(map(int, s)) for s in supports])
    if policy == 'auto':
        return spectrum(code, cutoff=target, max_work=work, backend=backend)
    checks, k = checks_from_construction(code['actions'], code['supports'])
    answer = algorithm1(n, checks, target, backend=backend, policy=policy, max_work=work)
    if (not answer['exact'] or answer['spectrum'].get(0) != 1
            or answer['transmitted_dimension'] != k or answer['kernel_dimension'] != 0):
        raise AssertionError('Complete spectrum dimension audit failed')
    return answer

def distance_pass(histogram, target):
    return not any(a for w, a in histogram.items() if 0 < w < target)

def coefficient_moves(edges, size, scope, target):
    if size == 1:
        return tuple((edge,) for edge in edges)
    moves = [pair for pair in combinations(edges, 2)
             if scope == 'all'
             or (scope in ('same_row', 'coupled') and pair[0][0] == pair[1][0])
             or (scope in ('same_source', 'coupled') and pair[0][1] == pair[1][1])]
    # First examine coupled edits involving the lowest-weight information rows.
    return tuple(sorted(moves, key=lambda pair: (
        not all(1 << i.bit_count() < target for j, i in pair),
        pair[0][1] != pair[1][1], pair)))

def apply_coefficient_move(columns, edges):
    if len(set(map(tuple, edges))) != len(edges):
        raise ValueError('A coefficient cannot occur twice in one joint edit')
    result = list(columns)
    for j, i in edges:
        if not 0 <= i < j < len(result):
            raise ValueError('Every coefficient edit must be causal')
        result[j] ^= 1 << i
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--trials', type=int, default=96)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--work', type=int, default=3000000)
    parser.add_argument('--policy', choices=('auto', 'enumerate', 'dp'), default='auto')
    parser.add_argument('--directions', choices=('core', 'floor', 'all'), default='core')
    parser.add_argument('--move-size', type=int, choices=(1, 2), default=1)
    parser.add_argument('--pair-scope', choices=('same_source', 'same_row', 'coupled', 'all'),
                        default='coupled')
    parser.add_argument('--unknown-streak', type=int, default=0,
                        help='Stop this local refinement after this many consecutive work-limited queries; zero disables it')
    args = parser.parse_args()
    if min(args.target, args.trials, args.seconds, args.work) <= 0 or args.unknown_streak < 0:
        parser.error('Positive target/budgets and a nonnegative unknown-streak limit are required')
    raw = read(args.source)
    design = normalize_json(raw)
    n, profile = design['N'], tuple(design['profile'])
    if n > 256:
        parser.error('This exact spectrum adapter supports N <= 256')
    columns = list(design['columns'])
    core = tuple(i for i in profile if 1 << i.bit_count() >= args.target)
    floor = min(1 << i.bit_count() for i in profile)
    # Each declared direction set has a fixed, causal coefficient order.
    sources = core if args.directions == 'core' else profile
    edges = tuple((j, i) for j in range(n) if j not in profile
                  and (args.directions == 'all' or 1 << j.bit_count() >= floor)
                  for i in sources if i < j)
    moves = coefficient_moves(edges, args.move_size, args.pair_scope, args.target)
    args.output.mkdir(parents=True, exist_ok=False)
    initial = measure(n, profile, columns, args.target, max(args.work, 300000000),
                      policy=args.policy)
    if not distance_pass(initial['spectrum'], args.target):
        raise ValueError('The initial design does not meet the target')
    best = int(initial['spectrum'].get(args.target, 0))
    initial_count = best
    write(args.output / 'protocol.json', dict(N=n, K=len(profile), target=args.target,
          source=str(args.source.resolve()), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
          protected_core=core, candidate_edges=edges, candidate_order='increasing (frozen index, information index)',
          initial_A_target=best, initial_spectrum=initial, reference_support_input=False,
          reference_profile_input=False, trials=args.trials, seconds=args.seconds, max_work=args.work,
          policy=args.policy, directions=args.directions, global_optimality_claim=False))
    protocol = read(args.output / 'protocol.json')
    protocol.update(move_size=args.move_size, pair_scope=args.pair_scope,
                    unknown_streak_limit=args.unknown_streak,
                    candidate_move_count=len(moves),
                    joint_order='lowest-weight sources first, same-source pairs first, then lexicographic')
    write(args.output / 'protocol.json', protocol)
    started = time.perf_counter()
    stats, records, improvements, seen = Counter(), [], [], set()

    def save_best(proof):
        ph, sh = identities(n, profile, columns)
        code = dict(N=n, K=len(profile), profile=profile, columns=columns,
                    profile_hash=ph, support_hash=sh, name=ph[:12]+'_'+sh[:12],
                    dmin=args.target if best else None, distance_lower_bound=args.target,
                    Admin=best if best else None, A_target=best, target=args.target,
                    GA_proxy=raw.get('GA_proxy'), verification=proof,
                    initialization='Verified independently constructed support',
                    reference_support_read=False, global_optimality_claim=False)
        write(args.output / 'code.json', code)
        write(args.output / 'improvements.json', improvements)

    save_best(initial)
    complete_sweep, last_progress = False, started - 1
    unknown_streak, resource_stop = 0, False
    with (args.output / 'events.jsonl').open('w', encoding='utf-8') as log:
        while len(records) < args.trials and time.perf_counter()-started < args.seconds:
            changed, exhausted = False, True
            for move in moves:
                if len(records) >= args.trials or time.perf_counter()-started >= args.seconds:
                    exhausted = False
                    break
                proposed = apply_coefficient_move(columns, move)
                key = tuple(proposed)
                if key in seen:
                    continue
                seen.add(key)
                edit = (dict(edge=list(move[0])) if args.move_size == 1
                        else dict(edges=[list(edge) for edge in move]))
                event = dict(**edit, before_A_target=best)
                try:
                    answer = measure(n, profile, proposed, args.target, args.work,
                                     policy=args.policy)
                    hist = answer['spectrum']
                    passed = distance_pass(hist, args.target)
                    count = int(hist.get(args.target, 0))
                    event.update(status='exact', distance_pass=passed, A_target=count,
                                 work_units=answer['stats']['work_units'])
                    if passed and count < best:
                        columns, best = proposed, count
                        changed = True
                        improvements.append(dict(**edit, A_target=best,
                                                 seconds=time.perf_counter()-started,
                                                 trial=len(records)+1))
                        save_best(answer)
                        print(f'Improvement: coefficients {move}, A_{args.target}={best}', flush=True)
                except WorkLimitError:
                    event.update(status='unknown', reason='enumeration_work_limit')
                event['seconds'] = time.perf_counter()-started
                unknown_streak = unknown_streak+1 if event['status'] == 'unknown' else 0
                stats[event['status']] += 1
                records.append(event)
                log.write(json.dumps(event)+'\n')
                log.flush()
                if args.unknown_streak and unknown_streak >= args.unknown_streak:
                    resource_stop, exhausted = True, False
                    break
                if changed or time.perf_counter()-last_progress >= .5:
                    write(args.output / 'progress.json', dict(trials=len(records), A_target=best,
                          seconds=event['seconds'], statistics=dict(stats), improvements=improvements))
                    last_progress = time.perf_counter()
                if changed:
                    break
            if resource_stop:
                break
            if exhausted and not changed:
                complete_sweep = True
                break
    final = measure(n, profile, columns, args.target, 300000000, policy=args.policy)
    if not distance_pass(final['spectrum'], args.target) or final['spectrum'].get(args.target, 0) != best:
        raise AssertionError('Final exact spectrum disagrees with the descent archive')
    actions, supports = construction(n, profile, columns)
    other = spectrum(dict(actions=list(map(int, actions)),
                          supports=[list(map(int, s)) for s in supports]),
                     args.target-1, 300000000, backend='python')
    if not distance_pass(other['spectrum'], args.target):
        raise AssertionError('Independent Python distance verification failed')
    save_best(final)
    summary = dict(initial_A_target=initial_count, final_A_target=best,
                   trials=len(records), statistics=dict(stats), improvements=improvements,
                   seconds=time.perf_counter()-started, complete_tested_edge_sweep=complete_sweep,
                   all_tested_edge_queries_exact=stats['unknown']==0,
                   resource_stopped=resource_stop,
                   cpp_python_distance_agree=True, global_optimality_claim=False,
                   tested_edge_domain_size=len(edges), tested_move_domain_size=len(moves),
                   move_size=args.move_size, pair_scope=args.pair_scope, final_spectrum=final,
                   code_path=str((args.output/'code.json').resolve()))
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
