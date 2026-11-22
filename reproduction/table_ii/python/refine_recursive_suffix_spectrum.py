"""Fixed-profile coefficient descent with exact constituent and full-code counts.

The shortened suffix is a repeated subcode, so its minimum-layer multiplicity
is a lower bound on the full-code multiplicity. Strict improvement of this
component is a screening rule, not a complete full-code optimality test. Every
accepted edit also strictly reduces the exact full-code count and preserves
all lower weight layers. No profile or reliability score is changed.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from extended_exact_spectrum import measure, WorkLimitError
from refine_protected_core_spectrum import distance_pass, write
from run_diverse_profile_suite import identities, read
from run_n64_scl_comparison import normalize_json
from select_designs_scl import noiseless_check

def suffix_design(n, profile, columns, length):
    start = n-length
    return tuple(i-start for i in profile if i >= start), [c >> start for c in columns[start:]]

def candidate_edges(n, profile, columns, length, target):
    start, info = n-length, set(profile)
    local_floor = min(1 << (i-start).bit_count() for i in profile if i >= start)
    edges = [(j, i) for j in range(start, n) if j not in info
             and 1 << (j-start).bit_count() >= local_floor
             for i in profile if start <= i < j]
    return tuple(sorted(edges, key=lambda edge: (
        1 << (edge[1]-start).bit_count() >= target,
        not bool(columns[edge[0]]), edge[0], edge[1])))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--suffix-length', type=int, default=256)
    parser.add_argument('--trials', type=int, default=300)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--work', type=int, default=60000000)
    args = parser.parse_args()
    raw = read(args.source)
    design = normalize_json(raw)
    n, profile, columns = design['N'], tuple(design['profile']), design['columns']
    length = args.suffix_length
    if (not 1 <= length <= n <= 1024 or length & (length-1)
            or args.target % (n//length) or min(args.trials, args.seconds, args.work) <= 0):
        parser.error('Require a power-of-two suffix, an integral subcode target and positive limits')
    local_target = args.target//(n//length)
    local_profile, local_columns = suffix_design(n, profile, columns, length)
    edges = candidate_edges(n, profile, columns, length, local_target)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    initial = measure(n, profile, columns, args.target, args.work)
    local = measure(length, local_profile, local_columns, local_target, args.work)
    if not distance_pass(initial['spectrum'], args.target):
        raise ValueError('The initial full code does not meet the requested distance')
    best, local_best = initial['spectrum'].get(args.target, 0), local['spectrum'].get(local_target, 0)
    initial_best, initial_local = best, local_best
    improvements, events, seen, stats = [], [], set(), Counter()
    write(args.output/'protocol.json', dict(N=n, K=len(profile), target=args.target,
          source=str(args.source.resolve()), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
          suffix_length=length, suffix_target=local_target, candidate_edges=edges,
          order='Lowest constituent row-weight sources first; initially active rows first; then (j,i)',
          component_filter='Require strict decrease of repeated shortened-subcode A_target',
          acceptance='Complete full-code spectrum preserves distance and strictly reduces A_target',
          initial_spectrum=initial, initial_suffix_spectrum=local,
          seconds=args.seconds, trials=args.trials, max_work=args.work,
          fixed_profile=True, random_sampling=False, global_optimality_claim=False))

    def save(proof):
        ph, sh = identities(n, profile, columns)
        code = dict(N=n, K=len(profile), profile=profile, columns=columns,
                    profile_hash=ph, support_hash=sh, name=ph[:12]+'_'+sh[:12],
                    target=args.target, dmin=args.target if best else None,
                    distance_lower_bound=args.target, Admin=best if best else None,
                    A_target=best, GA_proxy=raw.get('GA_proxy'), verification=proof,
                    fixed_profile=True, global_optimality_claim=False)
        write(args.output/'code.json', code)
        write(args.output/'improvements.json', improvements)

    save(initial)
    last_proof, exhausted = initial, False
    with (args.output/'events.jsonl').open('w', encoding='utf-8') as log:
        while len(events) < args.trials and time.perf_counter()-started < args.seconds:
            changed, completed = False, True
            for j, i in edges:
                if len(events) >= args.trials or time.perf_counter()-started >= args.seconds:
                    completed = False
                    break
                proposed = columns.copy()
                proposed[j] ^= 1 << i
                key = tuple(proposed)
                if key in seen:
                    continue
                seen.add(key)
                event = dict(edge=[j, i], before_A_target=best, before_suffix_A=local_best)
                try:
                    lp, lc = suffix_design(n, profile, proposed, length)
                    short = measure(length, lp, lc, local_target, args.work)
                    count = short['spectrum'].get(local_target, 0)
                    event.update(suffix_spectrum=short['spectrum'], suffix_A=count)
                    if not distance_pass(short['spectrum'], local_target):
                        event.update(status='constituent_distance_rejection')
                    elif count >= local_best:
                        event.update(status='component_objective_rejection')
                    else:
                        whole = measure(n, profile, proposed, args.target, args.work)
                        value = whole['spectrum'].get(args.target, 0)
                        passed = distance_pass(whole['spectrum'], args.target)
                        event.update(full_spectrum=whole['spectrum'], A_target=value,
                                     distance_pass=passed)
                        if passed and value < best:
                            columns, best, local_best = proposed, value, count
                            last_proof, changed = whole, True
                            event['status'] = 'accepted'
                            improvements.append(dict(edge=[j, i], A_target=best, suffix_A=local_best,
                                                     trial=len(events)+1,
                                                     seconds=time.perf_counter()-started))
                            save(whole)
                            print(json.dumps(dict(stage='improvement', **improvements[-1])), flush=True)
                        else:
                            event['status'] = 'full_code_rejection'
                except WorkLimitError:
                    event.update(status='unknown', reason='exact_enumeration_work_limit')
                event['seconds'] = time.perf_counter()-started
                events.append(event)
                stats[event['status']] += 1
                log.write(json.dumps(event)+'\n')
                log.flush()
                write(args.output/'progress.json', dict(trials=len(events), A_target=best,
                      suffix_A=local_best, statistics=dict(stats), improvements=improvements,
                      seconds=event['seconds']))
                if len(events) % 10 == 0:
                    print(json.dumps(dict(stage='progress', trials=len(events), A_target=best,
                                          suffix_A=local_best, seconds=event['seconds'])), flush=True)
                if changed:
                    break
            if completed and not changed:
                exhausted = True
                break
    final = measure(n, profile, columns, args.target, args.work)
    if final['spectrum'] != last_proof['spectrum'] or not distance_pass(final['spectrum'], args.target):
        raise AssertionError('Final complete spectrum disagrees with the accepted construction')
    final_design = dict(N=n, K=len(profile), profile=profile, columns=columns)
    noiseless_check(final_design, 32, 'cpp')
    save(final)
    summary = dict(initial_A_target=initial_best, final_A_target=best,
                   initial_suffix_A=initial_local, final_suffix_A=local_best,
                   trials=len(events), statistics=dict(stats), improvements=improvements,
                   seconds=time.perf_counter()-started,
                   complete_component_filtered_sweep=exhausted,
                   all_queries_complete=stats['unknown'] == 0,
                   fixed_profile=True, noiseless_L32_pass=True, final_spectrum=final,
                   global_optimality_claim=False,
                   code_path=str((args.output/'code.json').resolve()))
    write(args.output/'summary.json', summary)
    print(json.dumps(dict(stage='complete', **summary)), flush=True)

if __name__ == '__main__':
    main()
