"""Bounded, deterministic profile exchanges driven by exact spectrum feedback.

The seed profile is an explicit input. Its support is never used. Every new
profile is completed from zero in the same declared domain. Reliability and
spectral multiplicity are separate objectives, not FER predictions.
"""

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import math
from pathlib import Path
import time
from types import SimpleNamespace
from external_low_weight_spectrum import WorkLimitError
from obstruction_support_search import enumerate_words
from rank_ordered_profile_design import solve_domain, causal_edges
from refine_protected_core_spectrum import measure, distance_pass
from run_diverse_manuscript_profiles import necessary_solver, membership_check
from run_diverse_profile_suite import read, write, identities
from select_structured_domain import initial_structure

def signature(profile, n=64, target=12):
    return (sum(i >= n // 2 for i in profile),) + tuple(
        sum(n*q//4 <= i < n*(q+1)//4 and (1 << i.bit_count()) >= target
            for i in profile) for q in range(4))

def proposals(parent, costs, ceiling, floor=8, target=12):
    """Generate exchanges without reading a target profile or target support."""
    info = set(parent['profile'])
    histogram = parent['source_A_target']
    answer = defaultdict(list)
    for a in sorted(info):
        for b in range(len(costs)):
            if b in info or (1 << b.bit_count()) < floor:
                continue
            profile = sorted(info - {a} | {b})
            cost = sum(costs[i] for i in profile)
            if cost > ceiling:
                continue
            common = dict(profile=profile, removed=a, added=b, GA_cost=cost,
                          parent=parent['name'], parent_depth=parent['depth'],
                          source_A_target=int(histogram.get(str(a), 0)))
            if cost < parent['GA_cost'] - 1e-14:
                answer['reliability'].append(dict(common, mode='reliability'))
            if b < a and common['source_A_target']:
                answer['spectral_source'].append(dict(common, mode='spectral_source'))
            if b < a and (1 << a.bit_count()) >= target and (1 << b.bit_count()) < target:
                answer['core_release'].append(dict(common, mode='core_release'))
    for mode, values in answer.items():
        values.sort(key=lambda x: ((-x['source_A_target'] if mode == 'spectral_source' else 0),
                                  x['GA_cost'], x['removed'], x['added']))
    return answer

def candidate_order(parents, costs, ceiling, seen, floor=8, target=12):
    n = len(costs)
    width = max(1, n//4)
    queues = defaultdict(lambda: defaultdict(list))
    for parent in parents:
        for mode, values in proposals(parent, costs, ceiling, floor, target).items():
            for value in values:
                if tuple(value['profile']) in seen:
                    continue
                key = (signature(value['profile'], n, target), value['removed']//width, value['added']//width)
                queues[mode][key].append(value)
    for mode in queues:
        for key, values in queues[mode].items():
            values.sort(key=lambda x: ((-x['source_A_target'] if mode == 'spectral_source' else 0),
                                      x['GA_cost'], x['parent'], x['removed'], x['added']))
            queues[mode][key] = deque(values)
    modes = ('reliability', 'spectral_source', 'core_release')
    keys = {mode: deque(sorted(queues[mode])) for mode in modes}
    emitted = set()
    while any(keys.values()):
        for mode in modes:
            if not keys[mode]:
                continue
            key = keys[mode].popleft()
            queue = queues[mode][key]
            value = queue.popleft()
            if queue:
                keys[mode].append(key)
            profile = tuple(value['profile'])
            if profile not in emitted:
                emitted.add(profile)
                yield value

def anchors(archive, count, n=64, target=12):
    """Retain both Pareto endpoints and representatives of different splits."""
    fronts = []
    best = float('inf')
    for entry in sorted(archive, key=lambda x: (x['GA_cost'], x['A_target'], x['name'])):
        if entry['A_target'] < best:
            fronts.append(entry)
            best = entry['A_target']
    chosen = []
    # Alternating endpoints avoids a reliability-only beam.
    while fronts and len(chosen) < count:
        chosen.append(fronts.pop(0))
        if fronts and len(chosen) < count:
            chosen.append(fronts.pop())
    labels = {signature(e['profile'], n, target) for e in chosen}
    for entry in sorted(archive, key=lambda x: (x['A_target'], x['GA_cost'], x['name'])):
        if len(chosen) == count:
            break
        if signature(entry['profile'], n, target) not in labels:
            chosen.append(entry)
            labels.add(signature(entry['profile'], n, target))
    return chosen

def refine(profile, columns, target, edges, trials, seconds, work):
    start = time.perf_counter()
    initial = measure(64, profile, columns, target, work)
    assert distance_pass(initial['spectrum'], target)
    best = initial['spectrum'].get(target, 0)
    trace, visited, complete = [], set(), False
    while len(trace) < trials and time.perf_counter()-start < seconds:
        changed = False
        for j, i in edges:
            if len(trace) >= trials or time.perf_counter()-start >= seconds:
                break
            proposed = list(columns)
            proposed[j] ^= 1 << i
            key = tuple(proposed)
            if key in visited:
                continue
            visited.add(key)
            record = dict(edge=[j, i])
            try:
                spectrum = measure(64, profile, proposed, target, work)['spectrum']
                count = spectrum.get(target, 0)
                passed = distance_pass(spectrum, target)
                accepted = passed and count < best
                record.update(status='exact', distance_pass=passed, A_target=count, accepted=accepted)
                if accepted:
                    columns, best, changed = proposed, count, True
            except WorkLimitError:
                record.update(status='unknown', accepted=False)
            trace.append(record)
            if changed:
                break
        else:
            if not changed:
                complete = True
                break
        if not changed and (len(trace) >= trials or time.perf_counter()-start >= seconds):
            break
    return list(columns), dict(initial_A_target=initial['spectrum'].get(target, 0),
                               final_A_target=best, trace=trace, complete_sweep=complete,
                               seconds=time.perf_counter()-start)

def run(args):
    skip_polish = getattr(args, 'skip_polish', False)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    deadline = start + args.seconds
    source = read(args.seed)
    seed = sorted(source['profile'])
    config = read(args.reliability)
    costs = config['costs']
    assert len(costs) == 64 and len(seed) == 32 and source['N'] == 64
    ceiling = args.cost_factor * sum(costs[i] for i in seed)
    write(args.output/'protocol.json', dict(N=64, K=32, dp=8, target=12,
        seed_profile=seed, seed_source=str(args.seed.resolve()), seed_support_used=False,
        learned_profile_input=False, learned_support_input=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reliability_file=str(args.reliability.resolve()), GA_design_EbN0_dB=config['GA_design_EbN0_dB'],
        costs=costs, GA_cost_ceiling=ceiling, GA_proxy_ceiling=-math.expm1(-ceiling),
        rounds=args.rounds, candidates_per_round=args.per_round, seconds=args.seconds,
        support_seconds=args.support_seconds, refine_trials=args.refine_trials,
        final_polish=not skip_polish,
        refine_seconds=args.refine_seconds, expansion='Release core sources only after initial-domain UNSAT',
        global_optimality_claim=False, FER_claim=False))
    solver, variables, _ = necessary_solver(64, 32, 12, 8)
    archive, records, seen = [], [], set()
    import json

    def emit(kind, **fields):
        value = dict(kind=kind, seconds=round(time.perf_counter()-start, 3))
        value.update(fields)
        print(json.dumps(value), flush=True)
        with (args.output/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(value)+'\n')
        write(args.output/'progress.json', value)

    def construct(proposal, round_index):
        profile = proposal['profile']
        seen.add(tuple(profile))
        ph, _ = identities(64, profile, [0]*64)
        folder = args.output/'trials'/ph[:12]
        record = dict(proposal=proposal, profile=profile, profile_hash=ph, round=round_index)
        checked = membership_check(solver, variables, profile, milliseconds=500)
        if checked != 'sat':
            record['status'] = 'necessary_'+checked
            records.append(record)
            write(args.output/'records.json', records)
            return
        tick = time.perf_counter()
        emit('trial_start', profile=ph[:12], round=round_index, mode=proposal['mode'])
        structure = initial_structure(64, profile, 12)
        edges = tuple((structure['rows'][q], i) for q, i in structure['coordinates'])
        cfg = SimpleNamespace(work=args.work, support_seconds=args.support_seconds,
                              backend='z3', deadline=deadline)
        scoped = []
        try:
            result = solve_domain(64, profile, 12, edges, folder/'initial', cfg,
                                  lambda kind, **kw: None)
            scoped.append(dict(domain='initial', status=result['status']))
            if result['status'] == 'unsat' and time.perf_counter() < deadline:
                edges = tuple((j, i) for j in range(64) if j not in profile
                              and (1 << j.bit_count()) >= 8 for i in profile if i < j)
                result = solve_domain(64, profile, 12, edges, folder/'expanded', cfg,
                                      lambda kind, **kw: None)
                scoped.append(dict(domain='row_floor', status=result['status']))
            record.update(status=result['status'], scoped_results=scoped)
            if result['status'] == 'sat':
                columns, refinement = refine(profile, result['columns'], 12, edges,
                    args.refine_trials, min(args.refine_seconds, max(0, deadline-time.perf_counter())), args.work)
                write(folder/'refinement.json', refinement)
                spec = measure(64, profile, columns, 12, args.work)
                words, meta = enumerate_words(64, profile, columns, 12, args.work)
                assert distance_pass(spec['spectrum'], 12)
                histogram = Counter((u & -u).bit_length()-1 for u, w in words if w == 12)
                assert sum(histogram.values()) == spec['spectrum'].get(12, 0)
                ph, sh = identities(64, profile, columns)
                cost = sum(costs[i] for i in profile)
                entry = dict(N=64, K=32, dp=8, target=12, profile=profile, columns=columns,
                    profile_hash=ph, support_hash=sh, name=ph[:12]+'_'+sh[:12],
                    GA_cost=cost, GA_proxy=-math.expm1(-cost), A_target=spec['spectrum'].get(12, 0),
                    Admin=spec['spectrum'].get(12, 0), dmin=12 if spec['spectrum'].get(12, 0) else None,
                    distance_lower_bound=12, source_A_target={str(i): a for i, a in sorted(histogram.items())},
                    depth=proposal.get('parent_depth', -1)+1, proposal=proposal,
                    spectrum=spec['spectrum'], code_path=str((folder/'code.json').resolve()),
                    zero_support_initialization=True)
                write(folder/'code.json', entry)
                archive.append(entry)
                record.update(name=entry['name'], A_target=entry['A_target'], GA_proxy=entry['GA_proxy'])
        except WorkLimitError:
            record.update(status='unresolved', reason='enumeration_work_limit', scoped_results=scoped)
        record['seconds'] = time.perf_counter()-tick
        records.append(record)
        write(folder/'record.json', record)
        write(args.output/'archive.json', archive)
        write(args.output/'records.json', records)
        emit('trial_end', profile=ph[:12], status=record['status'],
             A12=record.get('A_target'), GA_proxy=record.get('GA_proxy'), feasible=len(archive))

    construct(dict(profile=seed, mode='seed_profile_only'), 0)
    if not archive:
        raise RuntimeError('Seed profile could not be completed within the declared budget')
    for round_index in range(1, args.rounds+1):
        if time.perf_counter() >= deadline:
            break
        parents = anchors(archive, args.anchors)
        # The latest generation receives a place even when temporarily dominated.
        latest = sorted((x for x in archive if x['depth'] == round_index-1),
                        key=lambda x: (x['A_target'], x['GA_cost']))
        parents += [x for x in latest[:args.anchors] if x not in parents]
        count = 0
        emit('round_start', round=round_index, parents=[x['name'] for x in parents])
        for proposal in candidate_order(parents, costs, ceiling, seen):
            if count >= args.per_round or time.perf_counter() >= deadline:
                break
            construct(proposal, round_index)
            count += 1
    shortlist = anchors(archive, args.anchors)
    write(args.output/'shortlist_before_polish.json', shortlist)
    # Equal final refinement in the full causal domain, outside the proposal loop.
    polished = list(shortlist) if skip_polish else []
    for entry in (() if skip_polish else shortlist):
        emit('polish_start', name=entry['name'], A12=entry['A_target'])
        profile = entry['profile']
        columns, refinement = refine(profile, entry['columns'], 12, causal_edges(64, profile),
                                     2000, 20, args.work)
        cpp = measure(64, profile, columns, 12, args.work)
        py = measure(64, profile, columns, 12, args.work, backend='python')
        assert cpp['spectrum'] == py['spectrum'] and distance_pass(py['spectrum'], 12)
        ph, sh = identities(64, profile, columns)
        final = dict(entry, columns=columns, support_hash=sh, name=ph[:12]+'_'+sh[:12],
                     A_target=cpp['spectrum'].get(12, 0), Admin=cpp['spectrum'].get(12, 0),
                     spectrum=cpp['spectrum'], cpp_python_spectrum_agree=True,
                     before_polish_name=entry['name'])
        final['source_A_target_before_polish'] = final.pop('source_A_target')
        words, _ = enumerate_words(64, profile, columns, 12, args.work)
        histogram = Counter((u & -u).bit_length()-1 for u, w in words if w == 12)
        assert sum(histogram.values()) == final['A_target']
        final['source_A_target'] = {str(i): a for i, a in sorted(histogram.items())}
        path = args.output/'designs'/(final['name']+'.json')
        final['code_path'] = str(path.resolve())
        write(path, final)
        write(args.output/'polish'/(ph[:12]+'.json'), refinement)
        polished.append(final)
        emit('polish_end', name=final['name'], A12=final['A_target'])
    write(args.output/'shortlist.json', polished)
    summary = dict(seconds=time.perf_counter()-start, proposals=len(records),
        feasible_profiles=len(archive), statuses=dict(Counter(x['status'] for x in records)),
        shortlist=[{k: x[k] for k in ('name', 'GA_proxy', 'A_target', 'depth', 'code_path')}
                   for x in polished], global_optimality_claim=False,
        scope='One-swap focused profile graph with bounded shared-coefficient completion')
    write(args.output/'summary.json', summary)
    emit('completed', **summary)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--reliability', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--per-round', type=int, default=24)
    parser.add_argument('--anchors', type=int, default=8)
    parser.add_argument('--cost-factor', type=float, default=1.65)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--support-seconds', type=float, default=4)
    parser.add_argument('--refine-trials', type=int, default=96)
    parser.add_argument('--refine-seconds', type=float, default=3)
    parser.add_argument('--work', type=int, default=30000000)
    parser.add_argument('--skip-polish', action='store_true',
                        help='Keep the proposal archive without separately polishing its shortlist')
    run(parser.parse_args())

if __name__ == '__main__':
    main()
