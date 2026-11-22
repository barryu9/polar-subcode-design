"""Bounded manuscript profile workflow with exact, zero-initialized supports.

Reliability priority applies within structural classes. All verified profiles
are archived. A small shortlist is selected only after the declared budget.
No historical profile, support, or algebraic auxiliary code is an input.
"""

import argparse
from collections import defaultdict, deque
import hashlib
import heapq
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from ranked_class_profiles import ranked_class_profiles
from layered_profile_classes import layer_label, layered_seed_profiles, ranked_layer_profiles, structural_signature, obstruction_neighbors

ROOT = Path(__file__).resolve().parents[1]

CASES = [(64, 32, 8, 12), (128, 64, 8, 16), (256, 64, 32, 48),
         (256, 128, 16, 24), (256, 171, 8, 12), (1024, 512, 16, 24)]

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    for attempt in range(50):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(min(.2,.01*(attempt+1)))

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(',', ':')).encode('ascii')).hexdigest()

def identities(n, profile, columns):
    return (digest(dict(N=n, profile=sorted(profile))),
            digest(dict(N=n, columns_hex=[hex(c) for c in columns])))

def label(n, profile, floor):
    low = [i for i in profile if (1 << i.bit_count()) == floor]
    return (len(low), max(low) // max(1, n // 8) if low else -1)

def class_seed(n, k, floor, count, bucket, costs):
    """Exact additive-cost minimizer of a count/maximum-index class."""
    width = max(1, n // 8)
    lo, hi = bucket * width, min(n, (bucket + 1) * width)
    low = sorted((i for i in range(hi) if 1 << i.bit_count() == floor),
                 key=lambda i: (costs[i], i))
    high = sorted((i for i in range(n) if 1 << i.bit_count() > floor),
                  key=lambda i: (costs[i], i))
    if not 1 <= count <= min(k, len(low)) or k-count > len(high):
        return None
    chosen = low[:count]
    if not any(i >= lo for i in chosen):
        tail = next((i for i in low if i >= lo), None)
        if tail is None:
            return None
        chosen[-1] = tail
    return tuple(sorted(chosen + high[:k-count]))

def select_diverse(archive, count=8):
    groups = defaultdict(list)
    key = lambda r: (r['GA_cost'], tuple(r['profile']))
    for row in archive:
        signature = row.get('structural_signature')
        group = (('exact_tail', signature['layer_counts_and_last'][2])
                 if signature else tuple(row['class']))
        groups[group].append(row)
    for group in groups.values():
        group.sort(key=key)
    selected = []
    for ordinal in range(max(map(len, groups.values()), default=0)):
        representatives = sorted((g[ordinal] for g in groups.values()
                                  if ordinal < len(g)), key=key)
        selected.extend(representatives[:max(0, count-len(selected))])
        if len(selected) >= count:
            break
    return selected

def reliability_class_order(queues, costs, layered=False):
    """Give each exact tail one turn before another layer allocation at that tail."""
    key = lambda c: (min(sum(costs[i] for i in p['profile']) for p in queues[c]), c)
    ordered = sorted((c for c in queues if queues[c]), key=key)
    if not layered:
        return ordered
    tails = defaultdict(list)
    for c in ordered:
        tails[c[2]].append(c)
    result = []
    for ordinal in range(max(map(len, tails.values()), default=0)):
        result.extend(sorted((group[ordinal] for group in tails.values()
                              if ordinal < len(group)), key=key))
    return result

class Coverage:
    """Fair rank-tuple coverage in the declared row-floor family."""

    def __init__(self, order, k):
        self.order, self.k = order, k
        initial = tuple(range(k))
        self.heap = [(sum(initial), initial)]
        self.seen = {initial}

    def next(self):
        if not self.heap:
            return None
        _, a = heapq.heappop(self.heap)
        for j in range(self.k):
            if a[j] + 1 < (a[j+1] if j+1 < self.k else len(self.order)):
                child = a[:j] + (a[j]+1,) + a[j+1:]
                if child not in self.seen:
                    self.seen.add(child)
                    heapq.heappush(self.heap, (sum(child), child))
        return tuple(sorted(self.order[i] for i in a))

def improve(n, profile, floor, costs):
    old = set(profile)
    rows = []
    for a in profile:
        for b in range(n):
            if b in old or costs[b] >= costs[a] or 1 << b.bit_count() < floor:
                continue
            child = tuple(sorted(old - {a} | {b}))
            if min(1 << i.bit_count() for i in child) != floor:
                continue
            rows.append(dict(profile=child, mode='reliability_improvement',
                             removed=a, added=b, delta_cost=costs[b]-costs[a]))
    return sorted(rows, key=lambda r: (r['delta_cost'], r['removed'], r['added']))

def feedback_kind(result):
    """Resource failures never certify a structural obstruction."""
    local = result.get('result', result)
    if result['status'] == 'verified':
        return 'verified'
    if proved_obstruction(result) is not None:
        return 'domain_obstruction'
    if local.get('status') == 'unsat':
        return 'domain_obstruction'
    if local.get('status') == 'pending' and local.get('method'):
        return 'unfinished_solver_alternative'
    return 'resource_or_representation_limit'

def proved_obstruction(result):
    """Return evidence only for an explicitly infeasible, scoped design domain."""
    local = result.get('result', result)
    if local.get('status') == 'unsat':
        return dict(kind='domain_obstruction', status='unsat',
                    method=local.get('method', 'declared_support_domain'),
                    failure_length=result.get('failure_length'), full_profile_excluded=False)
    for stage in local.get('stages', []):
        outer = stage.get('outer_result', {})
        if outer.get('status') == 'unsat' and 'distance_lower_bound' not in stage:
            return dict(kind='domain_obstruction', status='unsat',
                        method=stage['status'], failure_length=stage['N'],
                        full_profile_excluded=False)
    return None

def backend_limited(result):
    """A fixed representation limit cannot improve by repeating the same trial."""
    local = result.get('result', result)
    return (result.get('status') == 'representation_limit'
            or local.get('status') == 'representation_limit'
            or any(stage.get('status') == 'representation_limit'
                   for stage in local.get('stages', [])))

def has_pending_candidate(cache, n, profile, target):
    """Find saved suffix assignments that still require distance verification."""
    length = 1
    while length <= n:
        local = tuple(i - n + length for i in profile if i >= n - length)
        delta = (target * length + n - 1) // n
        key = hashlib.sha256(json.dumps((length, local, delta)).encode()).hexdigest()[:16]
        folder = cache / 'suffix_cache' / key
        outcome = folder / 'outcome.json'
        if not outcome.exists() or read(outcome).get('status') != 'sat':
            candidates = [folder / 'unverified_candidate.json',
                          folder / 'direct/unverified_candidate.json',
                          folder / 'outer/unverified_candidate.json']
            if any(p.exists() for p in candidates) or any(folder.glob('attempt_*/outer/unverified_candidate.json')):
                return True
        length *= 2
    return False

def has_outer_progress(cache, n, profile, target):
    """A saved nonempty outer prefix is resumable work, not a failed profile."""
    key = hashlib.sha256(json.dumps((n,tuple(profile),target)).encode()).hexdigest()[:16]
    folder = cache/'suffix_cache'/key/'outer'
    if (folder/'frontier.json').exists() and (folder/'result.json').exists():
        return (read(folder/'result.json').get('status') == 'pending'
                and read(folder/'frontier.json').get('index',0) > 0)
    return False

def worker(args):
    if getattr(args, 'cold_run_root', None):
        from run_empty_archive_six_group_replay import guard_history
        guard_history(args.cold_run_root)
    from feasible_design_bank import generator_rows, validate_design
    from select_designs_scl import noiseless_check
    from structure_feedback_profile_design import protected_trial, advance_full_job
    from replay_phase_timing import install_verification_timing
    raw = read(args.profile)
    n, profile, target = raw['N'], tuple(raw['profile']), raw['target']
    args.output.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    verification_timing = install_verification_timing(args.output/'phase_timing.json')

    def emit(kind, **fields):
        print(json.dumps(dict(kind=kind, elapsed_seconds=time.perf_counter()-begin, **fields)), flush=True)

    effective_seconds = args.support_seconds
    write(args.output/'execution_options.json',dict(
        baseline_solver_seconds=args.support_seconds,effective_solver_seconds=effective_seconds,
        trial_wall_seconds=args.trial_seconds,
        construction_work=args.work, verification_work=args.verification_work,
        outer_solver_seconds=args.outer_seconds,
        rule='Independent solver, construction and verification budgets',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    cfg = SimpleNamespace(work=args.work, support_seconds=effective_seconds,
                          direct_rows=8, nodes=args.solver_nodes, radius_limit=3,
                          joint_coefficients=args.joint_coefficients,
                          joint_max_length=getattr(args,'joint_max_length',128),
                          verification_work=args.verification_work,
                          outer_seconds=args.outer_seconds,
                          retry_incomplete=args.retry_incomplete,
                          pattern_outer=args.pattern_outer, backend='auto',
                          deadline=time.perf_counter()+args.trial_seconds)
    # A pending frontier is not evidence for releasing new coordinates. Continue
    # that same domain before changing its representation or admissible space.
    if args.mode == 'complete' and getattr(args,'cold_run_root',None):
        length = 1
        pending = []
        while length <= n:
            local = tuple(i-(n-length) for i in profile if i >= n-length)
            delta = (target*length+n-1)//n
            key = hashlib.sha256(json.dumps((length,local,delta)).encode()).hexdigest()[:16]
            outcome = args.cache/'suffix_cache'/key/'outcome.json'
            if outcome.exists():
                saved = read(outcome)
                if saved.get('status') in ('pending','unknown'):
                    pending.append(dict(N=length, cache=key, status=saved['status']))
            length *= 2
        if pending:
            args.mode = 'structured'
            cfg.retry_incomplete = True
            cfg.nodes *= 4
            write(args.output/'pending_frontier_service.json',dict(
                preceding_domains=pending, effective_nodes=cfg.nodes,
                policy='Resume the same saved frontier; do not infer a domain obstruction'))
    if args.mode == 'complete':
        if n > 256:
            result = dict(status='representation_limit',full_profile_excluded=False,
                          reason='The complete explicit-directory backend supports lengths at most 256')
        else:
            result = advance_full_job(profile, n, target, args.output/'complete', cfg, emit)
    elif n <= 256:
        result = None
        dp = min(1 << i.bit_count() for i in profile)
        if args.gap_first and dp < target <= 3*dp//2:
            from run_rm_ga_profile_polish import (protected_recursive,
                                                  ProtectedCompletionFailure, canonical_check)
            from external_low_weight_spectrum import WorkLimitError
            try:
                gap_cfg = SimpleNamespace(work=args.work, seconds=effective_seconds)
                columns = protected_recursive(n,profile,target,args.output/'gap_trial',gap_cfg)
                proof = canonical_check(n,profile,columns,target)
                proof.update(method='Complete recursive exclusions with verified row-threshold spectral gaps',
                             evidence_path=str((args.output/'gap_trial').resolve()),
                             external_algebraic_code_used=False)
                result = dict(status='verified',columns=columns,verification=proof)
            except (ProtectedCompletionFailure,ValueError,WorkLimitError) as error:
                write(args.output/'gap_trial_outcome.json',dict(status='retained',reason=str(error),
                       full_profile_infeasibility=False))
        if result is None:
            result = protected_trial(n, profile, target, args.cache, cfg, emit)
    else:
        from audit_suffix_construction import audit
        from run_manuscript_suffix_construction import run
        output = args.output/'suffix'
        run(SimpleNamespace(profile=args.profile, target=target, output=output,
                            seconds=args.support_seconds, nodes=100000, work=args.work))
        summary = read(output/'result.json')
        if summary['status'] == 'verified_feasible':
            proof = audit(output)
            result = dict(status='verified', columns=read(output/'support.json')['columns'],
                          verification=proof)
        else:
            stage = next((s for s in summary['stages'] if 'distance_lower_bound' not in s), {})
            length = stage.get('N', n)
            result = dict(status='retained', failure_length=length, verified_tail=length//2,
                          result=summary, full_profile_excluded=False)
    if result['status'] == 'verified':
        columns = result['columns']
        validate_design(n, profile, columns)
        rows = generator_rows(n, profile, columns)
        upper = min(row.bit_count() for row in rows)
        proof = result['verification']
        lower = proof['distance_lower_bound']
        assert lower >= target and upper >= lower
        proof['distance_upper_bound'] = upper
        if lower == upper:
            proof['dmin'] = lower
        noiseless_check(dict(N=n, K=len(profile), profile=profile, columns=columns), 32, 'cpp')
        result['noiseless_L32_checked'] = True
    result['worker_seconds'] = time.perf_counter()-begin
    result['phase_timing'] = verification_timing.finish(result['worker_seconds'])
    result['effective_solver_seconds'] = effective_seconds
    result['worker_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(args.output/'result.json', result)
    emit('worker_complete', status=result['status'])

def run_case(case, args):
    import z3
    from polar_code.rp_funcs import rp_GA
    from run_diverse_manuscript_profiles import membership_check, necessary_solver
    from select_ga_necessary_profile import ga_priority_completion, base_solver
    from structure_feedback_profile_design import repair_neighbors, select_seed
    from audit_general_candidate_profiles import audit
    from feasible_design_bank import generator_rows, code_fingerprint

    n, k, floor, target = case
    root = args.output/f'n{n}_k{k}_dp{floor}_d{target}'
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and (root/'configuration.json').exists():
        saved = read(root/'configuration.json')
        if saved.get('layered_classes', False) != args.layered_classes:
            raise ValueError('Resume requires the same structural class definition')
    class_key = lambda p: layer_label(p, floor) if args.layered_classes else label(n,p,floor)
    started = time.perf_counter()
    past_elapsed = 0
    if args.resume and (root/'events.jsonl').exists():
        past_elapsed = max(json.loads(line)['seconds'] for line in
                           (root/'events.jsonl').read_text().splitlines())
    deadline = started + max(0, args.case_seconds-past_elapsed)
    ga = rp_GA(n, k, args.snr)
    errors = list(map(float, ga.pe))
    costs = [-math.log1p(-p) for p in errors]
    order = sorted(range(n), key=lambda i: (costs[i], i))
    eligible = [i for i in order if 1 << i.bit_count() >= floor]
    qmin = max(1, k-sum(1 << i.bit_count() > floor for i in range(n)))
    qmax = min(k, sum(1 << i.bit_count() == floor for i in range(n)))
    qga = sum(1 << i.bit_count() == floor for i in eligible[:k])
    counts = sorted({min(qmax, qmin+d) for d in (0, 1, 2, 3, 4, 7, 11, 15, 19, 23)}
                    | {max(qmin, min(qmax, qga+d)) for d in (-1, 0, 1)})
    if args.dense_counts:
        counts = sorted(set(counts)|set(range(qmin,min(qmax,qga+1)+1)))
    if args.layered_classes:
        counts = list(range(qmin, qmax+1))
    archive, trials, retained, seed_log = [], [], deque(), []
    queues, queued, attempted, excluded = defaultdict(deque), set(), set(), set()
    focus_queue = deque()
    emit_path = root/'events.jsonl'
    if args.resume and (root/'trials.json').exists():
        trials = read(root/'trials.json')
        archive = read(root/'archive.json')
        pending = {}
        for record in trials:
            p = tuple(record['profile'])
            attempted.add(p)
            if record['status'] not in ('verified','excluded'):
                pending[p] = read(Path(record['path'])/'proposal.json')
        retained.extend(pending.values())
        if (root/'unfinished.json').exists():
            saved_state = read(root/'unfinished.json')
            retained = deque(saved_state['retained_support_tasks'])
            focus_queue = deque(saved_state.get('focused_proposals', []))
            for proposals in saved_state['queues'].values():
                for proposal in proposals:
                    p = tuple(proposal['profile'])
                    if p not in attempted and p not in queued:
                        queues[class_key(p)].append(proposal)
                        queued.add(p)
        for proposal in retained:
            previous = proposal.get('previous_trial')
            if previous and (Path(previous)/'outcome.json').exists():
                proposal['deferred_representation'] = backend_limited(read(Path(previous)/'outcome.json'))

    def emit(kind, **fields):
        row = dict(case=f'{n},{k}', kind=kind, seconds=past_elapsed+time.perf_counter()-started, **fields)
        print(json.dumps(row), flush=True)
        with emit_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        write(root/'progress.json', row)

    def enqueue(proposal, front=False, focus=False):
        p = tuple(proposal['profile'])
        if p not in queued and p not in attempted:
            queued.add(p)
            queue = queues[class_key(p)]
            (queue.appendleft if front else queue.append)(proposal)
            if focus:
                focus_queue.appendleft(proposal)

    if args.full_necessary and n <= 256:
        solver, variables = base_solver(n,k,target)
        solver.add(*[z3.Not(variables[i]) for i in range(n) if 1 << i.bit_count() < floor])
        checks = dict(method='Existing full table-free profile prechecks',
                      implementation='select_ga_necessary_profile.base_solver',
                      includes_even_all_one_bounds=True,includes_hamming_interface=True,
                      includes_full_length_coset_capacities=True)
    else:
        solver, variables, checks = necessary_solver(n, k, target, floor)
    solver.add(z3.Or(*[variables[i] for i in range(n) if 1 << i.bit_count() == floor]))
    write(root/'configuration.json', dict(N=n, K=k, dp=floor, target=target,
        GA_design_EbN0_dB=args.snr, pe=errors, costs=costs, order=order, counts=counts,
        structural_label=('(floor-layer count, next-layer count, exact maximum floor-layer index)'
                          if args.layered_classes else
                          '(number of floor-weight information rows, max index // (N/8))'),
        layered_classes=args.layered_classes, layered_seed_limit=args.layered_seed_limit,
        layered_scheduling='Reliability-ranked round robin across exact tail positions',
        shortlist_rule=('One representative per exact tail before a second representative'
                        if args.layered_classes else 'One representative per legacy class'),
        layered_source_sha256=hashlib.sha256(Path(__file__).with_name(
            'layered_profile_classes.py').read_bytes()).hexdigest(),
        necessary_checks=checks, case_seconds=args.case_seconds, trial_limit=args.trials,
        trial_wall_seconds=args.trial_seconds, per_solver_seconds=args.support_seconds,
        outer_solver_seconds=args.outer_seconds,
        work_limit=args.work, verification_work=args.verification_work,
        focus_batch=args.focus_batch, anchor=args.anchor, shortlist_size=8, reference_profile_read=False,
        class_depth=args.class_depth,dense_counts=args.dense_counts,
        joint_initial_domain_coefficient_limit=args.joint_coefficients,
        reference_support_read=False, random_sampling=False, support_initialization='all_zero',
        GA_proxy_definition='1 - product(1 - pe[i]); smaller is preferred',
        GA_cost_definition='sum(-log(1-pe[i])); used for additive priorities',
        bounded_execution=True, scope='Focused classes, fair rank coverage and bounded retained-task service'))
    emit('necessary_checks_ready', counts=counts)
    if args.anchor and not trials:
        anchor_order = sorted(range(n), key=lambda i: (-i.bit_count(), costs[i], i))
        anchor = select_seed(n,k,target,anchor_order,60,root/'anchor',emit)
        anchor_floor = min(1 << i.bit_count() for i in anchor)
        if anchor_floor == floor:
            enqueue(dict(profile=anchor,mode='rm_ga_cold_anchor'),front=True,focus=True)
        else:
            emit('anchor_outside_declared_floor', actual_dp=anchor_floor, declared_dp=floor,
                 scope='Skip this optional seed, retain all requested-floor classes')
    raw_seeds = []
    from itertools import islice
    if args.layered_classes:
        for seed in islice(layered_seed_profiles(n,k,floor,costs), args.layered_seed_limit):
            count, next_count, last = seed['class_label']
            alternatives = ([seed] if args.class_depth == 1 else list(islice(
                ranked_layer_profiles(n,k,floor,count,next_count,last,costs), args.class_depth)))
            for ordinal, alternative in enumerate(alternatives, 1):
                raw_seeds.append((alternative['cost'],seed['class_label'],alternative['profile'],ordinal))
    else:
        for count in counts:
            for bucket in range(8):
                if args.class_depth==1:
                    p = class_seed(n,k,floor,count,bucket,costs)
                    alternatives = [dict(profile=p,cost=sum(costs[i] for i in p))] if p else []
                else:
                    alternatives = list(islice(ranked_class_profiles(n,k,floor,count,bucket,costs),args.class_depth))
                for ordinal,alternative in enumerate(alternatives,1):
                    raw_seeds.append((alternative['cost'],(count,bucket),alternative['profile'],ordinal))
    # First retain every cheap feasible seed, then repair failed seed memberships.
    write(root/'raw_seeds.json', [dict(GA_cost=cost,class_label=c,profile=p,class_ordinal=ordinal)
                                 for cost,c,p,ordinal in sorted(raw_seeds)])
    unresolved_seeds = []
    seed_check_deadline = min(deadline, time.perf_counter() + args.seed_check_seconds)
    for _, structural_class, p, ordinal in sorted(raw_seeds):
        if time.perf_counter() >= seed_check_deadline:
            break
        answer = membership_check(solver, variables, p, 1000)
        seed_log.append(dict(class_label=structural_class, profile=p, necessary_answer=answer,class_ordinal=ordinal))
        if answer == 'sat':
            enqueue(dict(profile=p,mode=('layered_class_seed' if args.layered_classes else 'class_seed')
                         if ordinal==1 else 'ranked_class_alternative',
                         class_ordinal=ordinal))
        elif ordinal==1:
            unresolved_seeds.append((structural_class, p))
    seed_deadline = min(deadline, started + min(45, args.case_seconds/4))
    for structural_class, p in unresolved_seeds:
        if time.perf_counter() >= seed_deadline:
            break
        low = [i for i in range(n) if 1 << i.bit_count() == floor]
        solver.push()
        count = structural_class[0]
        solver.add(z3.PbEq([(variables[i], 1) for i in low], count))
        if args.layered_classes:
            _, next_count, last = structural_class
            middle = [i for i in range(n) if 1 << i.bit_count() == 2*floor]
            solver.add(z3.PbEq([(variables[i],1) for i in middle], next_count))
            solver.add(variables[last], *[z3.Not(variables[i]) for i in low if i > last])
        else:
            _, bucket = structural_class
            lo, hi = bucket*(n//8), (bucket+1)*(n//8)
            solver.add(*[z3.Not(variables[i]) for i in low if i >= hi])
            solver.add(z3.Or(*[variables[i] for i in low if lo <= i < hi]))
        seed, status = ga_priority_completion(solver, variables, order,
                                              min(seed_deadline, time.perf_counter()+2))
        solver.pop()
        seed_log.append(dict(class_label=structural_class, status=status,
                             profile=seed, method='necessary_constrained_lexicographic_seed'))
        if seed:
            enqueue(dict(profile=seed, mode='necessary_constrained_class_seed'))
    write(root/'seed_log.json', seed_log)
    coverage = Coverage(eligible, k)
    if args.resume and (root/'unfinished.json').exists():
        saved_state = read(root/'unfinished.json')
        coverage.heap = [(value,tuple(p)) for value,p in saved_state['coverage_heap']]
        heapq.heapify(coverage.heap)
        coverage.seen = {tuple(p) for p in saved_state['coverage_seen']}
    class_order = (sorted(queues) if args.class_order == 'structural' else
                   reliability_class_order(queues, costs, args.layered_classes))
    write(root/'class_order.json', class_order)
    cursor = 0
    focused = 0
    burst = 0
    service_due = False
    coverage_due = False
    emit('seeds_ready', classes=len(queues), queued=len(queued))

    def checkpoint():
        write(root/'unfinished.json', dict(retained_support_tasks=list(retained),
            focused_proposals=list(focus_queue),
            queues={str(c):list(q) for c,q in queues.items()}, coverage_heap=coverage.heap,
            coverage_seen=sorted(coverage.seen), coverage_order=eligible,
            excluded_profiles=sorted(excluded), global_exhaustion=False))

    checkpoint()

    while time.perf_counter() < deadline and len(trials) < args.trials:
        if len(archive) >= args.stop_after:
            break
        mode, proposal = 'structured', None
        recovery = next((p for p in retained
                         if not p.get('deferred_representation', False)
                         and not p.get('verification_recovery_attempted', False)
                         and (has_pending_candidate(root/'construction_cache', n,
                                                    p['profile'], target)
                              or has_outer_progress(root/'construction_cache', n,
                                                    p['profile'], target))), None)
        if recovery is not None:
            retained.remove(recovery)
            proposal = {**recovery, 'verification_recovery_attempted':True}
            mode = 'retry'
        elif service_due and any(not p.get('deferred_representation', False) for p in retained):
            proposal = next(p for p in retained if not p.get('deferred_representation', False))
            retained.remove(proposal)
            mode, service_due = proposal.get('retry_mode', 'complete'), False
        elif coverage_due:
            coverage_due = False
            for _ in range(256):
                p = coverage.next()
                if p is None:
                    break
                if p not in attempted and min(1 << i.bit_count() for i in p) == floor:
                    if membership_check(solver, variables, p, 500) == 'sat':
                        proposal = dict(profile=p, mode='rank_coverage')
                        break
        while proposal is None and focus_queue and burst < args.focus_batch:
            candidate = focus_queue.popleft()
            if tuple(candidate['profile']) not in attempted:
                proposal = candidate
                burst += 1
                focused += 1
        if proposal is None:
            for c in sorted(queues):
                if c not in class_order:
                    class_order.append(c)
            if not any(queues.values()):
                break
            while True:
                selected_class = class_order[cursor % len(class_order)]
                cursor += 1
                if queues[selected_class]:
                    break
            proposal = queues[selected_class].popleft()
            burst = 0
            focused += 1
        p = tuple(proposal['profile'])
        if p in excluded or (mode == 'structured' and p in attempted):
            continue
        if mode == 'structured':
            attempted.add(p)
            if membership_check(solver, variables, p, 1000) != 'sat':
                continue
            if args.full_necessary:
                checked = audit(n,p,target,False)
                if checked['target_status'] == 'excluded':
                    excluded.add(p)
                    emit('necessary_exclusion', profile_hash=identities(n,p,[0]*n)[0][:12])
                    continue
        ph, _ = identities(n, p, [0]*n)
        trial = root/'trials'/f'{len(trials):03d}_{ph[:12]}_{mode}'
        write(trial/'profile.json', dict(N=n, K=k, profile=p, target=target, dp=floor))
        write(trial/'proposal.json', proposal)
        available = deadline-time.perf_counter()
        if available < 5:
            break
        limit = min(args.trial_seconds, available)
        output = trial/'construction'
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                   '--profile', str(trial/'profile.json'), '--output', str(output),
                   '--cache', str(root/'construction_cache'), '--mode',
                   'structured' if mode == 'retry' else mode,
                   '--trial-seconds', str(limit), '--support-seconds', str(args.support_seconds),
                   '--work', str(args.work), '--verification-work', str(args.verification_work),
                   '--outer-seconds', str(args.outer_seconds),
                   '--solver-nodes', str(args.solver_nodes),
                   '--joint-max-length', str(getattr(args,'joint_max_length',128)),
                   '--joint-coefficients', str(args.joint_coefficients)]
        if getattr(args,'cold_run_root',None):
            command += ['--cold-run-root', str(args.cold_run_root)]
        if mode == 'retry':
            command.append('--retry-incomplete')
        if args.gap_first:
            command.append('--gap-first')
        if args.pattern_outer:
            command.append('--pattern-outer')
        emit('trial_started', number=len(trials), profile_hash=ph[:12],
             label=class_key(p), mode=mode, proposal=proposal['mode'])
        tick = time.perf_counter()
        with (trial/'stdout.log').open('w', encoding='utf-8') as log:
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                returncode = process.wait(timeout=limit)
                result = read(output/'result.json') if returncode == 0 else dict(
                    status='implementation_failure', returncode=returncode)
            except subprocess.TimeoutExpired:
                # Terminate only this owned worker tree, including the Windows venv launcher.
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                process.wait()
                result = dict(status='wall_limit', full_profile_excluded=False)
        record = dict(profile_hash=ph, profile=p, class_label=class_key(p),
                      structural_signature=structural_signature(n,p,floor),
                      status=result['status'], mode=mode, proposal=proposal['mode'],
                      trial_seconds=time.perf_counter()-tick, path=str(trial.resolve()))
        write(trial/'outcome.json', result)
        neighbors = []
        if result['status'] == 'verified':
            columns = result['columns']
            ph, sh = identities(n, p, columns)
            name = ph[:12]+'_'+sh[:12]
            cost = sum(costs[i] for i in p)
            proof = result['verification']
            code = dict(N=n, K=k, name=name, profile_hash=ph, support_hash=sh,
                        profile=list(p), columns=columns, columns_hex=[hex(c) for c in columns],
                        dp=min(1 << i.bit_count() for i in p), target=target,
                        distance_lower_bound=proof['distance_lower_bound'], dmin=proof.get('dmin'),
                        distance_upper_bound=proof['distance_upper_bound'], verification=proof,
                        GA_cost=cost, GA_proxy=-math.expm1(-cost), GA_sum_pe=sum(errors[i] for i in p),
                        GA_max_pe=max(errors[i] for i in p), GA_design_EbN0_dB=args.snr,
                        **{'class':label(n,p,floor)}, coefficients=sum(c.bit_count() for c in columns),
                        dynamic_expressions={str(j):[i for i in p if c >> i & 1]
                                             for j,c in enumerate(columns) if c},
                        code_hash=code_fingerprint(n,generator_rows(n,p,columns)),
                        construction_path=str(output.resolve()), initialization='all_zero',
                        historical_design_read=False, noiseless_L32_checked=True)
            if args.layered_classes:
                code['structural_signature'] = structural_signature(n,p,floor)
            destination = root/'designs'/name
            if p not in {tuple(r['profile']) for r in archive}:
                write(destination/'code.json', code)
                write(destination/'profile.json', {key:code[key] for key in
                      ('N','K','profile','profile_hash','dp','GA_proxy','GA_cost')})
                write(destination/'support.json', {key:code[key] for key in
                      ('N','K','profile','columns','support_hash','dynamic_expressions')})
                archive.append(code)
            neighbors = improve(n, p, floor, costs)[:32]
            record.update(name=name, dmin=code['dmin'], GA_proxy=code['GA_proxy'])
        elif result.get('full_profile_excluded') or result['status'] == 'excluded':
            excluded.add(p)
        else:
            kind = feedback_kind(result)
            record['feedback_kind'] = kind
            visits = proposal.get('service_visits', 0) + (mode in ('retry','complete'))
            retained.append({**proposal, 'previous_trial':str(trial), 'full_profile_excluded':False,
                'service_visits':visits,
                'deferred_representation':backend_limited(result),
                'retry_mode':'retry' if n > 256 or kind == 'resource_or_representation_limit' else 'complete'})
            record['deferred_representation'] = backend_limited(result)
            evidence = proved_obstruction(result)
            length = evidence.get('failure_length') if evidence else None
            if length:
                dimension = sum(i >= n-length for i in p)
                parent = proposal.get('parent_feedback')
                if parent is None or length > parent[0] or (length == parent[0] and dimension < parent[1]):
                    neighbors = (obstruction_neighbors(n,p,target,length,result.get('verified_tail',0),costs)
                                 if args.layered_classes else
                                 repair_neighbors(n,p,target,length,result.get('verified_tail',0),
                                                  dict(enumerate(costs))))
                    for neighbor in neighbors:
                        neighbor['parent_feedback'] = [length,dimension]
                        neighbor['feedback_basis'] = dict(evidence, outcome_path=str(trial/'outcome.json'))
                record['failure_length'] = length
                record['obstruction_evidence'] = evidence
        record['neighbors_generated'] = len(neighbors)
        for neighbor in reversed(neighbors):
            enqueue(neighbor, front=True, focus=True)
        trials.append(record)
        write(root/'trials.json', trials)
        write(root/'archive.json', archive)
        write(root/'selected.json', select_diverse(archive,8))
        checkpoint()
        emit('trial_complete', **{key:val for key,val in record.items() if key not in ('profile','path')},
             archived=len(archive))
        if focused >= args.focus_batch:
            service_due, coverage_due, focused = True, True, 0

    selected = select_diverse(archive,8)
    write(root/'selected.json', selected)
    write(root/'unfinished.json', dict(retained_support_tasks=list(retained),
        focused_proposals=list(focus_queue),
        queues={str(c):list(q) for c,q in queues.items()}, coverage_heap=coverage.heap,
        coverage_seen=sorted(coverage.seen), coverage_order=eligible,
        excluded_profiles=sorted(excluded), global_exhaustion=False))
    summary = dict(N=n,K=k,dp=floor,target=target,verified_profiles=len(archive),
        selected=len(selected),selected_names=[r['name'] for r in selected],
        trials=len(trials),seconds=past_elapsed+time.perf_counter()-started,
        stop_reason=('requested_archive_size' if len(archive) >= args.stop_after else
                     'wall_budget' if time.perf_counter() >= deadline-5 else 'trial_or_queue_budget'),
        globally_exhausted=False, proved_full_profile_exclusions=len(excluded),
        case_path=str(root.resolve()))
    write(root/'summary.json',summary)
    emit('case_complete',**{key:val for key,val in summary.items() if key != 'seconds'})
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-seconds',type=float,default=1800)
    parser.add_argument('--trials',type=int,default=256)
    parser.add_argument('--trial-seconds',type=float,default=600)
    parser.add_argument('--support-seconds',type=float,default=30)
    parser.add_argument('--outer-seconds',type=float,default=120)
    parser.add_argument('--work',type=int,default=30000000)
    parser.add_argument('--verification-work',type=int,default=300000000)
    parser.add_argument('--solver-nodes',type=int,default=50000)
    parser.add_argument('--joint-coefficients',type=int,default=0)
    parser.add_argument('--joint-max-length',type=int,default=128)
    parser.add_argument('--cold-run-root',type=Path)
    parser.add_argument('--class-depth',type=int,default=1)
    parser.add_argument('--layered-classes',action='store_true')
    parser.add_argument('--layered-seed-limit',type=int,default=5000)
    parser.add_argument('--seed-check-seconds',type=float,default=120)
    parser.add_argument('--dense-counts',action='store_true')
    parser.add_argument('--focus-batch',type=int,default=64)
    parser.add_argument('--stop-after',type=int,default=8)
    parser.add_argument('--anchor',action='store_true')
    parser.add_argument('--gap-first',action='store_true')
    parser.add_argument('--pattern-outer',action='store_true')
    parser.add_argument('--retry-incomplete',action='store_true')
    parser.add_argument('--snr',type=float,default=2.25)
    parser.add_argument('--cases',type=int,nargs='+',default=list(range(6)))
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reopen-completed',action='store_true')
    parser.add_argument('--class-order',choices=('reliability','structural'),default='reliability')
    parser.add_argument('--full-necessary',action='store_true')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--mode',default='structured',choices=('structured','complete'))
    args=parser.parse_args()
    if args.worker:
        from external_low_weight_spectrum import WorkLimitError
        try:
            worker(args)
        except WorkLimitError as error:
            write(args.output/'result.json', dict(status='enumeration_limit',
                  reason=str(error),full_profile_excluded=False,
                  worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        return
    args.output.mkdir(parents=True,exist_ok=args.resume)
    protocol_name = ('protocol_resume_'+str(time.time_ns())+'.json' if args.resume else 'protocol.json')
    write(args.output/protocol_name,dict(cases=[CASES[i] for i in args.cases],
        dp128_correction='12 is not a possible unprecoded polar distance; use 8 to improve to 16',
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        hash_schema='SHA256 of sorted-key compact JSON; profile={N,profile}; support={N,columns_hex}',
        filename_schema='profile_hash[:12] + _ + support_hash[:12]',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    summaries=[]
    for index in args.cases:
        n,k,floor,target=CASES[index]
        saved=args.output/f'n{n}_k{k}_dp{floor}_d{target}'/'summary.json'
        if args.resume and saved.exists() and not args.reopen_completed:
            summaries.append(read(saved))
            continue
        summaries.append(run_case(CASES[index],args))
        write(args.output/'summary.json',summaries)

if __name__ == '__main__':
    main()
