"""Deterministic conditional expression minima with exact distance protection."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace
from external_low_weight_spectrum import WorkLimitError
from manuscript_affine_experiment import optimize
from refine_protected_core_spectrum import measure, distance_pass, write
from row_neighborhood import RowCatalogue
from run_diverse_profile_suite import identities, read
from run_n64_scl_comparison import normalize_json

def run(args):
    raw = read(args.source)
    design = normalize_json(raw)
    n, profile = design['N'], tuple(design['profile'])
    if n > 256:
        raise ValueError('This exact catalogue adapter supports N <= 256')
    columns = list(design['columns'])
    floor = min(1 << i.bit_count() for i in profile)
    rows = tuple(j for j in range(n) if j not in profile and any(i < j for i in profile)
                 and (args.directions == 'all' or 1 << j.bit_count() >= floor))
    initial = measure(n, profile, columns, args.target, max(args.work, 300000000))
    if not distance_pass(initial['spectrum'], args.target):
        raise ValueError('The starting code does not meet the target')
    best = initial_count = initial['spectrum'].get(args.target, 0)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'protocol.json', dict(
        N=n, K=len(profile), source=str(args.source.resolve()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        target=args.target, rows=rows, initial_A_target=best,
        method='Complete supercode words, exact lower-weight rejection, conditional FWHT minimum',
        candidate_order='Increasing frozen index; restart after a strict improvement',
        tie_rule='First optimal observation image, zero kernel lift',
        seconds=args.seconds, max_work=args.work, max_rank=args.max_rank,
        global_optimality_claim=False))
    started = time.perf_counter()
    stats, improvements, records = Counter(), [], []
    last_progress = started

    def save(answer):
        ph, sh = identities(n, profile, columns)
        write(args.output/'code.json', dict(
            N=n, K=len(profile), profile=profile, columns=columns,
            profile_hash=ph, support_hash=sh, name=ph[:12]+'_'+sh[:12],
            dmin=args.target if best else None, distance_lower_bound=args.target,
            Admin=best, A_target=best, target=args.target, GA_proxy=raw.get('GA_proxy'),
            verification=answer, fixed_profile=True, global_optimality_claim=False))
        write(args.output/'improvements.json', improvements)

    save(initial)
    complete_sweep = False
    while time.perf_counter()-started < args.seconds and best:
        changed, exhausted = False, True
        for j in rows:
            if time.perf_counter()-started >= args.seconds:
                exhausted = False
                break
            event = dict(row=j, before_A_target=best)
            try:
                catalogue = RowCatalogue(SimpleNamespace(n=n, profile=profile), columns,
                                         j, cutoff=args.target, max_work=args.work)
                if catalogue.score(0) != (best, 0):
                    raise AssertionError('Supercode catalogue does not reproduce the incumbent')
                value, count, details = optimize([(u, [], w) for u, w in catalogue.words],
                                                catalogue.domain, j, args.target,
                                                max_rank=args.max_rank)
                column = sum(((value >> k) & 1) << i for k, i in enumerate(catalogue.domain))
                delta = column ^ columns[j]
                if catalogue.score(delta) != (count, 0) or count > best:
                    raise AssertionError('Conditional minimum or lower-weight protection disagrees')
                event.update(status='exact', A_target=count, catalogue_words=len(catalogue.words),
                             optimization=details)
                if count < best:
                    proposed = list(columns)
                    proposed[j] = column
                    answer = measure(n, profile, proposed, args.target, 300000000,
                                     policy='enumerate')
                    if (not distance_pass(answer['spectrum'], args.target)
                            or answer['spectrum'].get(args.target, 0) != count):
                        raise AssertionError('Independent complete count disagrees with the row minimum')
                    edges = [[j, i] for i in catalogue.domain if delta >> i & 1]
                    columns, best, changed = proposed, count, True
                    improvements.append(dict(edges=edges, A_target=best, row=j,
                                             seconds=time.perf_counter()-started))
                    save(answer)
                    print(f'Whole expression {j}: A_{args.target}={best}, edits={edges}', flush=True)
            except (RuntimeError, WorkLimitError) as exc:
                if not isinstance(exc, WorkLimitError) and str(exc) != 'residual_rank_limit':
                    raise
                event.update(status='unknown', reason=str(exc))
            stats[event['status']] += 1
            event['seconds'] = time.perf_counter()-started
            records.append(event)
            if changed or time.perf_counter()-last_progress >= 5:
                write(args.output/'progress.json', dict(
                    queries=len(records), A_target=best, seconds=event['seconds'],
                    statistics=dict(stats), improvements=improvements))
                last_progress = time.perf_counter()
            if changed:
                break
        if exhausted and not changed:
            complete_sweep = True
            break
    final = measure(n, profile, columns, args.target, 300000000, policy='enumerate')
    if not distance_pass(final['spectrum'], args.target) or final['spectrum'].get(args.target, 0) != best:
        raise AssertionError('Final complete spectrum disagrees with the improvement archive')
    save(final)
    write(args.output/'queries.json', records)
    summary = dict(initial_A_target=initial_count, final_A_target=best,
                   seconds=time.perf_counter()-started, rows=len(rows), queries=len(records),
                   statistics=dict(stats), improvements=improvements,
                   complete_row_sweep=complete_sweep,
                   all_queries_exact=stats['unknown'] == 0,
                   global_optimality_claim=False, code_path=str((args.output/'code.json').resolve()))
    write(args.output/'summary.json', summary)
    print(summary, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--work', type=int, default=3000000)
    parser.add_argument('--max-rank', type=int, default=22)
    parser.add_argument('--directions', choices=('floor', 'all'), default='floor')
    options = parser.parse_args()
    if min(options.target, options.seconds, options.work) <= 0 or not 0 <= options.max_rank <= 24:
        parser.error('Positive budgets and a residual rank limit in [0,24] are required')
    run(options)
