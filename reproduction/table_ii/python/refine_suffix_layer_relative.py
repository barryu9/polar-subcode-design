"""Deterministic coefficient descent with exact relative full-code validation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from exact_relative_spectrum import count_recursive
from external_low_weight_spectrum import WorkLimitError
from refine_recursive_suffix_spectrum import suffix_design
from row_neighborhood import RowCatalogue
from run_diverse_profile_suite import identities, read, write
from run_n64_scl_comparison import normalize_json
from select_designs_scl import noiseless_check

def one_bit_descent(catalogue, steps):
    delta = 0
    value, low = catalogue.score(delta)
    if low:
        raise ValueError('Incumbent constituent violates the protected distance')
    evaluated = 1
    for _ in range(steps):
        winner = None
        for i in catalogue.domain:
            candidate = delta ^ (1 << i)
            score, below = catalogue.score(candidate)
            evaluated += 1
            if not below and score < value and (winner is None or (score, i) < winner[:2]):
                winner = (score, i, candidate)
        if winner is None:
            break
        value, _, delta = winner
    return delta, value, evaluated

def run(args):
    raw = read(args.source)
    code = normalize_json(raw)
    n, p, c = code['N'], tuple(code['profile']), list(code['columns'])
    length, start = args.suffix_length, n-args.suffix_length
    if args.target % (n//length):
        raise ValueError('Target must be divisible by the suffix repetition factor')
    target = args.target//(n//length)
    local_p, local_c = suffix_design(n, p, c, length)
    if any(j in local_p for j in args.rows):
        raise ValueError('Edited rows must be frozen')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write(args.output/'protocol.json', dict(source=str(args.source.resolve()),
          source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
          N=n, K=len(p), target=args.target, suffix_length=length,
          local_target=target, rows=args.rows, steps=args.steps, seconds=args.seconds,
          work=args.work, fixed_profile=True, random_sampling=False,
          acceptance='Complete full-code spectrum retains distance and reduces the target layer',
          global_optimality_claim=False))
    best_spectrum = count_recursive(p, c, args.target, args.output/'initial_full', args.work)
    local_spectrum = count_recursive(local_p, local_c, target,
                                     args.output/'initial_suffix', args.work)
    if any(w and w < args.target and a for w, a in best_spectrum.items()):
        raise ValueError('Initial full code violates the target distance')
    initial, best = best_spectrum.get(args.target, 0), best_spectrum.get(args.target, 0)
    local_best = local_spectrum.get(target, 0)
    deadline = time.perf_counter()+args.seconds
    events, statistics = [], Counter()
    design = SimpleNamespace(n=length, profile=local_p)

    def save():
        ph, sh = identities(n, p, c)
        write(args.output/'code.json', dict(N=n, K=len(p), profile=p, columns=c,
              name=ph[:12]+'_'+sh[:12], profile_hash=ph, support_hash=sh,
              dmin=args.target if best else None, distance_lower_bound=args.target,
              Admin=best if best else None, A_target=best,
              complete_spectrum=best_spectrum, exact=True, GA_proxy=raw.get('GA_proxy'),
              fixed_profile=True, FER_verified=False, global_optimality_claim=False))

    save()
    with (args.output/'events.jsonl').open('w', encoding='utf-8') as stream:
        for row in args.rows:
            if time.perf_counter() >= deadline:
                break
            tick = time.perf_counter()
            event = dict(local_row=row, before_A=best, before_suffix_A=local_best)
            try:
                catalogue = RowCatalogue(design, local_c, row, target, args.work)
                delta, value, evaluated = one_bit_descent(catalogue, args.steps)
                event.update(catalogue_size=len(catalogue.words), evaluated=evaluated,
                             delta=delta, suffix_A=value)
                if value >= local_best:
                    event['status'] = 'no_improvement'
                else:
                    proposed = c.copy()
                    proposed[start+row] ^= delta << start
                    whole = count_recursive(p, proposed, args.target,
                                             args.output/f'row_{row:03d}', args.work)
                    count = whole.get(args.target, 0)
                    event.update(full_spectrum=whole, A_target=count)
                    if count < best and not any(w and w < args.target and a
                                                for w, a in whole.items()):
                        c, local_c = proposed, catalogue.child(delta)
                        best, local_best, best_spectrum = count, value, whole
                        event['status'] = 'accepted'
                        save()
                    else:
                        event['status'] = 'full_code_rejection'
                del catalogue
            except WorkLimitError as error:
                event.update(status='unknown', reason=str(error))
            event['seconds'] = time.perf_counter()-tick
            events.append(event)
            statistics[event['status']] += 1
            stream.write(json.dumps(event)+'\n')
            stream.flush()
            write(args.output/'progress.json', dict(A_target=best, suffix_A=local_best,
                  events=events, seconds=time.perf_counter()-started))
            print(json.dumps(dict(stage='row', **event)), flush=True)
    noiseless_check(dict(N=n, K=len(p), profile=p, columns=c), 32, 'cpp')
    save()
    summary = dict(initial_A_target=initial, final_A_target=best, final_suffix_A=local_best,
                   spectrum=best_spectrum, events=events, statistics=dict(statistics),
                   seconds=time.perf_counter()-started, fixed_profile=True,
                   global_optimality_claim=False, noiseless_L32_pass=True)
    write(args.output/'summary.json', summary)
    print(json.dumps(dict(stage='complete', **summary)), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--suffix-length', type=int, default=256)
    parser.add_argument('--rows', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--work', type=int, default=600000000)
    run(parser.parse_args())
