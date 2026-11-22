"""Exact row/block spectrum refinement with optional protected profile exchanges.

A new early information coordinate with polar row weight above the cutoff
cannot occur in a codeword below that cutoff. Freezing a late information
coordinate to zero therefore restricts the low-weight catalogue, provided all
removed sources are substituted by zero. This is a distance guarantee, not a
guarantee for prefix-list survival or the FER of the exchanged profile.
"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from audit_rm_core_rank_spectrum import rank_audit
from block_neighborhood import BlockCatalogue, optimize_pair
from extended_exact_spectrum import measure, WorkLimitError
from feasible_design_bank import validate_design
from polar_code.rp_funcs import rp_GA
from refine_protected_core_spectrum import distance_pass, write
from refine_recursive_suffix_spectrum import suffix_design
from row_neighborhood import RowCatalogue, optimize_row
from run_diverse_profile_suite import identities, read
from run_n64_scl_comparison import normalize_json
from select_designs_scl import noiseless_check

def protected_exchange(n, profile, columns, removed, added, cutoff):
    """Restrict the old low-weight codewords without importing another support."""
    profile = validate_design(n, profile, columns)
    old, removed, added = set(profile), tuple(removed), tuple(added)
    if (not removed or len(removed) != len(added)
            or len(set(removed)) != len(removed) or len(set(added)) != len(added)
            or not set(removed) <= old or set(added) & old
            or any(not 0 <= i < n for i in added)):
        raise ValueError('Expected distinct, balanced information exchanges')
    boundary = min((i for i in profile if 1 << i.bit_count() <= cutoff), default=n)
    if any(i < boundary for i in removed) or any(
            i >= boundary or 1 << i.bit_count() <= cutoff for i in added):
        raise ValueError('The exchange does not satisfy the early high-row-weight guard')
    removed_mask = sum(1 << i for i in removed)
    result = [c & ~removed_mask for c in columns]
    for i in added+removed:
        result[i] = 0
    new_profile = tuple(sorted((old-set(removed)) | set(added)))
    validate_design(n, new_profile, result)
    proof = dict(cutoff=cutoff, original_low_weight_boundary=boundary,
                 removed=list(removed), added=list(added),
                 removed_information_constrained_to_zero=True,
                 added_row_weights=[1 << i.bit_count() for i in added],
                 low_weight_relation='New codewords through the cutoff are exactly the old codewords with removed information zero.',
                 argument='A codeword using any information before the boundary has a leading polar coset of row weight above the cutoff. With these inputs zero, the new expressions coincide with the old expressions after substituting zero for removed information.',
                 prefix_loss_or_FER_guarantee=False)
    return new_profile, result, proof

def choose_exchange(n, profile, columns, cutoff, suffix_length, count, costs):
    start = n-suffix_length
    candidates = [i for i in profile if i >= start+suffix_length//4
                  and (i-start).bit_count() == 2]
    boundary = min((i for i in profile if 1 << i.bit_count() <= cutoff), default=n)
    additions = [i for i in range(boundary) if i not in profile and 1 << i.bit_count() > cutoff]
    if count > min(len(candidates), len(additions)):
        raise ValueError('Insufficient exchange coordinates in the protected family')
    removed = sorted(candidates, key=lambda i: (-costs[i], i))[:count]
    added = sorted(additions, key=lambda i: (costs[i], i))[:count]
    return protected_exchange(n, profile, columns, removed, added, cutoff)

def run(args):
    started = perf_counter()
    raw = read(args.source)
    code = normalize_json(raw)
    n, profile, columns = code['N'], tuple(code['profile']), list(code['columns'])
    length, target = args.suffix_length, args.target
    if (not 16 <= length <= min(n, 256) or length & (length-1)
            or target % (n//length)):
        raise ValueError('Require a supported suffix and an integral local target')
    local_target, start = target//(n//length), n-length
    ga = rp_GA(n, len(profile), args.design_snr)
    costs = [-math.log1p(-float(p)) for p in ga.pe]
    original_proxy = -math.expm1(-sum(costs[i] for i in profile))
    exchange = None
    if args.exchange_count:
        profile, columns, exchange = choose_exchange(
            n, profile, columns, target, length, args.exchange_count, costs)
    proxy = -math.expm1(-sum(costs[i] for i in profile))
    args.output.mkdir(parents=True, exist_ok=False)
    initial = measure(n, profile, columns, target, args.work, policy='enumerate')
    if not distance_pass(initial['spectrum'], target):
        raise ValueError('The initial construction does not meet the target')
    local_profile, local_columns = suffix_design(n, profile, columns, length)
    initial_local = measure(length, local_profile, local_columns, local_target,
                            args.work, policy='enumerate')
    best = initial['spectrum'].get(target, 0)
    local_best = initial_local['spectrum'].get(local_target, 0)
    floor = min(1 << i.bit_count() for i in local_profile)
    rows = tuple(j for j in range(length//4, length) if j not in local_profile
                 and j.bit_count() == 2 and 1 << j.bit_count() >= floor)
    pairs = sorted(combinations(rows, 2), key=lambda pair: (
        not all(local_columns[j] for j in pair), pair))
    jobs = [(j,) for j in rows] + list(pairs)
    write(args.output/'protocol.json', dict(
        source=str(args.source.resolve()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        N=n, K=len(profile), target=target, suffix_length=length,
        local_target=local_target, rows=rows, jobs=jobs, exchange=exchange,
        original_GA_proxy=original_proxy, GA_proxy=proxy,
        GA_proxy_definition='1 - product(1 - pe[i]) at the declared design Eb/N0',
        GA_design_EbN0_dB=args.design_snr, max_work=args.work,
        seconds=args.seconds, max_queries=args.queries,
        method='One complete relaxed row/block catalogue; deterministic one/two-coefficient descent from its unchanged incumbent',
        acceptance='Exact suffix decrease, complete full-code distance protection and strict full A_target decrease',
        core_information_sources_allowed=True,
        random_start_sampling=False, reference_support_read=False,
        global_optimality_claim=False, initial_spectrum=initial,
        initial_suffix_spectrum=initial_local))
    initial_best, initial_local_best = best, local_best
    local_design = SimpleNamespace(n=length, profile=local_profile,
                                   rows=tuple(j for j in range(length) if j not in local_profile))
    improvements, records, statistics = [], [], Counter()
    prefix_mask = (1 << start)-1
    current_proof = initial

    def save():
        ph, sh = identities(n, profile, columns)
        write(args.output/'code.json', dict(
            N=n, K=len(profile), profile=profile, columns=columns,
            name=ph[:12]+'_'+sh[:12], profile_hash=ph, support_hash=sh,
            dmin=target if best else None, distance_lower_bound=target,
            A_target=best, Admin=best if best else None, target=target,
            GA_proxy=proxy, GA_design_EbN0_dB=args.design_snr,
            verification=current_proof, profile_exchange=exchange,
            prefix_rows_preserved_after_initial_exchange=True,
            FER_verified=False, global_optimality_claim=False))
        write(args.output/'improvements.json', improvements)

    save()
    initial_phase_seconds = perf_counter()-started
    deadline = perf_counter()+args.seconds
    completed_sweep = False
    with (args.output/'events.jsonl').open('w', encoding='utf-8') as stream:
        while len(records) < args.queries and perf_counter() < deadline:
            changed, exhausted = False, True
            for block in jobs:
                if len(records) >= args.queries or perf_counter() >= deadline:
                    exhausted = False
                    break
                tick = perf_counter()
                record = dict(block=block, before_suffix_A=local_best, before_A_target=best)
                try:
                    if len(block) == 1:
                        catalogue = RowCatalogue(local_design, local_columns, block[0],
                                                 local_target, args.work)
                        answer = optimize_row(catalogue, seed=0, starts=1, steps=args.steps)
                        deltas = [answer['delta']]
                        proposed_local = catalogue.child(deltas[0])
                    else:
                        catalogue = BlockCatalogue(local_design, local_columns, block,
                                                   local_target, args.work,
                                                   causal_leaders_only=True)
                        answer = optimize_pair(catalogue, seed=0, starts=1, steps=args.steps)
                        deltas = answer['deltas']
                        proposed_local = catalogue.child(deltas)
                    count = answer['A16']
                    record.update(catalogue_words=len(catalogue.words),
                                  evaluated_moves=answer['evaluated'], suffix_A=count,
                                  changed_coefficients=sum(delta.bit_count() for delta in deltas))
                    if count >= local_best:
                        record['status'] = 'no_component_improvement'
                    else:
                        local_check = measure(length, local_profile, proposed_local,
                                              local_target, args.work, policy='enumerate')
                        if (not distance_pass(local_check['spectrum'], local_target)
                                or local_check['spectrum'].get(local_target, 0) != count):
                            raise AssertionError('The relaxed catalogue failed independent verification')
                        proposed = columns.copy()
                        for j in block:
                            proposed[start+j] = (columns[start+j] & prefix_mask) | (proposed_local[j] << start)
                        whole = measure(n, profile, proposed, target, args.work, policy='enumerate')
                        value = whole['spectrum'].get(target, 0)
                        record.update(full_spectrum=whole['spectrum'], A_target=value)
                        if distance_pass(whole['spectrum'], target) and value < best:
                            edges = [[start+j, start+i] for j, delta in zip(block, deltas)
                                     for i in local_profile if delta >> i & 1]
                            columns, local_columns = proposed, proposed_local
                            best, local_best, current_proof = value, count, whole
                            changed = True
                            record['status'] = 'accepted'
                            improvements.append(dict(edges=edges, suffix_A=count,
                                                     A_target=best, seconds=perf_counter()-started))
                            save()
                        else:
                            record['status'] = 'full_code_rejection'
                except WorkLimitError as exc:
                    record.update(status='unknown_work_limit', reason=str(exc))
                record.update(query_seconds=perf_counter()-tick, seconds=perf_counter()-started)
                records.append(record)
                statistics[record['status']] += 1
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                progress = dict(queries=len(records), A_target=best, suffix_A=local_best,
                                GA_proxy=proxy, statistics=dict(statistics),
                                seconds=record['seconds'])
                write(args.output/'progress.json', progress)
                print(json.dumps(dict(stage='query', **progress, block=block,
                                      result=record['status'])), flush=True)
                if changed:
                    break
            if exhausted and not changed:
                completed_sweep = True
                break
    noiseless_check(dict(N=n, K=len(profile), profile=profile, columns=columns), 32, 'cpp')
    save()
    structure = rank_audit(length, local_profile, local_columns)
    write(args.output/'final_structure.json', structure)
    summary = dict(initial_A_target=initial_best, final_A_target=best,
                   initial_suffix_A=initial_local_best, final_suffix_A=local_best,
                   original_GA_proxy=original_proxy, GA_proxy=proxy,
                   relative_GA_proxy_change=proxy/original_proxy-1,
                   exchange=exchange, improvements=improvements,
                   queries=len(records), statistics=dict(statistics),
                   completed_declared_local_sweep=completed_sweep,
                   initial_phase_seconds=initial_phase_seconds,
                   seconds=perf_counter()-started,
                   final_spectrum=current_proof, noiseless_L32_pass=True,
                   retained_RM_core_dimension=structure['intersection_with_natural_RM_core_dimension'],
                   fixed_profile=not args.exchange_count, FER_verified=False,
                   global_optimality_claim=False,
                   code_path=str((args.output/'code.json').resolve()))
    write(args.output/'summary.json', summary)
    print(json.dumps(dict(stage='complete', **summary)), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=int, default=24)
    parser.add_argument('--suffix-length', type=int, default=256)
    parser.add_argument('--design-snr', type=float, default=2.25)
    parser.add_argument('--exchange-count', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--queries', type=int, default=120)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--work', type=int, default=100_000_000)
    args = parser.parse_args()
    if (min(args.seconds, args.queries, args.steps, args.work, args.target) <= 0
            or args.exchange_count < 0 or not math.isfinite(args.design_snr)):
        parser.error('Positive limits, nonnegative exchange count and finite SNR required')
    run(args)
