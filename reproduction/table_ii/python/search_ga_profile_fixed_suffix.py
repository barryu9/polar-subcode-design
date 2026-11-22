"""GA-priority profile exchanges and exact completion with an independent suffix.

Only an existing independently constructed code, its optimized suffix, and GA
scores are inputs. No reference profile or reference support is read. The suffix
fixes a spectrum floor; shared cross coefficients protect newly selected inputs.
"""

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from complete_reference_profile_fixed_suffix import run as complete_fixed_suffix
from feasible_design_bank import validate_design
from polar_code.rp_funcs import rp_GA
from refine_recursive_suffix_spectrum import suffix_design
from run_diverse_profile_suite import identities, read, write
from run_n64_scl_comparison import normalize_json

def rank_profiles(original, suffix_profile, pe, pool_size, count):
    """Rank a declared exchange family, not all profiles of the given dimension."""
    n, k = original['N'], original['K']
    boundary = 3*n//4
    kept = {i for i in original['profile'] if i < boundary}
    tail = {boundary+i for i in suffix_profile}
    needed = k-len(kept)-len(tail)
    if needed <= 0:
        raise ValueError('The optimized suffix must free positive information capacity')
    costs = [-math.log1p(-float(p)) for p in pe]
    allowed = [i for i in range(boundary) if i not in kept
               and (1 << i.bit_count()) >= (32 if i < n//2 else 16)]
    pool = sorted(allowed, key=lambda i: (costs[i], i))[:pool_size]
    if needed > len(pool):
        raise ValueError('Insufficient GA-ranked replacement positions')
    records = []
    for added in combinations(pool, needed):
        profile = tuple(sorted(kept | tail | set(added)))
        cost = sum(costs[i] for i in profile)
        records.append(dict(profile=profile, added=tuple(sorted(added)),
                            removed=tuple(sorted(set(original['profile'])-set(profile))),
                            GA_proxy=-math.expm1(-cost), cost=cost))
    return sorted(records, key=lambda r: (r['cost'], r['added']))[:count], pool

def prepare_candidate(source, record):
    """Retain fixed expressions but clear every coefficient of a new input."""
    n = source['N']
    old, new = set(source['profile']), set(record['profile'])
    uncontrolled = (old-new) | set(record['added'])
    mask = sum(1 << i for i in uncontrolled)
    columns = [c & ~mask for c in source['columns']]
    for i in (old-new) | (new-old):
        columns[i] = 0
    validate_design(n, record['profile'], columns)
    if suffix_design(n, record['profile'], columns, n//4) != suffix_design(
            n, source['profile'], source['columns'], n//4):
        raise AssertionError('Candidate preparation changed the independent suffix')
    return dict(N=n, K=len(new), profile=record['profile'], columns=columns,
                GA_proxy=record['GA_proxy'], reference_profile_used=False,
                reference_support_used=False)

def run(args):
    original = normalize_json(read(args.original))
    source = normalize_json(read(args.suffix_source))
    if original['N'] != 1024 or original['K'] != 512:
        raise ValueError('This experiment currently expects N=1024, K=512')
    if (source['N'], source['K']) != (original['N'], original['K']):
        raise ValueError('Source codes have different parameters')
    n = original['N']
    suffix_p, _ = suffix_design(n, source['profile'], source['columns'], n//4)
    ga = rp_GA(n, original['K'], args.design_snr)
    records, pool = rank_profiles(original, suffix_p, ga.pe, args.pool_size, args.candidates)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write(args.output/'protocol.json', dict(
        original=str(args.original.resolve()),
        original_sha256=hashlib.sha256(args.original.read_bytes()).hexdigest(),
        suffix_source=str(args.suffix_source.resolve()),
        suffix_source_sha256=hashlib.sha256(args.suffix_source.read_bytes()).hexdigest(),
        reference_profile_used=False, reference_support_used=False,
        GA_design_EbN0_dB=args.design_snr, GA_pool=pool,
        rank_rule='Increasing sum(-log(1-pe)); lexicographic tie break',
        fixed_suffix_length=n//4, fixed_suffix_dimension=len(suffix_p),
        candidate_limit=args.candidates, completion_seconds=args.seconds,
        A32_goal=args.a32_goal, work=args.work,
        scope='GA-ranked balanced exchanges with the independent optimized suffix fixed',
        random_sampling=False, global_optimality_claim=False))
    archive = []
    for ordinal, record in enumerate(records):
        tick = time.perf_counter()
        candidate = prepare_candidate(source, record)
        ph, _ = identities(n, candidate['profile'], candidate['columns'])
        folder = args.output/f'{ordinal:02d}_{ph[:12]}'
        folder.mkdir()
        write(folder/'preparation.json', candidate)
        entry = dict(ordinal=ordinal, profile_hash=ph, added=record['added'],
                     removed=record['removed'], GA_proxy=record['GA_proxy'], status='pending')
        archive.append(entry)
        write(args.output/'candidate_archive.json', archive)
        print(json.dumps(dict(stage='candidate', **entry)), flush=True)
        new_middle = [i for i in record['added'] if i >= n//2]
        try:
            complete_fixed_suffix(SimpleNamespace(
                source=folder/'preparation.json', output=folder/'completion',
                target=32, new_information=new_middle, seconds=args.seconds,
                work=args.work, design_snr=args.design_snr))
            result = read(folder/'completion/summary.json')
            entry.update(result)
        except (ValueError, RuntimeError) as exc:
            entry.update(status='unresolved', reason=str(exc),
                         global_infeasibility_claim=False)
        entry['candidate_seconds'] = time.perf_counter()-tick
        write(args.output/'candidate_archive.json', archive)
        write(args.output/'summary.json', dict(records=archive, complete=False,
              seconds=time.perf_counter()-started, reference_profile_used=False))
        if entry['status'] == 'verified' and entry['A_target'] <= args.a32_goal:
            break
    good = [r for r in archive if r['status'] == 'verified' and r['A_target'] <= args.a32_goal]
    best = min(good, key=lambda r: (r['GA_proxy'], r['A_target'])) if good else None
    summary = dict(records=archive, best=best, complete=True,
                   seconds=time.perf_counter()-started, reference_profile_used=False,
                   reference_support_used=False, global_optimality_claim=False,
                   FER_verified=False)
    write(args.output/'summary.json', summary)
    print(json.dumps(dict(stage='finished', **summary)), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--suffix-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design-snr', type=float, default=2.25)
    parser.add_argument('--pool-size', type=int, default=8)
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--a32-goal', type=int, default=105000)
    parser.add_argument('--work', type=int, default=600000000)
    run(parser.parse_args())
