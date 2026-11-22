"""Verify a GA-priority protected profile exchange against every forbidden weight."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from extended_exact_spectrum import WorkLimitError
from feasible_design_bank import generator_rows
from leading_suffix_spectrum import leading_suffix_measure
from polar_code.rp_funcs import rp_GA
from refine_recursive_suffix_joint_spectrum import protected_exchange
from run_diverse_profile_suite import identities, write
from run_n64_scl_comparison import normalize_json
from select_designs_scl import noiseless_check

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remove', type=int, nargs='+', required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--snr', type=float, default=2.25)
    parser.add_argument('--work', type=int, default=200_000_000)
    args = parser.parse_args()
    start = perf_counter()
    raw = json.loads(args.source.read_text(encoding='utf-8-sig'))
    code = normalize_json(raw)
    n, k, profile, columns = code['N'], code['K'], code['profile'], code['columns']
    costs = [-math.log1p(-float(p)) for p in rp_GA(n, k, args.snr).pe]
    cutoff = args.target-1
    boundary = min((i for i in profile if 1 << i.bit_count() <= cutoff), default=n)
    eligible = [i for i in range(boundary) if i not in profile and 1 << i.bit_count() > cutoff]
    added = sorted(eligible, key=lambda i: (costs[i], i))[:len(args.remove)]
    new_profile, new_columns, relation = protected_exchange(
        n, profile, columns, args.remove, added, cutoff)
    args.output.mkdir(parents=True, exist_ok=False)
    proxy = -math.expm1(-sum(costs[i] for i in new_profile))
    protocol = dict(source=str(args.source.resolve()),
                    source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    N=n, K=k, target=args.target, GA_proxy=proxy,
                    GA_design_EbN0_dB=args.snr, GA_proxy_definition='1 - product(1 - pe[i])',
                    removed=args.remove, added=added, protected_relation=relation,
                    max_work=args.work, Admin_objective=False,
                    addition_selection='Lowest GA log-cost eligible early high-row-weight inputs',
                    global_optimality_claim=False, reference_support_used=False)
    write(args.output/'protocol.json', protocol)
    print(json.dumps(dict(stage='start', **protocol)), flush=True)
    try:
        proof = leading_suffix_measure(n, new_profile, new_columns, cutoff, args.work)
    except WorkLimitError as exc:
        summary = dict(protocol, status='unknown_work_limit', error=str(exc),
                       distance_target_verified=False, seconds=perf_counter()-start)
        write(args.output/'summary.json', summary)
        print(json.dumps(summary), flush=True)
        return
    passed = not any(a for w, a in proof['spectrum'].items() if 0 < w < args.target)
    summary = dict(protocol, status='verified' if passed else 'distance_not_met',
                   distance_target_verified=passed, verification=proof,
                   seconds=perf_counter()-start)
    if passed:
        rows = generator_rows(n, new_profile, new_columns)
        upper = min(row.bit_count() for row in rows)
        witness = None
        for a, row in enumerate(rows):
            if row.bit_count() == args.target:
                witness = dict(information_positions=[new_profile[a]], codeword_hex=hex(row))
                break
            hit = next((b for b in range(a+1, len(rows))
                        if (row ^ rows[b]).bit_count() == args.target), None)
            if hit is not None:
                witness = dict(information_positions=[new_profile[a], new_profile[hit]],
                               codeword_hex=hex(row ^ rows[hit]))
                break
        ph, sh = identities(n, new_profile, new_columns)
        new = dict(N=n, K=k, profile=new_profile, columns=new_columns,
                   name=ph[:12]+'_'+sh[:12], profile_hash=ph, support_hash=sh,
                   dp=min(1 << i.bit_count() for i in new_profile),
                   GA_proxy=proxy, GA_design_EbN0_dB=args.snr,
                   GA_proxy_definition='1 - product(1 - pe[i])',
                   target=args.target, distance_lower_bound=args.target,
                   dmin=args.target if witness else None,
                   distance_upper_bound=args.target if witness else upper,
                   matching_witness=witness, verification=proof,
                   profile_exchange=relation, FER_verified=False,
                   global_optimality_claim=False, source=str(args.source.resolve()))
        noiseless_check(new, 32, 'cpp')
        write(args.output/'code.json', new)
        write(args.output/'profile.json', dict(N=n, K=k, profile=new_profile, dp=new['dp']))
        summary.update(dmin=new['dmin'], distance_lower_bound=args.target,
                       distance_upper_bound=new['distance_upper_bound'], name=new['name'],
                       noiseless_L32_pass=True, code_path=str((args.output/'code.json').resolve()),
                       seconds=perf_counter()-start)
    write(args.output/'summary.json', summary)
    print(json.dumps(dict(stage='complete', **summary)), flush=True)

if __name__ == '__main__':
    main()
