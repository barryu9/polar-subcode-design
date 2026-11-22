"""Combine independently completed prefix and suffix coefficient domains.

The prefix may change only rows preceding the first information coset at or
below the enumeration cutoff. Hence the complete truncated spectrum is exactly
that of the verified suffix input, irrespective of its earlier coefficients.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from design_scl_protected_prefix import check_spectral_invariance, protected_boundary
from extended_exact_spectrum import measure, WorkLimitError
from refine_protected_core_spectrum import write
from run_diverse_profile_suite import identities
from run_n64_scl_comparison import normalize_json
from select_designs_scl import noiseless_check

def compose(prefix, suffix, cutoff):
    if (prefix['N'], prefix['profile']) != (suffix['N'], suffix['profile']):
        raise ValueError('Prefix and suffix must share the same ordered profile')
    n, profile = suffix['N'], suffix['profile']
    boundary = protected_boundary(profile, cutoff, n)
    columns = prefix['columns'][:boundary]+suffix['columns'][boundary:]
    proof = check_spectral_invariance(n, profile, suffix['columns'], columns, cutoff)
    return columns, proof

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--suffix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff', type=int, default=24)
    parser.add_argument('--work', type=int, default=100000000)
    args = parser.parse_args()
    started = time.perf_counter()
    raw = json.loads(args.suffix.read_text(encoding='utf-8-sig'))
    suffix = normalize_json(raw)
    prefix = normalize_json(json.loads(args.prefix.read_text(encoding='utf-8-sig')))
    columns, proof = compose(prefix, suffix, args.cutoff)
    n, profile = suffix['N'], suffix['profile']
    baseline = measure(n, profile, suffix['columns'], args.cutoff, args.work)
    verification = dict(baseline=baseline, invariance=proof)
    try:
        final = measure(n, profile, columns, args.cutoff, args.work)
        if final['spectrum'] != baseline['spectrum']:
            raise AssertionError('Direct enumeration contradicts prefix invariance')
        verification['direct_complete_enumeration'] = final
    except WorkLimitError:
        verification['direct_complete_enumeration'] = dict(
            status='work_limit', exact_invariance_proof_still_valid=True)
    positive = [w for w, a in baseline['spectrum'].items() if w > 0 and a]
    dmin = min(positive) if positive else None
    ph, sh = identities(n, profile, columns)
    code = dict(N=n, K=len(profile), profile=profile, columns=columns,
                profile_hash=ph, support_hash=sh, name=ph[:12]+'_'+sh[:12],
                dmin=dmin, distance_lower_bound=dmin or args.cutoff+1,
                Admin=baseline['spectrum'].get(dmin) if dmin else None,
                spectrum=baseline['spectrum'], verification=verification,
                prefix_source=str(args.prefix.resolve()), suffix_source=str(args.suffix.resolve()),
                prefix_sha256=hashlib.sha256(args.prefix.read_bytes()).hexdigest(),
                suffix_sha256=hashlib.sha256(args.suffix.read_bytes()).hexdigest(),
                reference_support_read=False, spectrum_preserving_prefix=True)
    noiseless_check(code, 32, 'cpp')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'code.json', code)
    write(args.output/'summary.json', dict(seconds=time.perf_counter()-started,
          name=code['name'], dmin=dmin, Admin=code['Admin'], verification=verification,
          nonzero_rows=sum(bool(c) for c in columns), coefficients=sum(c.bit_count() for c in columns)))
    print(json.dumps(dict(name=code['name'], dmin=dmin, Admin=code['Admin'],
                          seconds=time.perf_counter()-started)), flush=True)

if __name__ == '__main__':
    main()
