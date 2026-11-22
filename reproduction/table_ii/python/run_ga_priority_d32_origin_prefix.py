"""Reapply the original sampled prefix rule to a verified GA-priority d32 code.

The input's stored complete spectrum is the starting certificate, bound to its
file hash. A fresh recursive complete count audits the final code. No random
coefficient filling, reference-code coefficients, or FER-based selection is used.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

os.environ.setdefault('OMP_NUM_THREADS', '1')

os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from design_scl_protected_prefix import check_spectral_invariance, design_prefix, generator_distance_witness, protected_boundary
from exact_relative_spectrum import count_recursive
from feasible_design_bank import validate_design
from polar_code import _scl_step_decoder_cpp
from polar_code.pc_funcs import pc_init
from polar_code.rp_funcs import rp_GA
from polar_code.SCL_full_decoder import decode_full_path_batch, simulate_awgn_llr_batch
from refine_recursive_suffix_spectrum import suffix_design
from run_diverse_profile_suite import identities, read, write
from run_n64_scl_comparison import normalize_json
from select_designs_scl import construction, noiseless_check

ROOT = Path(__file__).resolve().parents[1]



def run(args):
    raw, origin = read(args.source), read(args.origin_protocol)
    before = normalize_json(raw)
    n, profile, columns = before['N'], tuple(before['profile']), before['columns']
    validate_design(n, profile, columns)
    baseline = {int(w): int(a) for w, a in raw.get('complete_spectrum', {}).items()}
    if ((n, before['K']) != (1024, 512) or not raw.get('spectrum_exact')
            or baseline.get(0) != 1 or baseline.get(32, 0) <= 0
            or any(0 < w < 32 and a for w, a in baseline.items())):
        raise ValueError('Input must include a complete verified spectrum through weight 32')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: origin[key] for key in (
        'cutoff', 'target', 'seed', 'design_frames', 'design_EbN0_dB',
        'L', 'row_floor', 'max_flips', 'coefficient_rule', 'objective')}
    if (config['target'], config['cutoff']) != (32, 31):
        raise ValueError('The original d32 rule must protect all weights below 32')
    config.update(source=str(args.source.resolve()),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  origin_protocol=str(args.origin_protocol.resolve()),
                  origin_protocol_sha256=hashlib.sha256(args.origin_protocol.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_decoder_sha256=hashlib.sha256(Path(_scl_step_decoder_cpp.__file__).read_bytes()).hexdigest(),
                  N=n, K=len(profile), source_name=raw['name'],
                  baseline_complete_spectrum=baseline,
                  baseline_validation='Existing complete spectrum stored with the hashed input',
                  final_validation=(
                      'No fresh count after prefix design. Verify by exact identity with '
                      'the independently counted code, or by a separate complete count.'
                      if args.defer_spectrum_verification
                      else 'Fresh complete recursive Plotkin count through weight 32'),
                  boundary=protected_boundary(profile, 31, n), work=args.work,
                  reference_support_read=False, random_coefficient_filling=False,
                  calibration_is_not_independent_FER=True)
    write(args.output/'protocol.json', config)
    print(json.dumps(dict(stage='start', source=raw['name'], boundary=config['boundary'])), flush=True)
    started = time.perf_counter()
    llr = simulate_awgn_llr_batch(n, len(profile), config['design_EbN0_dB'],
                                 range(config['seed'], config['seed']+config['design_frames']))
    params = pc_init(n, len(profile), 0, np.array(((1, 0), (1, 1)), dtype=int))
    actions, supports = construction(n, profile, columns)
    tick = time.perf_counter()
    _, old_loss, _, old_u = decode_full_path_batch(
        params, actions, supports, llr, config['L'], early_stop=False,
        terminal_margin=0., max_support_size=None, backend='cpp', return_decoded_u=True)
    before_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    with (args.output/'rows.jsonl').open('w', encoding='utf-8') as stream:
        def event(record):
            record = dict(record, seconds=time.perf_counter()-started)
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            if record['position'] % 32 < 2:
                print(json.dumps(record), flush=True)
        result, report = design_prefix(
            n, profile, columns, config['cutoff'], llr, config['L'],
            config['row_floor'], config['max_flips'], event)
    design_seconds = time.perf_counter()-tick
    unchanged_suffix = suffix_design(n, profile, columns, 256) == suffix_design(n, profile, result, 256)
    if not unchanged_suffix:
        raise AssertionError('The spectrum-dominant suffix changed')
    ph, sh = identities(n, profile, result)
    upper, witness = generator_distance_witness(n, profile, result, 32)
    try:
        layer_proof = check_spectral_invariance(n, profile, columns, result, 32)
    except ValueError:
        layer_proof = None
    ga_snr = float(raw.get('GA_design_EbN0_dB', 2.25))
    ga = rp_GA(n, len(profile), ga_snr)
    proxy = -math.expm1(sum(math.log1p(-float(ga.pe[i])) for i in profile))
    code = dict(N=n, K=len(profile), profile=profile, columns=result,
                name=ph[:12]+'_'+sh[:12], profile_hash=ph, support_hash=sh,
                dp=min(1 << i.bit_count() for i in profile),
                distance_lower_bound=32, distance_upper_bound=upper,
                matching_witness=witness, GA_proxy=proxy, GA_design_EbN0_dB=ga_snr,
                source=str(args.source.resolve()), source_sha256=config['source_sha256'],
                prefix_protection_completed=True, FER_verified=False,
                spectrum_invariance=report['spectrum_invariance'],
                target_layer_preservation_proved=layer_proof is not None,
                method='Original spectrum-neutral prefix-list completion',
                spectrum_exact=False, verification_status='pending_complete_count')
    write(args.output/'code.json', code)
    report.update(configuration=config, source_name=raw['name'], name=code['name'],
                  before_design_errors=int(np.any(old_u != 0, axis=1).sum()),
                  before_design_pruning_errors=int(np.sum(old_loss >= 0)),
                  calibration_seconds=before_seconds, prefix_design_seconds=design_seconds,
                  profile_unchanged=True, fixed_suffix_unchanged=unchanged_suffix,
                  independent_FER_test_performed=False)
    write(args.output/'design_summary.json', report)
    print(json.dumps(dict(stage='complete_prefix', name=code['name'],
                         changed_rows=report['spectrum_invariance']['changed_rows'],
                         before_errors=report['before_design_errors'], after_errors=report['design_errors'])), flush=True)
    if args.defer_spectrum_verification:
        report.update(seconds=time.perf_counter()-started,
                      code_path=str((args.output/'code.json').resolve()),
                      verification_status='pending_complete_count')
        write(args.output/'summary.json', report)
        return
    tick = time.perf_counter()
    spectrum = count_recursive(profile, result, 32, args.output/'full_spectrum', args.work)
    if any(0 < w < 32 and a for w, a in spectrum.items()) or not spectrum.get(32):
        raise AssertionError('Final complete spectrum does not have minimum distance 32')
    verification_seconds = time.perf_counter()-tick
    noiseless_check(code, config['L'], 'cpp')
    code.update(dmin=32, Admin=spectrum[32], A_target=spectrum[32],
                complete_spectrum=spectrum, spectrum=spectrum,
                spectrum_exact=True, verification_status='verified', noiseless_L32_pass=True)
    report.update(dmin=32, Admin=spectrum[32], before_Admin=baseline[32],
                  complete_spectrum=spectrum, complete_verification_seconds=verification_seconds,
                  seconds=time.perf_counter()-started,
                  code_path=str((args.output/'code.json').resolve()))
    write(args.output/'code.json', code)
    write(args.output/'summary.json', report)
    print(json.dumps(dict(stage='verified', name=code['name'], dmin=32,
                         Admin=spectrum[32], seconds=report['seconds'])), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--origin-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=int, default=600000000)
    parser.add_argument('--defer-spectrum-verification', action='store_true',
                        help='Freeze the chosen coefficients before the final spectrum audit')
    run(parser.parse_args())
