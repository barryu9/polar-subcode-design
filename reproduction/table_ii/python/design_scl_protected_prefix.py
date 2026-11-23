"""Decoding-prefix protection for a fixed information set (Section V.D).

The leading-position bound identifies early frozen expressions that can change
while preserving the set of low-weight codewords through the cutoff. For the
Fig. 10 Proposed code, the design score is the total weight of competing
decoding prefixes after applying a candidate expression at the current bit.
Full-frame decoding then measures how often the transmitted path leaves the list.
"""

import argparse
from collections import Counter
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
from extended_exact_spectrum import canonical_checks, measure
from feasible_design_bank import generator_rows
from polar_code import _scl_step_decoder_cpp
from polar_code.rp_funcs import rp_GA
from polar_code.pc_funcs import pc_init
from polar_code.SCL_full_decoder import decode_full_path_batch, simulate_awgn_llr_batch
from polar_code.SCL_step_decoder_batch import _propagate_llr_batch, pc_SCL_step_decoder_pxb
from run_diverse_profile_suite import identities
from run_n64_scl_comparison import normalize_json
from refine_protected_core_spectrum import write
from select_designs_scl import construction, noiseless_check

def protected_boundary(profile, cutoff, n):
    # i_{w_star} is the first information position whose polar row has weight
    # at most the protection cutoff. The prefix domain uses frozen j below it.
    return min((i for i in profile if 1 << i.bit_count() <= cutoff), default=n)

def check_spectral_invariance(n, profile, before, after, cutoff):
    canonical_checks(n, profile, before)
    canonical_checks(n, profile, after)
    boundary = protected_boundary(profile, cutoff, n)
    changed = [j for j, (a, b) in enumerate(zip(before, after)) if a != b]
    if any(j >= boundary for j in changed):
        raise ValueError('An edit is outside the spectrum-neutral prefix')
    return dict(exact=True, cutoff=cutoff, boundary=boundary, changed_rows=changed,
                earliest_information_coset_weight=min(
                    (1 << i.bit_count() for i in profile if i < boundary), default=None),
                argument='Every nonzero message before boundary has leading coset '
                         'weight above cutoff. Messages zero before boundary produce '
                         'identical full codewords because all changed rows precede it.')

def conditional_expectation_row(x, delta, order):
    """Minimize an affine parity cost relative to its fair-coefficient average."""
    ordered = x[:, order]
    last = x.shape[1]-1-np.argmax(ordered[:, ::-1], axis=1)
    last[~np.any(ordered, axis=1)] = -1
    chosen = np.zeros(x.shape[1], dtype=np.uint8)
    for t in range(x.shape[1]):
        group = last == t
        if np.any(group):
            parity = (ordered[group, :t] @ chosen[:t]) & 1
            gain = np.dot(delta[group], 1.-2.*parity)
            chosen[t] = gain < -1e-12
    result = np.zeros_like(chosen)
    result[order] = chosen
    return result

def choose_expression(x, c0, c1, baseline, max_flips=8):
    """Choose one frozen expression shared by all observed decoding prefixes.

    A row of x holds the earlier information bits on one competing path.
    Its parity with a candidate coefficient vector determines whether that
    path contributes c0 or c1 to the score. Compare the current expression,
    the all-zero expression and the proposals, then try single-bit improvements.
    """
    delta = c1-c0
    importance = x.T @ np.abs(delta)
    order = np.lexsort((np.arange(x.shape[1]), -importance))
    candidates = [baseline.copy(), np.zeros(x.shape[1], dtype=np.uint8)]
    candidates.extend(conditional_expectation_row(x, delta, sequence)
                      for sequence in (order, np.arange(x.shape[1]),
                                       np.arange(x.shape[1]-1, -1, -1)))
    def objective(v):
        return float(np.sum(c0)+np.dot(delta, (x @ v) & 1))
    current = min(candidates, key=lambda v: (objective(v), int(v.sum()), tuple(v)))
    current = current.copy()
    parity = (x @ current) & 1
    for _ in range(max_flips):
        gains = x.T @ (delta*(1.-2.*parity))
        index = int(np.argmin(gains))
        if gains[index] >= -1e-12:
            break
        current[index] ^= 1
        parity ^= x[:, index]
    return current, dict(before=objective(baseline), after=objective(current),
                         observations=len(x), variables=x.shape[1])

def generator_distance_witness(n, profile, columns, target):
    """A target-weight generator proves equality, not absence of lighter words."""
    rows = generator_rows(n, profile, columns)
    upper = min(x.bit_count() for x in rows)
    if upper < target:
        raise AssertionError('A generator violates the protected distance bound')
    witness = next((dict(information_positions=[i], codeword_hex=hex(x), weight=target)
                    for i, x in zip(profile, rows) if x.bit_count() == target), None)
    return upper, witness

def rank_risk_costs(relative_metric, llr, temperature=2.):
    """Soft upper surrogate for wrong paths outranking the true prefix.

    relative_metric is the updated true-prefix metric minus the current
    wrong-prefix metric. This local surrogate is not a FER probability.
    """
    if temperature <= 0:
        raise ValueError('Risk temperature must be positive')
    return (np.logaddexp(0., (relative_metric-np.maximum(-llr, 0.))/temperature),
            np.logaddexp(0., (relative_metric-np.maximum(llr, 0.))/temperature))

def design_prefix(n, profile, columns, cutoff, llr, list_size, row_floor=1,
                  max_flips=8, event=None, cost_mode='mass', risk_temperature=2.):
    if cost_mode not in ('mass', 'rank_risk'):
        raise ValueError('Unknown prefix objective')
    profile = tuple(profile)
    info = set(profile)
    boundary = protected_boundary(profile, cutoff, n)
    result = list(columns)
    batch, length = llr.shape
    if length != n:
        raise ValueError('Channel observations have the wrong length')
    params = pc_init(n, len(profile), 0, np.asarray(((1, 0), (1, 1)), dtype=int))
    pm = np.zeros((batch, list_size), dtype=np.float64)
    p = np.zeros((batch, n-1, list_size), dtype=np.float64)
    c = np.zeros((batch, n-1, 2*list_size), dtype=np.int8)
    u = np.zeros((batch, n, list_size), dtype=np.int8)
    active = np.zeros((batch, list_size), dtype=bool)
    active[:, 0] = True
    first_loss = np.full(batch, -1, dtype=np.int64)
    events = []
    for j in range(n):
        sources = np.asarray([i for i in profile if i < j], dtype=int)
        baseline = np.asarray([(result[j] >> int(i)) & 1 for i in sources], dtype=np.uint8)
        candidate = (j not in info and j < boundary and len(sources)
                     and 1 << j.bit_count() >= row_floor)
        if candidate:
            # Propagation alone leaves path metrics and partial sums unchanged.
            # The standard native transition recomputes the same LLR afterwards.
            _propagate_llr_batch(params, j, llr, p, c, list_size)
            history = u[:, sources, :].transpose(0, 2, 1)
            wrong = np.any(history != 0, axis=2)
            # Design frames transmit the all-zero input. A competing prefix
            # contributes to this score while the transmitted path is still
            # in the list for the same frame.
            correct_alive = np.any(active & ~wrong, axis=1)
            eligible = active & wrong & correct_alive[:, None]
            if np.any(eligible):
                val = p[:, 0, :][eligible]
                if cost_mode == 'mass':
                    raw = np.exp(-np.minimum(80., pm-np.min(np.where(active, pm, np.inf), axis=1)[:, None]))
                    raw[~active] = 0.
                    weights = raw/np.sum(raw, axis=1)[:, None]
                    w = weights[eligible]
                    c0 = w*np.exp(-np.minimum(80., np.maximum(-val, 0.)))
                    c1 = w*np.exp(-np.minimum(80., np.maximum(val, 0.)))
                else:
                    correct = active & ~wrong
                    true_index = np.argmin(np.where(correct, pm, np.inf), axis=1)
                    true_metric = pm[np.arange(batch), true_index]
                    true_llr = p[np.arange(batch), 0, true_index]
                    updated_true = true_metric+np.maximum(-true_llr, 0.)
                    relative = (updated_true[:, None]-pm)[eligible]
                    c0, c1 = rank_risk_costs(relative, val, risk_temperature)
                chosen, score = choose_expression(
                    np.ascontiguousarray(history[eligible], dtype=np.uint8),
                    c0, c1, baseline, max_flips)
                result[j] = sum(1 << int(i) for i, bit in zip(sources, chosen) if bit)
                record = dict(position=j, support_size=int(chosen.sum()),
                              live_design_frames=int(correct_alive.sum()), **score)
                events.append(record)
                if event is not None:
                    event(record)
        support = [i for i in profile if result[j] >> i & 1]
        expressions = np.full((batch, max(1, len(support))), -1, dtype=np.int64)
        if support:
            expressions[:, :len(support)] = support
        actions = np.full(batch, 0 if j in info else (2 if support else 1), dtype=np.int8)
        loss, _, _ = pc_SCL_step_decoder_pxb(
            params, j, actions, expressions, llr, list_size, pm, p, c, u, active,
            backend='cpp')
        newly_lost = loss & (first_loss < 0)
        first_loss[newly_lost] = j
    best = np.argmin(np.where(active, pm, np.inf), axis=1)
    decoded = u[np.arange(batch), :, best]
    failures = np.any(decoded != 0, axis=1)
    proof = check_spectral_invariance(n, profile, columns, result, cutoff)
    # Replaying the final construction must match every adaptive design trace.
    actions, supports = construction(n, profile, result)
    _, replay_loss, _, replay_u = decode_full_path_batch(
        params, actions, supports, llr, list_size, early_stop=False,
        terminal_margin=0., max_support_size=None, backend='cpp', return_decoded_u=True)
    if not (np.array_equal(replay_loss, first_loss) and np.array_equal(replay_u, decoded)):
        raise AssertionError('Stepwise construction and native full-path replay disagree')
    return result, dict(spectrum_invariance=proof, rows=events,
                        cost_mode=cost_mode, risk_temperature=risk_temperature,
                        design_errors=int(failures.sum()),
                        design_pruning_errors=int(np.sum(first_loss >= 0)),
                        first_loss_histogram=dict(sorted(Counter(map(int, first_loss[first_loss >= 0])).items())),
                        step_full_replay_identical=True)

def run(args):
    raw = json.loads(args.source.read_text(encoding='utf-8-sig'))
    design = normalize_json(raw)
    n, profile, columns = design['N'], tuple(design['profile']), design['columns']
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    spectrum = measure(n, profile, columns, args.cutoff, args.work)
    if any(0 < int(w) < args.target and a for w, a in spectrum['spectrum'].items()):
        raise ValueError('The starting construction does not meet the target distance')
    config = dict(N=n, K=len(profile), cutoff=args.cutoff, target=args.target,
                  source=str(args.source.resolve()),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  seed=args.seed, design_frames=args.frames, design_EbN0_dB=args.snr,
                  L=args.list_size, row_floor=args.row_floor, max_flips=args.max_flips,
                  native_decoder_sha256=hashlib.sha256(Path(_scl_step_decoder_cpp.__file__).read_bytes()).hexdigest(),
                  coefficient_rule='Conditional expectation, fixed tie breaking, then coordinate descent',
                  objective='Sampled surviving wrong-prefix mass after the frozen transition',
                  entropy_claim=False, FER_guarantee=False, coefficient_seed_search=False,
                  reference_support_read=False, sample_driven=True,
                  profile_design_replayed=False, baseline_spectrum=spectrum,
                  target_layer_preserved_by_requested_cutoff=args.cutoff >= args.target)
    write(args.output/'protocol.json', config)
    llr = simulate_awgn_llr_batch(n, len(profile), args.snr, range(args.seed, args.seed+args.frames))
    params = pc_init(n, len(profile), 0, np.asarray(((1, 0), (1, 1)), dtype=int))
    actions, supports = construction(n, profile, columns)
    _, before_loss, _, before_u = decode_full_path_batch(
        params, actions, supports, llr, args.list_size, early_stop=False,
        terminal_margin=0., max_support_size=None, backend='cpp', return_decoded_u=True)
    before_errors = int(np.any(before_u != 0, axis=1).sum())
    before_pruning_errors = int(np.sum(before_loss >= 0))
    with (args.output/'rows.jsonl').open('w', encoding='utf-8') as log:
        def event(record):
            record = dict(**record, seconds=time.perf_counter()-started)
            log.write(json.dumps(record)+'\n')
            log.flush()
            if record['position'] % 32 < 2:
                print(json.dumps(record), flush=True)
        result, report = design_prefix(n, profile, columns, args.cutoff, llr,
                                        args.list_size, args.row_floor, args.max_flips, event)
    ph, sh = identities(n, profile, result)
    final_spectrum = measure(n, profile, result, args.cutoff, args.work,
                             reduce_repetitions=False)
    if final_spectrum['spectrum'] != spectrum['spectrum']:
        raise AssertionError('The protected complete spectrum changed')
    try:
        target_layer_proof = check_spectral_invariance(n, profile, columns, result, args.target)
    except ValueError:
        target_layer_proof = None
    distances = [int(w) for w, a in spectrum['spectrum'].items() if w and a]
    dmin = min(distances) if distances else None
    upper, witness = generator_distance_witness(n, profile, result, args.target)
    if dmin is None and witness is not None:
        dmin = args.target
    ga_snr = float(raw.get('GA_design_EbN0_dB', args.ga_design_snr))
    ga = rp_GA(n, len(profile), ga_snr)
    ga_proxy = -math.expm1(sum(math.log1p(-float(ga.pe[i])) for i in profile))
    code = dict(N=n, K=len(profile), profile=profile, columns=result,
                profile_hash=ph, support_hash=sh, name=ph[:12]+'_'+sh[:12],
                dmin=dmin, distance_lower_bound=args.target,
                Admin=spectrum['spectrum'].get(dmin) if dmin else None,
                spectrum=spectrum['spectrum'], spectrum_invariance=report['spectrum_invariance'],
                verification=final_spectrum, distance_upper_bound=upper,
                matching_witness=witness, dp=min(1 << i.bit_count() for i in profile),
                GA_proxy=ga_proxy, GA_design_EbN0_dB=ga_snr,
                GA_proxy_definition='1 - product(1 - pe[i])',
                source=str(args.source.resolve()), source_sha256=config['source_sha256'],
                prefix_protection_completed=True, FER_verified=False,
                target_layer_preserved_by_requested_cutoff=args.cutoff >= args.target,
                target_layer_preservation_proved=target_layer_proof is not None,
                target_layer_invariance_proof=target_layer_proof,
                reference_support_read=False, method='Spectrum-neutral prefix-list completion')
    noiseless_check(code, args.list_size, 'cpp')
    write(args.output/'code.json', code)
    report.update(seconds=time.perf_counter()-started, configuration=config,
                  code_path=str((args.output/'code.json').resolve()),
                  nonzero_rows=sum(bool(c) for c in result),
                  coefficients=sum(c.bit_count() for c in result),
                  before_design_errors=before_errors,
                  before_design_pruning_errors=before_pruning_errors,
                  final_complete_spectrum=final_spectrum,
                  matching_witness=witness, dmin=dmin, GA_proxy=ga_proxy,
                  GA_design_EbN0_dB=ga_snr, independent_test_performed=False)
    write(args.output/'summary.json', report)
    print(json.dumps(dict(stage='complete', code=code['name'], seconds=report['seconds'],
                         dmin=dmin, Admin=code['Admin'], design_errors=report['design_errors'],
                         nonzero_rows=report['nonzero_rows'], coefficients=report['coefficients'])), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target', type=int, default=24)
    p.add_argument('--cutoff', type=int, default=24)
    p.add_argument('--work', type=int, default=100000000)
    p.add_argument('--frames', type=int, default=512)
    p.add_argument('--snr', type=float, default=1.0)
    p.add_argument('--ga-design-snr', type=float, default=2.25)
    p.add_argument('--list-size', type=int, default=32)
    p.add_argument('--row-floor', type=int, default=1)
    p.add_argument('--max-flips', type=int, default=8)
    p.add_argument('--seed', type=int, default=2026100461000000)
    args = p.parse_args()
    if (min(args.frames, args.list_size, args.target, args.row_floor, args.work) < 1
            or args.cutoff < args.target-1):
        p.error('Positive parameters and cutoff >= target-1 are required')
    run(args)
