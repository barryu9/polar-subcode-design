"""Bounded structured suffix trials using the manuscript's exact outer solver.

Only a profile is read. No algebraic auxiliary code or existing support is used.
Unresolved trials do not imply domain or profile infeasibility.
"""

from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from external_low_weight_spectrum import WorkLimitError
from feasible_design_bank import code_fingerprint, validate_design
from plotkin_relative_structure import RelativePlotkinSpectrum
from run_manuscript_n256_reduced_joint import verify
from run_profile_target_construction import complete_outer
from run_reference_domain_sweep import write
from select_designs_scl import noiseless_check
from select_structured_domain import complete_domain, initial_structure
from verify_causal_flag_theory import polar_transform, rref

def static_distance(profile, n):
    return min((1 << i.bit_count() for i in profile), default=n+1)

def run(args):
    raw = json.loads(args.profile.read_text(encoding='utf-8-sig'))
    n, profile = raw['N'], tuple(raw['profile'])
    if profile != tuple(sorted(set(profile))) or len(profile) != raw['K']:
        raise ValueError('Invalid profile')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'profile.json', dict(N=n, K=len(profile), profile=profile, target=args.target))
    sources = ['run_manuscript_suffix_construction.py', 'run_profile_target_construction.py',
               'radius_three_completion.py', 'plotkin_relative_structure.py',
               'obstruction_support_search.py', 'external_low_weight_spectrum.py']
    write(args.output/'manifest.json', dict(
        initialization='all_zero', profile_source=str(args.profile.resolve()),
        profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in sources},
        old_support_read=False, algebraic_kernel_used=False, random_sampling=False,
        local_direct_row_threshold=8, maximum_outside_radius=3,
        solver_seconds_per_pair=args.seconds, enumeration_work_limit=args.work,
        scope='Structured-trial branch; no complete-domain fallback or general infeasibility claim'))
    started = time.perf_counter()
    stages = []
    columns = []
    status = 'running'

    def event(kind, **fields):
        record = dict(kind=kind, elapsed_seconds=time.perf_counter()-started, **fields)
        with (args.output/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        write(args.output/'progress.json', record)
        print(json.dumps(record), flush=True)

    try:
        for length in [1 << p for p in range(3, n.bit_length())]:
            offset = n-length
            info = tuple(i-offset for i in profile if i >= offset)
            target = (args.target*length+n-1)//n
            dp = static_distance(info, length)
            folder = args.output/f'n{length}'
            folder.mkdir()
            tick = time.perf_counter()
            structure = initial_structure(length, info, target) if info else {}
            record = dict(N=length, K=len(info), target=target, dp=dp,
                          initial_rows=len(structure.get('rows', [])),
                          initial_coefficients=len(structure.get('coordinates', [])))
            write(folder/'domain.json', structure)
            if dp >= target:
                columns = [0]*length
                record.update(status='static_sufficient', distance_lower_bound=dp,
                              proof='Unprecoded polar minimum distance from generator row weights')
            else:
                half = length//2
                if len(columns) != half:
                    raise ValueError('Missing verified shorter suffix')
                left = tuple(i for i in info if i < half)
                left_dp = static_distance(left, half)
                seed = [0]*half+[c << half for c in columns]
                record.update(left_K=len(left), left_dp=left_dp,
                              maximum_outside_radius=max(0, (target-1-left_dp)//2))
                event('stage_started', **record)
                if left_dp >= target:
                    columns = seed
                    record.update(status='zero_outer_sufficient',
                                  proof='Plotkin bound min(d_left,2*d_right)',
                                  distance_lower_bound=min(left_dp, 2*stages[-1]['distance_lower_bound']))
                else:
                    direct = record['initial_rows'] <= 8
                    if (direct and length > 256) or (not direct and (
                            record['maximum_outside_radius'] > 3 or half > 256)):
                        record.update(status='representation_limit', profile_infeasibility_proved=False)
                        record['seconds'] = time.perf_counter()-tick
                        stages.append(record)
                        status = 'structured_trial_unfinished'
                        break
                    outer_started = time.perf_counter()
                    with (folder/'outer_stdout.log').open('w', encoding='utf-8') as log:
                        with redirect_stdout(log):
                            if direct:
                                config = SimpleNamespace(work=args.work, seconds=args.seconds, nodes=args.nodes)
                                child, result = complete_domain(length, info, [0]*length,
                                    structure['rows'], structure['coordinates'], target,
                                    folder/'direct', config)
                            else:
                                child, result = complete_outer(info, seed, target, args.work,
                                                               args.seconds, args.nodes, folder/'outer')
                    record['outer_total_seconds'] = time.perf_counter()-outer_started
                    record['outer_result'] = result
                    if child is None:
                        record.update(status=('local_domain_' if direct else 'fixed_pair_')+result['status'],
                                      profile_infeasibility_proved=False)
                        record['seconds'] = time.perf_counter()-tick
                        stages.append(record)
                        status = 'structured_trial_unfinished'
                        break
                    columns = child
                    record.update(status='local_domain_completed' if direct else 'outer_completed',
                                  distance_lower_bound=target)
                    verification_started = time.perf_counter()
                    if length <= 256:
                        check = verify(length, info, columns, target-1, args.work, both=True)
                        assert check['cpp']['spectrum'] == {0: 1}
                        write(folder/'independent_enumeration.json', check)
                        record['verification'] = 'Independent C++ and Python cutoff enumerations'
                    else:
                        # Recompute the exact Plotkin spectrum after rereading the complete catalogue.
                        cat = json.loads((folder/'outer'/'left_catalogue.json').read_text())
                        right = json.loads((folder/'outer'/'right_verified.json').read_text())
                        hist = {int(w):a for w,a in right['cpp']['spectrum'].items()}
                        check = RelativePlotkinSpectrum(info, columns).spectrum(cat, hist, target-1)
                        assert check['spectrum'] == {0: 1}
                        write(folder/'relative_recheck.json', check)
                        record['verification'] = 'Exact Plotkin recomputation; shared quotient implementation'
                    record['verification_seconds'] = time.perf_counter()-verification_started
            validate_design(length, info, columns)
            rows = [polar_transform((1 << i) | sum(((c >> i) & 1) << j
                    for j,c in enumerate(columns)), length) for i in info]
            assert len(rref(rows)) == len(info)
            record['minimum_generator_weight'] = min((x.bit_count() for x in rows), default=None)
            record['seconds'] = time.perf_counter()-tick
            write(folder/'support.json', dict(N=length, K=len(info), profile=info, columns=columns,
                                             distance_lower_bound=record['distance_lower_bound']))
            write(folder/'result.json', record)
            stages.append(record)
            brief = {key:value for key,value in record.items() if key != 'outer_result'}
            if 'outer_result' in record:
                brief['outer_summary'] = {key:record['outer_result'][key]
                                          for key in ('status', 'nodes', 'backtracks', 'seconds', 'left_cosets')
                                          if key in record['outer_result']}
            event('stage_complete', **brief)
        else:
            final = dict(N=n, K=len(profile), profile=profile, columns=columns,
                         dp=static_distance(profile, n), distance_lower_bound=args.target,
                         initialization='all_zero', reference_support_read=False,
                         method='Manuscript structured suffix and exact conditional-mean outer completion',
                         design_id=code_fingerprint(n, rows))
            minimum = min(x.bit_count() for x in rows)
            final['dmin'] = args.target if minimum == args.target else None
            final['distance_upper_bound'] = minimum
            noiseless_check(final, 32, 'cpp')
            write(args.output/'support.json', final)
            write(args.output/'generators.json', [dict(source=i, word_hex=hex(x), weight=x.bit_count())
                                                 for i,x in zip(profile, rows)])
            status = 'verified_feasible'
    except WorkLimitError as exc:
        status = 'enumeration_limit'
        event('representation_limit', error=str(exc))
    except Exception as exc:
        status = 'implementation_failure'
        event('exception', error=repr(exc))
        raise
    finally:
        report = dict(status=status, stages=stages, seconds=time.perf_counter()-started,
                      old_support_read=False, algebraic_kernel_used=False,
                      profile_design_replayed=False, fixed_profile_trial=True,
                      profile_infeasibility_proved=False)
        write(args.output/'result.json', report)
        event('finished', status=status, seconds=report['seconds'])
