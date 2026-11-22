"""Profile-only deterministic construction using existing exact search kernels.

Try the profile-derived first domain, directly or on the right constituent.
In split mode the left support stays zero and outer siblings remain revisitable.
A failed first domain/pair is not declared infeasible for the full profile.
"""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import time
from obstruction_support_search import enumerate_words
from native_paths import library_name
from plotkin_relative_structure import RelativePlotkinSpectrum
from radius_three_completion import SourceCompletion, prepare_exclusion
from run_manuscript_n256_extension import write
from run_manuscript_n256_reduced_joint import verify
from select_structured_domain import initial_structure, complete_domain
from verify_causal_flag_theory import polar_transform, rref

def complete_outer(profile, columns, distance, work, seconds, max_nodes, folder, resume=False):
    """Exact source-wise completion of a fixed pair; no parity shortcuts."""
    folder.mkdir(parents=True, exist_ok=resume)
    saved = None
    if resume and (folder/'frontier.json').exists():
        saved = json.loads((folder/'frontier.json').read_text())
        if (saved['fixed_profile'] != list(profile) or saved['fixed_columns'] != list(columns)
                or saved['distance'] != distance):
            raise ValueError('The outer frontier belongs to a different fixed pair')
    model = RelativePlotkinSpectrum(profile, columns)
    half, cutoff = model.half, distance-1
    rows = tuple(j for j in range(half) if j not in model.right_profile)
    stages = model.left_profile
    info_mask = sum(1 << i for i in stages)
    right = verify(half, model.right_profile, model.right_columns,
                   cutoff//2, work, both=True)
    if right['cpp']['spectrum'] != {0: 1}:
        raise ValueError('The right constituent fails its necessary distance condition')
    left_words, left = enumerate_words(half, stages, model.left_columns, cutoff, work)
    checked_left = verify(half, stages, model.left_columns, cutoff, work, both=True)
    if left['spectrum'] != checked_left['cpp']['spectrum']:
        raise AssertionError('Left catalogue is incomplete')
    write(folder/'left_verified.json', checked_left)
    write(folder/'right_verified.json', right)
    write(folder/'left_catalogue.json', left_words)
    groups = defaultdict(list)
    for u, _ in left_words:
        groups[(u & info_mask).bit_length()-1].append(u)
    unfinished_stages = stages[saved['index']:] if saved is not None else stages
    prepared = {i: [prepare_exclusion(model, u, cutoff, info_mask) for u in groups[i]]
                for i in unfinished_stages}
    # Complete quotient coverage cannot be repaired by changing earlier sources.
    covered = next((item for items in prepared.values() for item in items
                    if len(item.labels) == 1 << item.rank), None)
    if covered is not None:
        item = model.constraint(covered.left_input, cutoff, materialize=False)
        certificate = dict(left_input=covered.left_input, message=covered.message,
            left_weight=item['left_weight'], radius=item['radius'],
            projected_rank=covered.rank, forbidden_labels=len(covered.labels),
            fixed_pair_distance_upper_bound=item['left_weight']+2*item['radius'],
            full_profile_infeasibility=False)
        write(folder/'covered_quotient_obstruction.json', certificate)
        result = dict(status='unsat', nodes=0, backtracks=0, seconds=0,
            source_statistics={}, left_cosets=len(left_words),
            cross_variables=len(rows)*len(stages), infeasible_pair=True,
            unresolved=False, reason='A single coset covers its entire quotient',
            scope='Only this fixed constituent pair with unrestricted outer coefficients',
            obstruction=certificate)
        write(folder/'result.json', result)
        return None, result
    cross, frames, statistics = {}, [], {}
    index = nodes = backtracks = 0
    if saved is not None:
        cross = {int(i): value for i,value in saved['cross'].items()}
        frames = saved['frames']
        index, nodes, backtracks = saved['index'], saved['nodes'], saved['backtracks']
        if (folder/'result.json').exists():
            statistics = {int(i): row for i,row in
                          json.loads((folder/'result.json').read_text())['source_statistics'].items()}
        max_nodes += nodes
    deadline = time.perf_counter()+seconds
    start = time.perf_counter()
    status = 'pending'
    with (folder/'events.jsonl').open('a' if saved is not None else 'w', encoding='utf-8') as log:
        def emit(kind, **fields):
            record = dict(kind=kind, seconds=time.perf_counter()-start, **fields)
            log.write(json.dumps(record)+'\n')
            log.flush()
            print(json.dumps(record), flush=True)

        if saved is not None:
            emit('outer_resumed', completed_sources=index, total_sources=len(stages),
                 previous_nodes=nodes, additional_nodes=max_nodes-nodes)
        while index < len(stages) and nodes < max_nodes and time.perf_counter() < deadline:
            source = stages[index]
            if source not in prepared:
                # Reconstruct an earlier group only if backtracking revisits it.
                prepared[source] = [prepare_exclusion(model,u,cutoff,info_mask) for u in groups[source]]
            frontier = frames[index]['frontier'] if index < len(frames) else None
            solver = SourceCompletion(prepared[source], cross, len(rows), frontier)
            initial = solver.expectation()
            statistics[source] = dict(
                constraints=len(prepared[source]), initial_mean=str(initial),
                projected_ranks=dict(Counter(x.rank for x in prepared[source])),
                forbidden_set_sizes=[len(x.labels) for x in prepared[source]])
            answer = dict(status='pending')
            while nodes+solver.nodes < max_nodes and time.perf_counter() < deadline:
                answer = solver.step()
                if answer['status'] != 'pending':
                    break
            nodes += solver.nodes
            frame = dict(source=source, frontier=solver.snapshot())
            if index < len(frames):
                frames[index] = frame
            else:
                frames.append(frame)
            if answer['status'] == 'sat':
                value = answer['value']
                cross[source] = value
                frame['selected'] = value
                frame['expectation_path'] = solver.expectation_path(value)
                strict = next(p for p in frame['expectation_path']
                              if Fraction(p['expectation']) < 1)
                frame['strict_prefix_bound'] = solver.prefix_completion_bound(
                    strict['fixed_bits'], value & ((1 << strict['fixed_bits'])-1))
                del frames[index+1:]
                emit('source_completed', source=source, value=value,
                     initial_mean=str(initial), nodes=solver.nodes,
                     strict_prefix=frame['strict_prefix_bound'])
                index += 1
            elif answer['status'] == 'unsat':
                del frames[index:]
                if index == 0:
                    status = 'unsat'
                    break
                index -= 1
                cross.pop(stages[index])
                backtracks += 1
                emit('backtrack', source=source, revisit=stages[index])
            else:
                break
        if index == len(stages):
            status = 'sat'
        write(folder/'frontier.json', dict(
            fixed_profile=profile, fixed_columns=columns, distance=distance,
            index=index, cross=cross, frames=frames, nodes=nodes, backtracks=backtracks))
    result = dict(status=status, nodes=nodes, backtracks=backtracks,
                  seconds=time.perf_counter()-start, source_statistics=statistics,
                  left_cosets=len(left_words), cross_variables=len(rows)*len(stages),
                  scope='Only this fixed constituent pair with unrestricted outer coefficients',
                  infeasible_pair=status == 'unsat', unresolved=status == 'pending')
    write(folder/'result.json', result)
    if status != 'sat':
        return None, result
    child = list(columns)
    for q, row in enumerate(rows):
        child[half+row] = ((child[half+row] >> half) << half) | sum(
            ((value >> q) & 1) << source for source, value in cross.items())
    write(folder/'unverified_candidate.json', dict(N=len(child), profile=profile,
          columns=child, target=distance, construction_result=result))
    relative = RelativePlotkinSpectrum(profile, child).spectrum(
        left_words, right['cpp']['spectrum'], cutoff)
    if relative['spectrum'] != {0: 1}:
        raise AssertionError('Outer conditions did not protect the entire cutoff')
    write(folder/'relative_verified.json', relative)
    return child, result

def validate_candidate(profile, columns, distance, work, folder):
    n = len(columns)
    information = sum(1 << i for i in profile)
    if any(c < 0 or c >> j or c & ~information or (j in profile and c)
           for j, c in enumerate(columns)):
        raise AssertionError('Invalid canonical support')
    generators = [polar_transform((1 << i) | sum(
        ((c >> i) & 1) << j for j, c in enumerate(columns)), n) for i in profile]
    if len(rref(generators)) != len(profile):
        raise AssertionError('Wrong generator rank')
    audited = verify(n, profile, columns, distance, work, both=True)
    histogram = audited['cpp']['spectrum']
    if any(count for weight, count in histogram.items() if 0 < weight < distance):
        raise AssertionError('A forbidden weight survived')
    words, metadata = enumerate_words(n, profile, columns, distance, work)
    if metadata['spectrum'] != histogram:
        raise AssertionError('Scalar counts and explicit catalogue disagree')
    target_words = [(u, w) for u, w in words if w == distance]
    witness = None
    if target_words:
        u = target_words[0][0]
        x = polar_transform(u, n)
        if any((u & ((1 << j) ^ columns[j])).bit_count() & 1
               for j in range(n) if j not in profile):
            raise AssertionError('Witness fails a dynamic constraint')
        witness = dict(input=u, codeword=x, weight=x.bit_count(),
                       input_support=[i for i in range(n) if u >> i & 1])
    audited.update(N=n, K=len(profile), target_distance=distance,
                   dmin=distance if witness else None, distance_lower_bound=distance,
                   witness=witness, generator_rank=len(generators),
                   positive_word_count=len(words), explicit_catalogue_verified=True,
                   reference_support_read=False, initialization='all_zero')
    write(folder/'final_code.json', audited)
    write(folder/'low_weight_words.json', words)
    write(folder/'support.json', dict(N=n, K=len(profile), profile=profile, columns=columns,
        dynamic_expressions={str(j): [i for i in profile if c >> i & 1]
                             for j, c in enumerate(columns) if c}))
    return audited

def run(args):
    start = time.perf_counter()
    raw = json.loads(args.profile_source.read_text(encoding='utf-8'))
    n, profile = raw['N'], tuple(raw['profile'])
    if len(profile) != raw['K'] or profile != tuple(sorted(set(profile))):
        raise ValueError('Invalid profile')
    initial_structure(n, profile, args.distance)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(N=n, K=len(profile), profile=profile, target_distance=args.distance,
                    mode=args.mode, initialization='all_zero', reference_support_read=False,
                    random_sampling=False, domain_scope='profile-derived first domain only',
                    order='decreasing source; exact product lookahead; weight; integer',
                    config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    snapshots = args.output/'source_snapshot'
    snapshots.mkdir()
    sources = ('run_profile_target_construction.py', 'select_structured_domain.py',
               'compressed_source_search.py', 'block_neighborhood.py',
               library_name('_candidate_block_neighborhood_kernel'), 'radius_three_completion.py',
               'conditional_cross_completion.py', 'plotkin_relative_structure.py',
               'run_manuscript_n256_reduced_joint.py', 'external_low_weight_spectrum.py',
               'obstruction_support_search.py')
    manifest['source_hashes'] = {}
    for name in sources:
        path = Path(__file__).with_name(name)
        shutil.copy2(path, snapshots/name)
        manifest['source_hashes'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(args.output/'manifest.json', manifest)
    initial = verify(n, profile, [0]*n, args.distance, args.work, both=True)
    write(args.output/'zero_support_verified.json', initial)
    half = n//2
    local_n = n if args.mode == 'direct' else half
    local_profile = profile if args.mode == 'direct' else tuple(i-half for i in profile if i >= half)
    target = args.distance if args.mode == 'direct' else (args.distance+1)//2
    structure = initial_structure(local_n, local_profile, target)
    write(args.output/'domain.json', structure)
    if not 1 <= len(structure['rows']) <= 8:
        raise ValueError('The dense first-domain backend requires one to eight released rows')
    mother = verify(local_n, tuple(sorted((*local_profile, *structure['rows']))),
                    [0]*local_n, target-1, args.work, both=True)
    write(args.output/'mother_verified.json', mother)
    print(json.dumps(dict(kind='domain_selected', local_length=local_n,
        local_dimension=len(local_profile), target=target, rows=structure['rows'],
        core_dimension=len(structure['core']), free_coefficients=len(structure['coordinates']),
        mother_spectrum=mother['cpp']['spectrum'])), flush=True)
    child, local_result = complete_domain(local_n, local_profile, [0]*local_n,
        structure['rows'], structure['coordinates'], target, args.output/'local_support', args)
    outer_result = None
    if child is not None and args.mode == 'split':
        embedded = [0]*half+[c << half for c in child]
        write(args.output/'fixed_constituents.json', dict(profile=profile, columns=embedded))
        child, outer_result = complete_outer(profile, embedded, args.distance, args.work,
            args.outer_seconds, args.outer_nodes, args.output/'outer_completion')
    audit = validate_candidate(profile, child, args.distance, args.work, args.output) if child is not None else None
    report = dict(success=audit is not None, dmin=audit['dmin'] if audit else None,
                  spectrum=audit['cpp']['spectrum'] if audit else None,
                  local_result=local_result, outer_result=outer_result,
                  seconds=time.perf_counter()-start, global_infeasibility_claim=False)
    write(args.output/'report.json', report)
    print(json.dumps(dict(kind='complete', success=report['success'], dmin=report['dmin'],
                         spectrum=report['spectrum'], seconds=report['seconds'])), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distance', type=int, required=True)
    parser.add_argument('--mode', choices=('direct', 'split'), default='split')
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--nodes', type=int, default=200_000)
    parser.add_argument('--outer-seconds', type=float, default=120)
    parser.add_argument('--outer-nodes', type=int, default=200_000)
    parser.add_argument('--work', type=int, default=30_000_000)
    run(parser.parse_args())
