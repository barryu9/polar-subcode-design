"""Profile-only zero-support target construction with complete mother directories."""

from collections import Counter
import json
import time
from types import SimpleNamespace
from compressed_source_search import PatternSearch, SourcePatternProblem
from obstruction_support_search import enumerate_words
from run_manuscript_n256_extension import write
from run_manuscript_n256_reduced_joint import verify
from select_structured_domain import initial_structure
from support_graph_policy import Design
from verify_causal_flag_theory import polar_transform, rref

def rm_gap_membership(profile, columns, target):
    n = len(columns)
    m = n.bit_length()-1
    threshold = min(i.bit_count() for i in profile)
    baseline = 1 << threshold
    order = m-threshold
    if n != 1 << m or not 2 <= order <= m-2 or not baseline < target <= 3*baseline//2:
        raise ValueError('Unsupported minimum-to-second-weight interval')
    active = set(profile) | {j for j, c in enumerate(columns) if c}
    if any(j.bit_count() < threshold for j in active):
        raise ValueError('Candidate is outside the claimed RM mother')
    return dict(N=n, order=order, variables=m, minimum_weight=baseline,
        next_possible_weight=3*baseline//2, member=True,
        excluded_open_interval=[baseline, 3*baseline//2],
        justification='All active polar inputs have index weight >= m-r; the RM low-weight theorem excludes (d,3d/2).',
        theorem_doi='10.1109/TIT.1970.1054545',
        accessible_source='https://arxiv.org/pdf/2301.13497', source_location='Theorem 2; Section 5.2 for RM(3,8)')

def search_problem(problem, folder, seconds, nodes):
    start = last = time.perf_counter()
    search = PatternSearch(problem, 'reverse', 'product')
    while search.status == 'pending' and search.nodes < nodes and time.perf_counter()-start < seconds:
        search.step()
        if time.perf_counter()-last >= 10:
            print(json.dumps(dict(kind='search', nodes=search.nodes, probes=search.probes,
                                  deepest=search.deepest)), flush=True)
            write(folder/'frontier.json', search.snapshot())
            last = time.perf_counter()
    write(folder/'frontier.json', search.snapshot())
    result = dict(status=search.status, nodes=search.nodes, probes=search.probes,
                  deepest=search.deepest, search_seconds=time.perf_counter()-start,
                  global_infeasibility_claim=False)
    write(folder/'search_result.json', result)
    return search, result

def right(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = json.loads(args.profile.read_text(encoding='utf-8'))
    n = data['N']//2
    profile = tuple(i-n for i in data['profile'] if i >= n)
    target = (args.distance+1)//2
    structure = initial_structure(n, profile, target)
    rows, coordinates = structure['rows'], structure['coordinates']
    if len(rows) > 12:
        raise ValueError('Dense source patterns are limited to twelve released rows')
    write(args.output/'manifest.json', dict(
        profile_input=str(args.profile.resolve()), N=n, K=len(profile), profile=profile,
        target=target, structure=structure, initialization='all_zero',
        reference_support_used=False, random_sampling=False))
    mother_profile = tuple(sorted((*profile, *rows)))
    words, metadata = enumerate_words(n, mother_profile, [0]*n, target-1, args.work)
    write(args.output/'mother_verified.json', metadata)
    catalogue = SimpleNamespace(design=Design(n, profile), columns=[0]*n,
                                rows=rows, domain=coordinates, words=words)
    problem = SourcePatternProblem(catalogue, max_width=12)
    write(args.output/'model.json', problem.statistics())
    print(json.dumps(dict(kind='mother_ready', words=len(words), spectrum=metadata['spectrum'],
        seconds=time.perf_counter()-start, statistics=problem.statistics())), flush=True)
    search, result = search_problem(problem, args.output, args.seconds, args.nodes)
    if search.solution is not None:
        columns = [0]*n
        for j, delta in zip(rows, problem.deltas(search.solution)):
            columns[j] = delta
        write(args.output/'support.json', dict(N=n, K=len(profile), profile=profile, columns=columns))
        check = verify(n, profile, columns, target-1, args.work, both=True)
        if any(w and w < target and c for w, c in check['cpp']['spectrum'].items()):
            raise AssertionError('Right distance was not attained')
        write(args.output/'verified.json', check)
        write(args.output/'outer_input.json', dict(**data, columns=[0]*n+[c << n for c in columns],
            reference_support_read=False, initialization='all_zero'))
        result['spectrum'] = check['cpp']['spectrum']
        result['verified'] = True
        result['witness_sources'] = [i for i in profile if polar_transform(
            (1 << i) | sum(((c >> i) & 1) << j for j, c in enumerate(columns)), n).bit_count() == target]
    result['total_seconds'] = time.perf_counter()-start
    write(args.output/'result.json', result)
    print(json.dumps(dict(kind='right_complete', **result)), flush=True)

def restricted_outer(args):
    """Solve projected coset exclusions on the first profile-derived outer domain."""
    from plotkin_relative_structure import RelativePlotkinSpectrum
    from small_code_words import all_words
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = json.loads(args.profile.read_text(encoding='utf-8'))
    profile, columns = tuple(data['profile']), data['columns']
    n = len(columns)
    structure = initial_structure(n, profile, args.distance)
    rows = tuple(j for j in structure['rows'] if j >= n//2)
    coordinates = tuple((q, i) for q, j in enumerate(rows) for i in profile
                        if i < n//2 and i not in structure['core'])
    cutoff = args.cutoff if args.cutoff is not None else args.distance-1
    write(args.output/'manifest.json', dict(profile=profile, structure=structure, rows=rows,
        coordinates=coordinates, target=args.distance, constraint_cutoff=cutoff,
        lower_cutoff_requires_separate_gap_verification=cutoff < args.distance-1,
        reference_support_used=False, random_sampling=False))
    model = RelativePlotkinSpectrum(profile, columns)
    if len(rows) > 12 or len(model.left_profile) > 20:
        raise ValueError('Restricted pattern width or complete left dimension exceeds limits')
    left = all_words(model.left_profile, model.left_columns)
    write(args.output/'left_spectrum.json', dict(Counter(w for _, w in left)))
    sources = tuple(sorted({i for _, i in coordinates}, reverse=True))
    checks = tuple(j for j in range(n//2) if j not in model.right_profile)
    selected = [checks.index(j-n//2) for j in rows]
    groups = Counter()
    critical = 0
    for u, weight in left:
        if not 0 < weight <= cutoff:
            continue
        critical += 1
        item = model.constraint(u, cutoff)
        images = [item['syndrome_map'][q] for q in selected]
        labels = [0]
        scope = sum(((u >> i) & 1) << q for q, i in enumerate(sources))
        for value in range(1 << len(rows)):
            if value:
                bit = value & -value
                labels.append(labels[value ^ bit] ^ images[bit.bit_length()-1])
            if labels[value] in item['forbidden']:
                groups[scope, value] += 1
    catalogue = SimpleNamespace(design=Design(n, profile), columns=columns,
                                rows=rows, domain=coordinates)
    problem = SourcePatternProblem(catalogue, projected_groups=groups, max_width=12)
    write(args.output/'model.json', dict(statistics=problem.statistics(), critical_left=critical))
    print(json.dumps(dict(kind='outer_model', seconds=time.perf_counter()-start,
        critical_left=critical, statistics=problem.statistics())), flush=True)
    search, result = search_problem(problem, args.output, args.seconds, args.nodes)
    if search.solution is not None:
        child = list(columns)
        for j, delta in zip(rows, problem.deltas(search.solution)):
            child[j] ^= delta
        write(args.output/'support.json', dict(N=n, K=len(profile), profile=profile, columns=child,
            initialization='all_zero', reference_support_read=False))
        # This only verifies the actual cutoff, not an unproved gap above it.
        right = verify(n//2, model.right_profile, model.right_columns, cutoff//2, args.work, both=True)
        check = RelativePlotkinSpectrum(profile, child).spectrum(
            [(u, w) for u, w in left if 0 < w <= cutoff], right['cpp']['spectrum'], cutoff)
        if check['spectrum'] != {0: 1}:
            raise AssertionError('A forbidden coset survived')
        write(args.output/'relative_verified.json', check)
        result.update(cutoff_verified=cutoff, target_verified=cutoff >= args.distance-1)
    result['total_seconds'] = time.perf_counter()-start
    write(args.output/'summary.json', result)
    print(json.dumps(dict(kind='restricted_outer_complete', **result)), flush=True)
