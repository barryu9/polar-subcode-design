"""Zero-support construction with explicit domains, complete directories and checks.

The practical driver tries one profile-derived domain at each level. It is
not the exhaustive all-domain existence algorithm. Exhaustion or a timeout
in this domain never proves that the entire profile is infeasible.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import gc
import time
from types import SimpleNamespace

from polar_subcodes.algebra import all_words, generator, rref
from polar_subcodes.code import Design, PolarSubcode, load_profile
from polar_subcodes.io import Progress, hardware, write_json
from polar_subcodes.spectrum.backend import count, words, WorkLimitError
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.spectrum.decomposition import spectrum
from polar_subcodes.theory.rm_gap import rm_gap_membership
from .domains import initial_structure
from .source_patterns import SourcePatternProblem, PatternSearch
from .bounded import outer_pattern_problem
from .nested_patterns import right_problem
from .completion import SourceCompletion, prepare_exclusion


@dataclass
class Limits:
    seconds: float = 600
    nodes: int = 1_000_000
    work: int = 100_000_000


class SearchIncomplete(RuntimeError):
    """No construction obtained within the chosen domain and resource limits."""


def search_patterns(problem, limits, folder, emit):
    start = last = time.perf_counter()
    solver = PatternSearch(problem,'reverse','product')
    emit('support','Reject every forbidden source pattern using shared coefficients',
         variables=len(problem.catalogue.domain),conditions=len(problem.scopes),
         distinct_pairs=len(problem.groups))
    while solver.status == 'pending' and solver.nodes < limits.nodes and time.perf_counter()-start < limits.seconds:
        solver.step()
        if time.perf_counter()-last >= 10:
            emit('support','Deterministic source-pattern traversal',nodes=solver.nodes,deepest=solver.deepest)
            write_json(folder/'frontier.json',solver.snapshot())
            last = time.perf_counter()
    write_json(folder/'frontier.json',solver.snapshot())
    report = dict(status=solver.status,nodes=solver.nodes,probes=solver.probes,
                  seconds=time.perf_counter()-start,statistics=problem.statistics(),
                  scope='selected coefficient domain only',global_infeasibility_claim=False)
    write_json(folder/'search.json',report)
    if solver.solution is None:
        raise SearchIncomplete(f'Source search ended with {solver.status}, not global infeasibility')
    if problem.survivors(solver.solution):
        raise AssertionError('A forbidden pattern survived')
    columns = list(problem.catalogue.columns)
    for row,delta in zip(problem.catalogue.rows,problem.deltas(solver.solution)):
        columns[row] ^= delta
    emit('support','All constraints in the complete directory are rejected',nodes=solver.nodes)
    return columns,report


def right_construct(code, target, limits, folder, emit, nested=False):
    n = code.n//2
    profile = tuple(i-n for i in code.profile if i>=n)
    structure = initial_structure(n,profile,target)
    rows = structure['rows']
    write_json(folder/'domain.json',structure)
    emit('domain','Release frozen rows of sufficient weight and retain the high-weight core',
         N=n,K=len(profile),d_P=structure['baseline'],target=target,
         released_rows=rows,core_dimension=len(structure['core']),
         free_coefficients=len(structure['coordinates']))
    if nested:
        problem,model = right_problem(profile,target,n)
        write_json(folder/'projected_model.json',model)
        emit('directory','Nested quotient constraints represent all critical cosets',
             critical_cosets=model['critical_left_cosets'])
    else:
        if len(rows)>12:
            raise SearchIncomplete('Dense source patterns require at most twelve released rows')
        mother = PolarSubcode(n,tuple(sorted((*profile,*rows))),(0,)*n)
        catalogue,meta = words(mother,target-1,limits.work)
        independent = count(mother,target-1,limits.work)['spectrum']
        if meta['spectrum'] != independent:
            raise AssertionError('Complete mother list and count disagree')
        write_json(folder/'mother.json',meta)
        emit('directory','Enumerate the complete common mother, including words absent at zero support',
             nonzero_words=len(catalogue),spectrum=independent)
        data = SimpleNamespace(design=Design(n,profile),columns=[0]*n,rows=rows,
                               domain=structure['coordinates'],words=catalogue)
        problem = SourcePatternProblem(data,max_width=12)
    columns,report = search_patterns(problem,limits,folder,emit)
    right = PolarSubcode(n,profile,tuple(columns))
    verified = count(right,target-1,limits.work)
    if verified['spectrum'] != {0:1}:
        raise AssertionError('Right constituent verification failed')
    write_json(folder/'support.json',right.as_dict())
    write_json(folder/'verification.json',verified)
    return PolarSubcode(code.n,code.profile,(0,)*n+tuple(c<<n for c in columns)),report


def complete_outer(code,target,limits,folder,emit):
    """Source-wise conditional means order complete binary-prefix backtracking."""
    model = RelativePlotkinSpectrum(code.profile,code.columns)
    half = model.half
    rows = tuple(j for j in range(half) if j not in model.right_profile)
    if any(c & ((1<<half)-1) for c in code.columns[half:]):
        raise ValueError('Outer completion must begin with zero cross-block coefficients')
    right = PolarSubcode(half,model.right_profile,model.right_columns)
    right_check = count(right,(target-1)//2,limits.work)['spectrum']
    if right_check != {0:1}:
        raise ValueError('The right constituent violates its necessary distance condition')
    left = PolarSubcode(half,model.left_profile,model.left_columns)
    catalogue,metadata = words(left,target-1,limits.work)
    if count(left,target-1,limits.work)['spectrum'] != metadata['spectrum']:
        raise AssertionError('Incomplete left directory')
    groups = defaultdict(list)
    mask = sum(1<<i for i in model.left_profile)
    for u,_ in catalogue:
        groups[(u & mask).bit_length()-1].append(u)
    write_json(folder/'left_directory.json',dict(words=[(str(u),w) for u,w in catalogue],**metadata))
    emit('outer','Protect every weight below the target through exact quotient exclusions',
         target=target,left_cosets=len(catalogue),cross_coefficients=len(rows)*len(model.left_profile))
    sources = model.left_profile
    cross,frames,statistics = {},[],{}
    index = nodes = backtracks = 0
    start = time.perf_counter()
    deadline = start+limits.seconds
    while index<len(sources) and nodes<limits.nodes and time.perf_counter()<deadline:
        source = sources[index]
        prepared = [prepare_exclusion(model,u,target-1,mask) for u in groups[source]]
        frontier = frames[index]['frontier'] if index<len(frames) else None
        solver = SourceCompletion(prepared,cross,len(rows),frontier)
        expectation = solver.expectation()
        statistics[source] = dict(events=len(prepared),initial_mean=str(expectation))
        emit('outer','Solve one source column, retaining unvisited sibling prefixes',
             source=source,events=len(prepared),conditional_mean=str(expectation))
        answer = dict(status='pending')
        while nodes+solver.nodes<limits.nodes and time.perf_counter()<deadline:
            answer = solver.step()
            if answer['status']!='pending':
                break
        nodes += solver.nodes
        frame = dict(source=source,frontier=solver.snapshot())
        if index<len(frames):
            frames[index] = frame
        else:
            frames.append(frame)
        if answer['status']=='sat':
            value = answer['value']
            cross[source] = value
            frame['value'] = value
            frame['conditional_mean_path'] = solver.expectation_path(value)
            del frames[index+1:]
            index += 1
        elif answer['status']=='unsat':
            del frames[index:]
            if index == 0:
                break
            index -= 1
            cross.pop(sources[index])
            backtracks += 1
            emit('backtrack','Revisit the earlier source sharing the same support variables',source=sources[index])
        else:
            break
        write_json(folder/'frontier.json',dict(index=index,cross=cross,frames=frames,nodes=nodes))
        del solver,prepared
        gc.collect()
    report = dict(completed=index==len(sources),nodes=nodes,backtracks=backtracks,
                  seconds=time.perf_counter()-start,source_statistics=statistics)
    write_json(folder/'completion.json',report)
    if not report['completed']:
        raise SearchIncomplete('Outer completion did not finish for the selected constituent pair')
    columns = list(code.columns)
    for q,row in enumerate(rows):
        columns[half+row] |= sum(((value>>q)&1)<<source for source,value in cross.items())
    child = PolarSubcode(code.n,code.profile,tuple(columns))
    check = RelativePlotkinSpectrum(child.profile,child.columns).spectrum(catalogue,right_check,target-1)
    if check['spectrum'] != {0:1}:
        raise AssertionError('A forbidden mixed word survived')
    write_json(folder/'verification.json',check)
    return child,report


def restricted_outer(code,target,limits,folder,emit,use_gap=False):
    structure = initial_structure(code.n,code.profile,target)
    half = code.n//2
    rows = tuple(j for j in structure['rows'] if j>=half)
    coordinates = tuple((q,i) for q,j in enumerate(rows) for i in code.profile
                        if i<half and i not in structure['core'])
    cutoff = target-1
    gap = None
    if use_gap:
        gap = rm_gap_membership(code.profile,code.columns,target)
        cutoff = gap['minimum_weight']
        if any(1<<j.bit_count()<cutoff for j in rows):
            raise ValueError('Selected rows leave the claimed RM weight gap')
        write_json(folder/'gap.json',gap)
    def report(event):
        emit('directory','Build projected forbidden values for the restricted outer domain',**event)
    problem,left = outer_pattern_problem(code.profile,code.columns,rows,coordinates,cutoff+1,report)
    columns,search = search_patterns(problem,limits,folder,emit)
    child = PolarSubcode(code.n,code.profile,tuple(columns))
    model = RelativePlotkinSpectrum(child.profile,child.columns)
    right = PolarSubcode(half,model.right_profile,model.right_columns)
    histogram = count(right,cutoff//2,limits.work)['spectrum']
    if cutoff <= 32 and max(((cutoff-w)//2 for _,w in left if 0<w<=cutoff),default=0)<=4:
        checked = model.spectrum([(u,w) for u,w in left if 0<w<=cutoff],histogram,cutoff)['spectrum']
    else:
        from polar_subcodes.spectrum.meet_in_middle import relative_spectrum
        checked = relative_spectrum(model,left,histogram,cutoff)['spectrum']
    if checked != {0:1}:
        raise AssertionError('Restricted outer construction failed its complete cutoff')
    if gap:
        rm_gap_membership(child.profile,child.columns,target)
    write_json(folder/'verification.json',dict(cutoff=cutoff,spectrum=checked,gap=gap))
    return child,search,gap


def run(profile_path,target,output,limits=None,recipe='auto',count_minimum=True):
    limits = limits or Limits()
    if limits.seconds<=0 or limits.nodes<=0 or limits.work<=0:
        raise ValueError('Resource limits must be positive')
    code = load_profile(profile_path)
    if code.n<8 or not code.k or not 1<=target<=code.n:
        raise ValueError('Construction requires N >= 8, K > 0 and 1 <= target <= N')
    emit = Progress(output)
    folder = emit.folder
    write_json(folder/'profile.json',dict(N=code.n,K=code.k,profile=code.profile))
    write_json(folder/'hardware.json',hardware())
    write_json(folder/'zero_support.json',code.as_dict())
    emit('start','Start from zero support. No stored support is read.',N=code.n,K=code.k,d_P=code.d_P,target=target)
    try:
        tick = time.perf_counter()
        local_n = code.n//2
        local_profile = tuple(i-local_n for i in code.profile if i>=local_n)
        shape = initial_structure(local_n,local_profile,(target+1)//2)
        cross_only = (bool(shape['rows']) and all(j>=local_n//2 for j in shape['rows'])
                      and all(i<local_n//2 for _,i in shape['coordinates'])
                      and max(sum(i<local_n//2 for i in local_profile),
                              sum(i>=local_n//2 for i in local_profile))<=20)
        nested = recipe=='nested' or (recipe=='auto' and cross_only)
        fixed,right_report = right_construct(code,(target+1)//2,limits,folder/'right',emit,nested)
        right_seconds = time.perf_counter()-tick
        write_json(folder/'fixed_constituents.json',fixed.as_dict())
        tick = time.perf_counter()
        left_weights = [1<<i.bit_count() for i in code.profile if i<code.n//2]
        radius = (target-1-min(left_weights))//2 if left_weights else 0
        gap = None
        if recipe=='nested' or nested:
            final,outer_report,gap = restricted_outer(fixed,target,limits,folder/'outer',emit)
        elif recipe=='gap' or (recipe=='auto' and radius>3):
            final,outer_report,gap = restricted_outer(fixed,target,limits,folder/'outer',emit,use_gap=True)
        else:
            final,outer_report = complete_outer(fixed,target,limits,folder/'outer',emit)
        outer_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        rows = generator(final.profile,final.columns)
        if len(rref(rows)) != final.k:
            raise AssertionError('Generator rank mismatch')
        hist = None
        witness = next((dict(message=str(1<<q),input=str(final.input_word(1<<q)),
                            codeword=str(x),weight=x.bit_count())
                        for q,x in enumerate(rows) if x.bit_count()==target),None)
        if count_minimum:
            hist = spectrum(final,target,limits.work,emit,folder/'spectrum')
            if any(a for w,a in hist.items() if 0<w<target) or not hist.get(target):
                raise AssertionError('Final spectrum does not establish the requested distance')
        elif not witness:
            raise SearchIncomplete('Lower bound attained, but no target-weight witness found')
        if gap:
            rm_gap_membership(final.profile,final.columns,target)
        verification = dict(exact=True,distance_lower_bound=target,dmin=target,
            Admin=hist.get(target) if hist else None,coefficient_counted=hist is not None,
            spectrum=hist,generator_rank=final.k,causal=True,witness=witness,gap=gap)
        write_json(folder/'verification.json',verification)
        saved = final.as_dict() | dict(d_P=final.d_P,**{k:verification[k] for k in
                ('dmin','Admin','coefficient_counted')},initialization='zero_support')
        write_json(folder/'support.json',saved)
        (folder/'equations.txt').write_text(final.describe()+'\n',encoding='utf-8')
        report = dict(success=True,N=final.n,K=final.k,**verification,
            right=right_report,outer=outer_report,wall_seconds=dict(
                right=right_seconds,outer=outer_seconds,final_verification=time.perf_counter()-tick),
            total_seconds=time.perf_counter()-emit.start,
            scope='Selected deterministic structured domain, not a universal complexity guarantee')
        write_json(folder/'report.json',report)
        emit('verified','Rank, causality, complete low-weight exclusion and distance checked',
             dmin=target,Admin=verification['Admin'])
        return final,report
    except (SearchIncomplete,WorkLimitError,ValueError) as error:
        write_json(folder/'report.json',dict(success=False,reason=str(error),
                   global_infeasibility_claim=False,elapsed_seconds=time.perf_counter()-emit.start))
        emit('incomplete',str(error))
        raise
