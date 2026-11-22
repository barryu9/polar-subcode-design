"""Budgeted whole-row block search in the Proposition 1 representative domain."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from block_neighborhood import BlockCatalogue
from native_paths import library_name
from external_low_weight_spectrum import spectrum, WorkLimitError
from manuscript_affine_experiment import code_from_columns
from manuscript_joint_block import JointBlockDesigner
from run_manuscript_n256_extension import cutoff_source_start, write
from support_graph_policy import Design

ROOT = Path(__file__).resolve().parents[1]

def verify(n, profile, columns, target, work, both=False):
    code = code_from_columns(n, profile, columns)
    result = dict(profile=profile, columns=columns, code=code)
    for backend in (('cpp', 'python') if both else ('cpp',)):
        result[backend] = spectrum(code, target, work, backend)
        assert result[backend]['transmitted_dimension'] == len(profile)
    if both:
        assert result['cpp']['spectrum'] == result['python']['spectrum']
    return result

def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    state = json.loads(a.source.read_text(encoding='utf-8'))
    full_profile = tuple(state.get('profile') or
                         [i for i, action in enumerate(state['code']['actions']) if action == 0])
    offset = 256-a.n
    profile = tuple(i-offset for i in full_profile if i >= offset)
    columns = [c >> offset for c in state['columns'][offset:]]
    target = a.n*20//256
    first = cutoff_source_start(profile, target, a.n)
    columns = [c & ~((1 << first)-1) for c in columns]
    design = Design(a.n, profile)
    rows = [j for j in design.rows if any(first <= i < j for i in profile)]
    snapshot = a.output/'source_snapshot'
    snapshot.mkdir()
    hashes = {}
    for name in ('run_manuscript_n256_reduced_joint.py', 'manuscript_joint_block.py',
                 'block_neighborhood.py', library_name('_block_neighborhood_kernel'),
                 'run_manuscript_n256_extension.py', 'external_low_weight_spectrum.py') + (
                     (library_name('_candidate_block_neighborhood_kernel'), 'candidate_block_neighborhood_kernel.cpp',
                      'block_neighborhood_kernel.cpp') if a.causal_catalogue else ()):
        path = (ROOT.parents[1]/'cpp'/name if name.endswith('.cpp') else ROOT/'python'/name)
        shutil.copy2(path, snapshot/name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(a.output/'source_code.json', state)
    write(a.output/'manifest.json', dict(config={k: str(v) if isinstance(v, Path) else v
                                                for k, v in vars(a).items()},
          source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(), sources=hashes,
          profile=profile, first_source=first,
          retained_coefficients=sum(first <= i < j for i in profile for j in rows),
          no_rm_constraints=True, protocol='Complete relaxed catalogue; all retained block coefficients are variables; exact word exclusion and bounded solver.'))
    start = time.perf_counter()
    initial = verify(a.n, profile, columns, target, a.enum_work, both=True)
    write(a.output/'initial_code.json', initial)
    hist = initial['cpp']['spectrum']
    if any(c for w, c in hist.items() if 0 < w < target):
        raise ValueError('Initial code has lower-weight words')
    best = hist.get(target, 0)
    initial_count = best
    rng = random.Random(a.seed)
    costs = Counter()
    history = [dict(step=0, value=best, seconds=time.perf_counter()-start)]
    with (a.output/'events.jsonl').open('w', encoding='utf-8') as events:
        for step in range(1, a.steps+1):
            if best == 0 or time.perf_counter()-start >= a.seconds:
                break
            size = a.block_sizes[(step-1) % len(a.block_sizes)]
            # Alternate full-domain blocks and late blocks covering suffix words.
            pool = rows if step % 2 == 0 else rows[len(rows)//2:]
            selected = tuple(sorted(rng.sample(pool, min(size, len(pool)))))
            tick = time.perf_counter()
            print('BLOCK', step, selected, 'best', best, flush=True)
            try:
                catalogue = BlockCatalogue(design, columns, selected, target, a.enum_work,
                                           causal_leaders_only=a.causal_catalogue)
                costs['catalogues'] += 1
                costs['enumeration_seconds'] += time.perf_counter()-tick
                build = time.perf_counter()
                engine = JointBlockDesigner(catalogue, first, a.seed+step)
                costs['build_seconds'] += time.perf_counter()-build
                remaining = a.seconds-(time.perf_counter()-start)
                result = engine.minimize(min(a.block_seconds, max(0, remaining)),
                                         a.call_seconds, a.max_calls)
                costs['solver_seconds'] += result['seconds']
                costs['solver_calls'] += len(result['trace'])
                before = best
                if result['value'] < best:
                    candidate = catalogue.child(result['deltas'])
                    audit = verify(a.n, profile, candidate, target, a.enum_work)
                    h = audit['cpp']['spectrum']
                    assert not any(c for w, c in h.items() if 0 < w < target)
                    assert h.get(target, 0) == result['value']
                    columns, best = candidate, result['value']
                    history.append(dict(step=step, value=best, rows=selected,
                                        seconds=time.perf_counter()-start))
                    write(a.output/'best_code.json', audit)
                    print('BEST', step, 'A'+str(target), best, flush=True)
                event = dict(kind='block', step=step, rows=selected, before=before,
                             after=best, catalogue=catalogue.stats, result=result,
                             seconds=time.perf_counter()-start)
            except WorkLimitError as exc:
                costs['resource_limits'] += 1
                event = dict(kind='resource_limit', step=step, rows=selected, error=str(exc),
                             seconds=time.perf_counter()-start)
            events.write(json.dumps(event)+'\n')
            events.flush()
            write(a.output/'progress.json', dict(best=best, costs=dict(costs), history=history,
                  seconds=time.perf_counter()-start, latest_event=event))
    final = verify(a.n, profile, columns, target, a.enum_work, both=True)
    write(a.output/'final_code_verified.json', final)
    report = dict(n=a.n, target=target, initial=initial_count, best=best, success=best == 0,
                  seconds=time.perf_counter()-start, costs=dict(costs), history=history,
                  final_spectrum=final['cpp']['spectrum'],
                  full_n256_success=(best == 0 if a.n == 256 else None))
    write(a.output/'report.json', report)
    print('COMPLETE', json.dumps(report), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, choices=(128, 256), default=256)
    p.add_argument('--seed', type=int, default=120916601)
    p.add_argument('--seconds', type=float, default=900)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--enum-work', type=int, default=100_000_000)
    p.add_argument('--block-sizes', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--block-seconds', type=float, default=90)
    p.add_argument('--call-seconds', type=float, default=15)
    p.add_argument('--max-calls', type=int, default=8)
    p.add_argument('--causal-catalogue', action='store_true',
                   help='Exclude relaxed words led by a released frozen bit; all candidate spectra remain exact.')
    run(p.parse_args())
