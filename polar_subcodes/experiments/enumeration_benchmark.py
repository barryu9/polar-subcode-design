"""Matched direct and quotient complete-word enumeration on paper cases."""
from collections import Counter
import gc
import statistics
import time
import tracemalloc

from polar_subcodes.algebra import polar_transform
from polar_subcodes.code import PolarSubcode,load_code
from polar_subcodes.io import ROOT,Progress,write_json,hardware
from polar_subcodes.spectrum.backend import words
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum
from polar_subcodes.spectrum.quotient import evaluate,checksum


def fixtures():
    profile = load_code(ROOT/'profile/n64_k32.json').profile
    cases = []
    for n,info,cutoff in ((32,tuple(i-32 for i in profile if i>=32),5),(64,profile,9)):
        baseline = min(1<<i.bit_count() for i in info)
        released = [j for j in range(min(info)+1,n) if j not in info and 1<<j.bit_count()>=baseline]
        mother = tuple(sorted(set(info)|set(released)))
        cases.append(dict(name=f'n{n}_mother',kind='directory',n=n,profile=mother,
                          cutoff=cutoff,checks=[1<<j for j in range(n) if j not in mother]))
    saved = load_code(ROOT/'results/n256_k128/support.json')
    columns = saved.columns[:128]+tuple(c & ~((1<<128)-1) for c in saved.columns[128:])
    fixed = PolarSubcode(256,saved.profile,columns)
    for source in [i for i in fixed.profile if i<128 and i.bit_count()==4][:2]:
        cases.append(dict(name=f'n256_source{source}',kind='extension',n=256,cutoff=18,
                          checks=fixed.checks,left_input=1<<source))
    return cases,fixed


def run(output,repetitions=3):
    if repetitions<3:
        raise ValueError('Use at least three alternating timed repetitions')
    emit = Progress(output)
    cases,fixed = fixtures()
    report = dict(hardware=hardware(),repetitions=repetitions,cases=[],
                  scope='Matched primitives, not a comparison against optimized prior implementations')
    for case in cases:
        output_words,stats = evaluate(case,'quotient')
        if case['kind']=='directory':
            native,_ = words(PolarSubcode(case['n'],case['profile'],(0,)*case['n']),case['cutoff'])
            assert set(output_words)=={0}|{u for u,_ in native}
        else:
            expected = RelativePlotkinSpectrum(fixed.profile,fixed.columns).inspect(case['left_input'],case['cutoff'])['spectrum']
            assert Counter(polar_transform(u,case['n']).bit_count() for u in output_words)==expected
        digest = checksum(output_words,case['n'])
        warmup,_ = evaluate(case,'direct')
        if checksum(warmup,case['n']) != digest or len(set(warmup)) != len(warmup):
            raise AssertionError('Untimed direct and quotient directories disagree')
        timing = {'direct':[],'quotient':[]}
        for repeat in range(repetitions):
            for method in (('direct','quotient') if repeat%2==0 else ('quotient','direct')):
                gc.collect()
                start = time.perf_counter()
                values,current = evaluate(case,method)
                timing[method].append(time.perf_counter()-start)
                if checksum(values,case['n'])!=digest or len(set(values))!=len(values):
                    raise AssertionError('Complete directories disagree or contain duplicates')
        memory = {}
        for method in timing:
            gc.collect()
            tracemalloc.start()
            values,_ = evaluate(case,method)
            _,memory[method] = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert checksum(values,case['n'])==digest
        record = dict(name=case['name'],kind=case['kind'],N=case['n'],cutoff=case['cutoff'],
                      output_words=len(output_words),sha256=digest,seconds=timing,
                      median_seconds={k:statistics.median(v) for k,v in timing.items()},
                      separate_python_traced_peak_bytes=memory,quotient_statistics=stats)
        report['cases'].append(record)
        write_json(emit.folder/'report.json',report)
        emit('benchmark','Both methods emitted identical complete input sets',case=case['name'],
             words=len(output_words),seconds=record['median_seconds'])
    return report
