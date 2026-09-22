"""Exact disjoint Plotkin-coset counting with shortening multiplicities."""
from collections import Counter
from math import comb
import time

from polar_subcodes.algebra import check_rows, rref
from polar_subcodes.code import PolarSubcode
from polar_subcodes.io import write_json
from .backend import count, words
from .relative import RelativePlotkinSpectrum
from .meet_in_middle import inspect_coset


def dual_spectrum(code, max_dimension=16):
    """Integer MacWilliams transform after complete dual enumeration."""
    checks = check_rows(code.profile, code.columns)
    if len(checks) > max_dimension:
        raise ValueError('Dual dimension exceeds the configured enumeration limit')
    if len(rref(checks)) != len(checks):
        raise AssertionError('Dependent physical checks')
    values = [0]
    for row in checks:
        values.extend([word ^ row for word in values])
    histogram = Counter(v.bit_count() for v in values)
    result = {}
    for w in range(code.n+1):
        total = 0
        for j, multiplicity in histogram.items():
            polynomial = sum((-1)**q*comb(j,q)*comb(code.n-j,w-q)
                             for q in range(max(0,w-code.n+j),min(w,j)+1))
            total += multiplicity*polynomial
        coefficient, remainder = divmod(total,len(values))
        if remainder or coefficient < 0:
            raise AssertionError('Invalid MacWilliams coefficient')
        if coefficient:
            result[w] = coefficient
    if sum(result.values()) != 1 << code.k:
        raise AssertionError('Wrong total number of codewords')
    return result


def spectrum(code, cutoff, max_work=100_000_000, emit=None, output=None):
    """Count, without materializing all full-code words, through the cutoff.

    Complete left directories and exact right spectra are required at each
    split. High-radius nodes fall back to the complete native counter.
    Work-limit failures propagate and never become zero coefficients.
    """
    if type(cutoff) is not int or not 0 <= cutoff <= code.n:
        raise ValueError('Cutoff must lie in [0,N]')
    start = time.perf_counter()
    if cutoff == 0 or code.k == 0:
        return {0:1}
    if code.n-code.k <= 16:
        result = {w:a for w,a in dual_spectrum(code).items() if w <= cutoff}
        if emit:
            emit('enumeration','Complete dual enumeration and integer MacWilliams transform',
                 N=code.n,K=code.k,cutoff=cutoff,spectrum=result)
        return result
    if code.n <= 8:
        return count(code,cutoff,max_work)['spectrum']
    model = RelativePlotkinSpectrum(code.profile,code.columns)
    right_code = PolarSubcode(model.half,model.right_profile,model.right_columns)
    left_code = PolarSubcode(model.half,model.left_profile,model.left_columns)
    left_cutoff = min(cutoff, model.half)
    left, metadata = words(left_code,left_cutoff,max_work)
    independent = count(left_code,left_cutoff,max_work)['spectrum']
    if independent != metadata['spectrum']:
        raise AssertionError('Left word directory and scalar counter disagree')
    maximum_radius = max(((cutoff-w)//2 for _,w in left),default=0)
    if maximum_radius > 6:
        if emit:
            emit('enumeration','Radius exceeds subset-join implementation, using exact native recursion',
                 N=code.n,radius=maximum_radius)
        return count(code,cutoff,max_work)['spectrum']
    right = spectrum(right_code,cutoff//2,max_work,emit,output)
    total = Counter({2*w:a for w,a in right.items() if 2*w <= cutoff})
    last = time.perf_counter()
    for index,(u,weight) in enumerate(left,1):
        item = (model.inspect(u,cutoff) if (cutoff-weight)//2 <= 4
                else inspect_coset(model,u,cutoff))
        total.update(item['spectrum'])
        now = time.perf_counter()
        if emit and (now-last >= 10 or index == len(left)):
            emit('enumeration','Disjoint left cosets counted with exact kernel multiplicities',
                 N=code.n,completed=index,total=len(left))
            last = now
    result = dict(sorted(total.items()))
    if output:
        write_json(output/f'spectrum_n{code.n}_d{cutoff}.json',dict(exact=True,
                   spectrum=result,left_spectrum=independent,left_cosets=len(left),
                   pure_right_spectrum={2*w:a for w,a in right.items() if 2*w<=cutoff},
                   seconds=time.perf_counter()-start))
    return result
