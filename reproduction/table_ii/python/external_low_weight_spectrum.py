"""Adapter to the user-supplied exact Algorithm1 v0.1.0, not its GA fixtures.

Distance>=target is certified by an exact D=target-1 spectrum with no nonzero
terms. WorkLimitError means UNKNOWN, never pass. No actor/SCL inputs change.
"""

from pathlib import Path
import sys
import time

VENDOR=Path(__file__).resolve().parents[3]/'vendor/polar-low-weight-spectrum'

if str(VENDOR/'python') not in sys.path:sys.path.insert(0,str(VENDOR/'python'))

import os
if __package__:
    from .native_paths import library_path
else:
    from native_paths import library_path
os.environ.setdefault("POLAR_SPECTRUM_LIBRARY", str(library_path("_candidate_block_neighborhood_kernel")))
from polar_spectrum import algorithm1,WorkLimitError

def checks_from_construction(actions,supports,k=None):
    n=len(actions)
    if n<1 or n>256 or n&(n-1) or len(supports)!=n:raise ValueError('Expected power-of-two N<=256')
    prior=set();checks=[]
    for t,(action,support) in enumerate(zip(actions,supports)):
        if type(action) is not int or action not in (0,1,2):raise ValueError('Invalid direct action')
        if any(type(i) is not int for i in support) or len(set(support))!=len(support):
            raise ValueError('Invalid support')
        if action==0:
            if support:raise ValueError('Information action cannot have support')
            prior.add(t)
        else:
            if bool(support)!=(action==2) or not set(support)<=prior:raise ValueError('Invalid causal support')
            checks.append((1<<t)^sum(1<<i for i in support))
    if k is not None and len(prior)!=k:raise ValueError('Wrong dimension')
    return tuple(checks),len(prior)

def spectrum(code,cutoff=11,max_work=1000000,backend='cpp'):
    tick=time.perf_counter();checks,k=checks_from_construction(code['actions'],code['supports'])
    answer=algorithm1(len(code['actions']),checks,cutoff,backend=backend,max_work=max_work)
    if (not answer['exact'] or answer['spectrum'].get(0)!=1 or
            answer['transmitted_dimension']!=k or answer['kernel_dimension']!=0):
        raise RuntimeError('Spectrum dimension/zero-word audit failed')
    answer['adapter_seconds']=time.perf_counter()-tick
    return answer
