"""Exact arbitrary single-row support edits from one supercode catalogue.

Removing frozen equation j gives C+<g_j>. Any replacement support for that
row is a codimension-one subcode, so filtering all low-weight supercode words
gives the COMPLETE low-weight spectrum of every such replacement, including
words absent from the parent. Work exhaustion is UNKNOWN, never a partial score.
"""

import ctypes
from native_paths import library_path
import json
import random
from pathlib import Path
from external_low_weight_spectrum import WorkLimitError

_LIB=None

def packed_mask(size, indices):
    """Build a bit plane without repeatedly allocating growing integers."""
    buffer = bytearray((size + 7) // 8)
    for k in indices:
        buffer[k >> 3] |= 1 << (k & 7)
    return int.from_bytes(buffer, 'little')

def library():
    global _LIB
    if _LIB is None:
        _LIB=ctypes.CDLL(str(library_path('_row_neighborhood_kernel')))
        ptr=ctypes.POINTER(ctypes.c_uint8)
        _LIB.row_neighborhood.argtypes=[ctypes.c_int,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint64]
        _LIB.row_neighborhood.restype=ctypes.c_void_p
        _LIB.polar_spectrum_free.argtypes=[ctypes.c_void_p]
        _LIB.polar_spectrum_free.restype=None
    return _LIB

class RowCatalogue:
    def __init__(self, design, columns, row, cutoff=16, max_work=2000000):
        n=design.n;lib=library()
        infos=(ctypes.c_uint8*n)(*(int(i in design.profile) for i in range(n)))
        raw=(ctypes.c_uint8*(n*n))(*(int(columns[j])>>i&1 for j in range(n) for i in range(n)))
        pointer=lib.row_neighborhood(n,infos,raw,row,cutoff,max_work)
        if not pointer:raise MemoryError('Native allocation failed')
        try:result=json.loads(ctypes.string_at(pointer))
        finally:lib.polar_spectrum_free(pointer)
        if not result['exact']:
            raise (WorkLimitError if 'work limit' in result['error'] else RuntimeError)(result['error'])
        self.design,self.columns,self.row,self.cutoff=design,list(columns),row,cutoff
        self.domain=tuple(i for i in design.profile if i<row)
        self.words=[(int(u,2),w) for u,w in result.pop('words')]
        self.stats=result
        if len({u for u,w in self.words})!=len(self.words):raise RuntimeError('Duplicate supercode word')
        size = len(self.words)
        self.target_mask=packed_mask(size, (k for k,(u,w) in enumerate(self.words) if w==cutoff))
        self.low_mask=packed_mask(size, (k for k,(u,w) in enumerate(self.words) if 0<w<cutoff))
        self.defect=packed_mask(size, (k for k,(u,w) in enumerate(self.words)
                                     if (u>>row&1)^((u&columns[row]).bit_count()&1)))
        self.planes={i:packed_mask(size, (k for k,(u,w) in enumerate(self.words) if u>>i&1))
                     for i in self.domain}
        assert self.score(0)==(result['parent_target'],result['parent_low'])

    def error(self,delta):
        result=self.defect
        while delta:
            bit=delta&-delta;i=bit.bit_length()-1
            if i not in self.planes:raise ValueError('Noncausal row edit')
            result^=self.planes[i];delta^=bit
        return result

    def score(self,delta):
        return self.score_error(self.error(delta))

    def score_error(self,error):
        return ((~error&self.target_mask).bit_count(),(~error&self.low_mask).bit_count())

    def child(self,delta):
        self.error(delta)
        result=list(self.columns);result[self.row]^=delta;return result

    def witnesses(self,delta):
        selected=~self.error(delta)&self.target_mask
        return [u for k,(u,w) in enumerate(self.words) if selected>>k&1]

def optimize_row(catalogue,seed,starts=8,steps=12):
    """Exact-scored multistart one/two-bit descent inside one frozen row.

    A local optimum is only a one/two-bit optimum in this row, not a proof
    of global optimality or impossibility. Every returned score is exact.
    """
    rng=random.Random(seed)
    active=[i for i in catalogue.domain if catalogue.planes[i]]
    moves=[(1<<i,catalogue.planes[i]) for i in active]
    moves += [((1<<i)|(1<<j),catalogue.planes[i]^catalogue.planes[j])
              for index,i in enumerate(active) for j in active[index+1:]]
    best_delta=0;best_value,best_low=catalogue.score(0)
    assert not best_low
    evaluated=1
    minima=[]
    for start in range(starts):
        radius=0 if start==0 else min(len(active),(1,2,4,8,16)[(start-1)%5])
        delta=sum(1<<i for i in rng.sample(active,radius))
        error=catalogue.error(delta);value,low=catalogue.score_error(error)
        for _ in range(steps):
            winning=None;best_key=(low,value)
            for toggle,plane in moves:
                candidate=catalogue.score_error(error^plane);evaluated+=1
                key=(candidate[1],candidate[0])
                if key<best_key:
                    best_key=key;winning=(toggle,plane,candidate)
            if winning is None:break
            toggle,plane,(value,low)=winning;delta^=toggle;error^=plane
        minima.append((value,low,delta))
        if not low and (value,delta.bit_count())<(best_value,best_delta.bit_count()):
            best_delta,best_value=delta,value
    # Random exact-neutral displacement supplies cheap movement on a plateau.
    neutral=[]
    inactive=[i for i in catalogue.domain if not catalogue.planes[i]]
    if inactive:
        neutral.append(sum(1<<i for i in rng.sample(inactive,min(len(inactive),rng.choice((1,2,4,8))))))
    for _ in range(64):
        radius=min(len(catalogue.domain),rng.choice((1,2,4,8)))
        delta=sum(1<<i for i in rng.sample(catalogue.domain,radius))
        if delta and catalogue.score(delta)==catalogue.score(0):neutral.append(delta)
    return dict(delta=best_delta,A16=best_value,evaluated=evaluated,
                neutral_delta=rng.choice(neutral) if neutral else None,
                local_minima=[dict(A16=v,low=lo,delta=de) for v,lo,de in minima])
