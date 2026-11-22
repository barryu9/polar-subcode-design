"""Exact joint support edits of two frozen positions from a full supercode."""

import ctypes
import json
import random
from pathlib import Path
import numpy as np
from external_low_weight_spectrum import WorkLimitError
from native_paths import library_path

_LIB=None

_CAUSAL_LIB=None

class BlockCatalogue:
    def __init__(self,design,columns,rows,cutoff=16,max_work=2000000,causal_leaders_only=False):
        global _LIB, _CAUSAL_LIB
        if len(set(rows))!=len(rows) or not 1<=len(rows)<=8:raise ValueError('Invalid removed rows')
        lib = _CAUSAL_LIB if causal_leaders_only else _LIB
        if lib is None:
            stem = '_candidate_block_neighborhood_kernel' if causal_leaders_only else '_block_neighborhood_kernel'
            lib=ctypes.CDLL(str(library_path(stem)))
            ptr=ctypes.POINTER(ctypes.c_uint8)
            lib.block_neighborhood.argtypes=[ctypes.c_int,ptr,ptr,ptr,ctypes.c_int,ctypes.c_uint64]
            lib.block_neighborhood.restype=ctypes.c_void_p
            lib.polar_spectrum_free.argtypes=[ctypes.c_void_p];lib.polar_spectrum_free.restype=None
            if causal_leaders_only: _CAUSAL_LIB = lib
            else: _LIB = lib
        n=design.n
        if any(j not in design.rows for j in rows):raise ValueError('Expected mutable frozen positions')
        info=(ctypes.c_uint8*n)(*(int(i in design.profile) for i in range(n)))
        raw=(ctypes.c_uint8*(n*n))(*(columns[j]>>i&1 for j in range(n) for i in range(n)))
        removed=(ctypes.c_uint8*n)(*(int(j in rows) for j in range(n)))
        pointer=lib.block_neighborhood(n,info,raw,removed,cutoff,max_work)
        if not pointer:raise MemoryError('Native allocation failed')
        try:result=json.loads(ctypes.string_at(pointer))
        finally:lib.polar_spectrum_free(pointer)
        if not result['exact']:raise (WorkLimitError if 'work limit' in result['error'] else RuntimeError)(result['error'])
        self.design,self.columns,self.rows,self.cutoff=design,list(columns),tuple(rows),cutoff
        self.words=[(int(u,2),w) for u,w in result.pop('words')];self.stats=result
        assert len({u for u,w in self.words})==len(self.words)
        def packed(values):return int.from_bytes(np.packbits(values,bitorder='little').tobytes(),'little')
        weights=np.asarray([w for u,w in self.words])
        self.target_mask=packed(weights==cutoff);self.low_mask=packed((weights>0)&(weights<cutoff))
        array=np.frombuffer(b''.join(u.to_bytes(n//8,'little') for u,w in self.words),dtype=np.uint8).reshape(-1,n//8)
        bits=np.unpackbits(array,axis=1,bitorder='little')
        all_planes={i:packed(bits[:,i]) for i in set(design.profile)|set(rows)}
        self.planes={i:all_planes[i] for i in design.profile if i<max(rows)}
        self.defects=[]
        for j in rows:
            defect=all_planes[j];support=columns[j]
            while support:
                bit=support&-support;defect^=all_planes[bit.bit_length()-1];support^=bit
            self.defects.append(defect)
        self.domain=[(b,i) for b,j in enumerate(rows) for i in design.profile if i<j]
        assert self.score([0]*len(rows))==(result['parent_target'],result['parent_low'])

    def errors(self,deltas):
        if len(deltas)!=len(self.rows):raise ValueError('Wrong delta count')
        result=list(self.defects)
        for b,delta in enumerate(deltas):
            while delta:
                bit=delta&-delta;i=bit.bit_length()-1
                if (b,i) not in self.domain:raise ValueError('Noncausal block edit')
                result[b]^=self.planes[i];delta^=bit
        return result

    def score_errors(self,errors):
        excluded=0
        for e in errors:excluded|=e
        return ((~excluded&self.target_mask).bit_count(),(~excluded&self.low_mask).bit_count())

    def score(self,deltas):return self.score_errors(self.errors(deltas))

    def child(self,deltas):
        self.errors(deltas);result=list(self.columns)
        for j,de in zip(self.rows,deltas):result[j]^=de
        return result

def optimize_pair(c,seed,starts=8,steps=12):
    if len(c.rows)!=2:raise ValueError('Pair optimizer expects two rows')
    rng=random.Random(seed)
    moves=[]
    active=[(b,i) for b,i in c.domain if c.planes[i]]
    singles=[]
    for b,i in active:
        ds=[0,0];ps=[0,0];ds[b]=1<<i;ps[b]=c.planes[i]
        singles.append((ds[0],ds[1],ps[0],ps[1]))
    moves=singles+[(a[0]^b[0],a[1]^b[1],a[2]^b[2],a[3]^b[3])
                   for index,a in enumerate(singles) for b in singles[index+1:]]
    best=[0,0];best_value,best_low=c.score(best);assert not best_low
    evaluated=0
    for start in range(starts):
        deltas=[0,0]
        if start:
            radius=min(len(active),(1,2,4,8,16)[(start-1)%5])
            for b,i in rng.sample(active,radius):deltas[b]^=1<<i
        e0,e1=c.errors(deltas);value,low=c.score_errors([e0,e1])
        for _ in range(steps):
            best_key=(low,value);winning=None
            for d0,d1,p0,p1 in moves:
                excluded=(e0^p0)|(e1^p1)
                val=(~excluded&c.target_mask).bit_count();lo=(~excluded&c.low_mask).bit_count();evaluated+=1
                if (lo,val)<best_key:best_key=(lo,val);winning=(d0,d1,p0,p1,val,lo)
            if winning is None:break
            d0,d1,p0,p1,value,low=winning;deltas[0]^=d0;deltas[1]^=d1;e0^=p0;e1^=p1
        if not low and (value,sum(x.bit_count() for x in deltas))<(best_value,sum(x.bit_count() for x in best)):
            best,best_value=deltas.copy(),value
    return dict(deltas=best,A16=best_value,evaluated=evaluated)
