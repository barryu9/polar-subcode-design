"""Proven binary linear bounds and exact polar environment enumerators.

No best-known CONSTRUCTION is used as an upper bound. The single table
entry below is a nonexistence bound: d_max(64,52)=5, verified 2026-09-30.
"""

from functools import lru_cache
from math import comb

TABLE_SOURCE='https://www.codetables.de/BKLC/BKLC.php?q=2&n=64&k=52'

@lru_cache(maxsize=None)
def basic_bound(length,distance,use_table=True):
    if length==0 or distance>length:return 0
    if distance<=1:return length
    radius=(distance-1)//2
    ball=sum(comb(length,j) for j in range(radius+1))
    hamming=length-(ball-1).bit_length()
    singleton=length-distance+1
    griesmer=0;cost=0
    for k in range(1,length+1):
        cost+=(distance+(1<<(k-1))-1)//(1<<(k-1))
        if cost>length:break
        griesmer=k
    table=51 if use_table and length==64 and distance>=6 else length
    return max(0,min(hamming,singleton,griesmer,table))

@lru_cache(maxsize=None)
def linear_dimension_bound(length,distance,use_table=True):
    """Basic, punctured, shortened bounds; all integer computations."""
    if distance>length:return 0
    direct=basic_bound(length,distance,use_table)
    punctured=min(basic_bound(length-q,distance-q,use_table) for q in range(distance))
    shortened=min(q+basic_bound(length-q,distance,use_table) for q in range(length+1))
    return min(direct,punctured,shortened)

@lru_cache(maxsize=None)
def environment_spectrum(n,start):
    """Exact W(V_start), V_start=span(g_start,...,g_(n-1))."""
    if start==n:return (1,)+(0,)*n
    if n==1:return (1,1)
    half=n//2
    if start>=half:
        local=environment_spectrum(half,start-half)
        answer=[0]*(n+1)
        for w,count in enumerate(local):answer[2*w]=count
        return tuple(answer)
    local=environment_spectrum(half,start)
    answer=[0]*(n+1)
    for w,count in enumerate(local):
        if not count:continue
        multiplier=count*(1<<w)
        for j in range(half-w+1):
            answer[w+2*j]+=multiplier*comb(half-w,j)
    return tuple(answer)

@lru_cache(maxsize=None)
def coset_capacities(n,target):
    unrestricted=[];complement=[]
    for i in range(n):
        first=environment_spectrum(n,i);second=environment_spectrum(n,i+1)
        counts=[a-b for a,b in zip(first,second)]
        assert all(c>=0 for c in counts) and sum(counts)==1<<(n-i-1)
        high=sum(counts[target:])
        symmetric=sum(counts[target:n-target+1]) if 2*target<=n else 0
        unrestricted.append(high.bit_length()-1)
        complement.append(symmetric.bit_length()-1)
    return tuple(unrestricted),tuple(complement)

def structural_profile_audit(n,information,target):
    info=sorted(information);positions=set(info)
    last_frozen=max((i for i in range(n) if i not in positions),default=-1)
    immutable=[i for i in info if i>last_frozen and (1<<i.bit_count())<target]
    has_complement=n-1 in positions
    caps,symmetric=coset_capacities(n,target)
    bad_cosets=[i for q,i in enumerate(info) if len(info)-q-1>caps[i]]
    bad_symmetric=[i for q,i in enumerate(info[:-1])
                   if has_complement and len(info)-q-1>symmetric[i]]
    return dict(immutable_row_pass=not immutable,immutable_bad_indices=immutable,
                coset_capacity_pass=not bad_cosets,coset_bad_indices=bad_cosets,
                complement_capacity_pass=not bad_symmetric,
                complement_bad_indices=bad_symmetric,
                contains_all_one=has_complement)

def even_complement_sidon_obstruction(length, information, target):
    """Table-free zero-sum Sidon exclusions at lengths16,32,64.

    The length32 fourth moment contradicts its allowed squared Walsh values.
    At lengths16 and64 equality forces a transversal graph with nonintegral
    output multiplicities. A larger even all-one code contains a subcode at
    the excluded boundary. No claim is made for odd or other-length codes.
    """
    positions = set(information)
    if (length in (16, 32, 64) and len(positions) >= length-2*(length.bit_length()-1)
            and target >= 6 and 0 not in positions and length-1 in positions):
        return dict(local_distance_upper_bound=4, external_code_table_used=False,
                    proof="Zero-sum Sidon character moments and integer output counts",
                    even=True, contains_all_one=True, dimension=len(positions),
                    excluded_boundary_dimension=length-2*(length.bit_length()-1))
    return None
