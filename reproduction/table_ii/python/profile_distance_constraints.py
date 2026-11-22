"""Necessary linear-code distance bounds on repeated polar suffix subcodes.

These bounds reject impossible profiles but do not certify existence of
distance-target causal supports. Log-DP conditions the GA subset prior on
all bounds and supplies prefix-safe legal actions.
"""

from math import comb
from profile_group_capacity import group_capacity, early_width, full_ball_audit
from profile_linear_bounds import linear_dimension_bound, coset_capacities, structural_profile_audit, TABLE_SOURCE

def suffix_caps(n,target,use_table=True):
    if n<2 or n&(n-1) or not 1<=target<=n:
        raise ValueError('Invalid length or distance target')
    records=[]
    length=n
    while length:
        repeats=n//length
        distance=(target+repeats-1)//repeats
        radius=(distance-1)//2
        ball=sum(comb(length,j) for j in range(min(radius,length)+1))
        hamming=length-(ball-1).bit_length()
        singleton=max(0,length-distance+1)
        griesmer=0
        cost=0
        for k in range(1,length+1):
            cost+=(distance+(1<<(k-1))-1)//(1<<(k-1))
            if cost>length:break
            griesmer=k
        cap=min(max(0,min(length,hamming,singleton,griesmer)),linear_dimension_bound(length,distance,use_table))
        records.append(dict(start=n-length,length=length,repetitions=repeats,
                            required_distance=distance,hamming_cap=hamming,
                            singleton_cap=singleton,griesmer_cap=griesmer,max_information=cap,
                            bound_sources=['integer puncturing/shortening of basic bounds']+
                            ([TABLE_SOURCE] if use_table and length==64 and distance>=6 else [])))
        length//=2
    return records

def column_capacity_cap(length, first):
    """Necessary K upper bound for distinct parity-check columns (d >= 3).

    first=0 has no leading static segment, so the stated layout condition
    does not apply. Empty suffixes are checked separately.
    """
    if first == 0:
        return length
    return length-first+(first-1).bit_length()-(length.bit_length()-1)

def profile_constraint_audit(n, information, target,group_capacity_enabled=True,structural_enabled=True,use_table=True):
    info=sorted(information)
    records=[]
    for bound in suffix_caps(n,target,use_table):
        local=[i-bound['start'] for i in info if i>=bound['start']]
        first=local[0] if local else None
        cap=(column_capacity_cap(bound['length'],first)
             if first is not None and bound['required_distance']>=3 else bound['length'])
        even=bool(local) and first>0
        effective=bound['required_distance']+int(even and bound['required_distance']%2==1)
        parity_cap=linear_dimension_bound(bound['length'],effective,use_table)
        single_pass=(len(local)!=bound['length']-1 or bound['required_distance']<2 or first!=0)
        grouping=None
        if group_capacity_enabled and local and bound['required_distance']>=3:
            width=early_width(bound['length'])
            pattern=sum(1<<i for i in local if i<width)
            grouping=group_capacity(bound['length'],width,pattern,min(2,(effective-1)//2))
        records.append(dict(**bound,information_count=len(local),first_information=first,
                            column_capacity_cap=cap,
                            distance_capacity_pass=len(local)<=bound['max_information'],
                            column_capacity_pass=len(local)<=cap,
                            even_subcode=even,effective_distance=effective,
                            parity_capacity_cap=parity_cap,
                            parity_capacity_pass=not structural_enabled or len(local)<=parity_cap,
                            single_check_pass=not structural_enabled or single_pass,
                            group_capacity=grouping,
                            group_capacity_pass=grouping is None or len(local)<=grouping['max_information']))
    return records

def full_group_profile_audit(n,information,target,use_table=True):
    """Checkpoint-only higher-order tests; a budget limit never rejects."""
    records=[]
    for b in profile_constraint_audit(n,information,target,use_table=use_table):
        if b['group_capacity'] is None:continue
        width=b['group_capacity']['early_width']
        pattern=sum(1<<(i-b['start']) for i in information if b['start']<=i<b['start']+width)
        result=full_ball_audit(b['length'],width,pattern,b['effective_distance'])
        passed=(b['information_count']<=result['max_information'] if 'max_information' in result else None)
        records.append(dict(length=b['length'],information_count=b['information_count'],pass_required=passed,
                            details=result))
    return dict(status='not_excluded' if all(r['pass_required'] is not False for r in records) else 'proven_excluded',
                conditions=records,distance_certified=False)
