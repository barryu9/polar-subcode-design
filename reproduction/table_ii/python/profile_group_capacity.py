"""Exact existential grouping tests on bounded early causal check subsets.

For each dyadic suffix, examine the first min(8,M/2) input positions.
Include early frozen rows in increasing order up to 10 causal taps in total.
Enumerate EVERY assignment on the selected taps. Rows outside this subset
are given unrestricted labeling power. Online tests use error radii 1 and 2;
checkpoint audits can count the full required error ball with exact integers.
Rejection is necessary even when later rows have unrestricted causal supports.
Passing this bounded test does not certify support existence.
"""

from collections import Counter
from functools import lru_cache
from itertools import product
from math import comb

EARLY_WIDTH=8

TAP_BUDGET=10

def error_group_counts(groups,radius):
    """Exact integer error-ball counts by syndrome, grouped equal columns."""
    layers=[{} for _ in range(radius+1)];layers[0][0]=1
    for label,count in groups.items():
        updated=[{} for _ in range(radius+1)]
        for weight,values in enumerate(layers):
            for added in range(min(count,radius-weight)+1):
                multiplier=comb(count,added)
                delta=label if added%2 else 0
                for syndrome,value in values.items():
                    target=syndrome^delta
                    updated[weight+added][target]=updated[weight+added].get(target,0)+value*multiplier
        layers=updated
    total={}
    for values in layers:
        for syndrome,count in values.items():total[syndrome]=total.get(syndrome,0)+count
    return total

def early_width(length):
    return min(EARLY_WIDTH,max(1,length//2))

@lru_cache(maxsize=None)
def group_capacity(length,width,information_pattern,radius=1):
    if (length<2 or length&(length-1) or not 1<=width<=length
            or not 0<=information_pattern<1<<width):
        raise ValueError('Invalid early grouping pattern')
    info=[j for j in range(width) if information_pattern>>j&1]
    rows=[];taps=0
    for j in range(width):
        if j in info:continue
        sources=[i for i in info if i<j]
        if taps+len(sources)>TAP_BUDGET:continue
        rows.append((j,sources));taps+=len(sources)
    period=1<<(width-1).bit_length()
    repetitions=length//period
    # u=xG, so the parity row for u_j is G[:,j], the monomial
    # 1{coordinate & j == j}. This is a COLUMN of G, not G's row.
    monomials=[sum(int((c&j)==j)<<c for c in range(period)) for j in range(width)]
    choices=[]
    for j,sources in rows:
        options=[]
        for mask in range(1<<len(sources)):
            parity=monomials[j]
            for q,i in enumerate(sources):
                if mask>>q&1:parity^=monomials[i]
            options.append(parity)
        choices.append(options)
    best=-1;best_sizes=None;best_assignment=None;enumerated=0
    for checks in product(*choices):
        labels=[sum(((row>>c)&1)<<q for q,row in enumerate(checks)) for c in range(period)]
        groups={label:count*repetitions for label,count in Counter(labels).items()}
        # Nonzero labels have 2^remaining slots. The zero group has one less
        # because a final zero parity column would permit a weight-one word.
        if not 1<=radius<=length:raise ValueError('Invalid error radius')
        errors=dict(groups);errors[0]=errors.get(0,0)+1
        if radius==2:
            entries=list(groups.items())
            for q,(a,count) in enumerate(entries):
                errors[0]+=count*(count-1)//2
                for b,other in entries[q+1:]:
                    errors[a^b]=errors.get(a^b,0)+count*other
        elif radius>2:
            errors=error_group_counts(groups,radius)
        slots=max(errors.values())
        required=(slots-1).bit_length()
        cap=length-len(rows)-required
        enumerated+=1
        if cap>best:
            best=cap;best_sizes=sorted(groups.values(),reverse=True);best_assignment=list(checks)
    return dict(max_information=best,early_width=width,
                selected_frozen_rows=[j for j,_ in rows],causal_taps=taps,
                assignments_enumerated=enumerated,best_group_sizes=best_sizes,
                tested_error_radius=radius,
                best_periodic_parity_rows=best_assignment,period=period,
                scope='Exact existential test on selected early rows only; no full support certificate')

def full_ball_audit(length,width,pattern,distance,work_limit=1_000_000):
    radius=(distance-1)//2
    if radius<1:return dict(status='not_applicable',pass_required=True)
    bounded=group_capacity(length,width,pattern,min(2,radius))
    s=len(bounded['selected_frozen_rows'])
    estimate=(radius+1)**2*(1<<s)*max(s,1)*bounded['assignments_enumerated']
    if radius>2 and estimate>work_limit:
        return dict(status='unknown_budget',pass_required=None,tested_radius=min(2,radius),
                    required_radius=radius,estimated_operations=estimate)
    result=group_capacity(length,width,pattern,radius)
    return dict(status='exact_on_selected_rows',required_radius=radius,
                estimated_operations=estimate,**result)
