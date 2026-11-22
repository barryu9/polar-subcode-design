"""Deterministic additive-cost coverage inside a declared polar profile class.

Class labels are proposal preferences, not distance-feasibility conditions.
All combination successors are generated before filtering the maximum index.
No reference profile or support is an input to this iterator.
"""

import heapq
import math

def ranked_class_profiles(n,k,floor,count,bucket,costs):
    if n<8 or n & (n-1) or len(costs)!=n or not 0<=bucket<8:
        raise ValueError('Expected power-of-two length, a cost per index, and one of 8 buckets')
    if any(not math.isfinite(x) or x<0 for x in costs):
        raise ValueError('Costs must be finite and nonnegative')
    width = n//8
    low = sorted((i for i in range((bucket+1)*width) if 1<<i.bit_count()==floor),
                 key=lambda i:(costs[i],i))
    high = sorted((i for i in range(n) if 1<<i.bit_count()>floor),
                  key=lambda i:(costs[i],i))
    if not 1<=count<=min(k,len(low)) or not 0<=k-count<=len(high):
        return
    initial = (tuple(range(count)),tuple(range(k-count)))

    def cost(state):
        return math.fsum([costs[low[j]] for j in state[0]]+
                         [costs[high[j]] for j in state[1]])

    heap = [(cost(initial),initial)]
    seen = {initial}
    while heap:
        total,state = heapq.heappop(heap)
        for side,domain in enumerate((low,high)):
            ranks = state[side]
            for j,value in enumerate(ranks):
                stop = ranks[j+1] if j+1<len(ranks) else len(domain)
                if value+1>=stop:
                    continue
                advanced = ranks[:j]+(value+1,)+ranks[j+1:]
                child = (advanced,state[1]) if side==0 else (state[0],advanced)
                if child not in seen:
                    seen.add(child)
                    heapq.heappush(heap,(cost(child),child))
        selected_low = [low[j] for j in state[0]]
        if max(selected_low)//width!=bucket:
            continue
        profile = tuple(sorted(selected_low+[high[j] for j in state[1]]))
        yield dict(profile=profile,cost=total,rank_state=state,
                   class_label=[count,bucket],generated_states=len(seen))
