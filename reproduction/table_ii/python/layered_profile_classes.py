"""Reliability-ordered profile classes with two layer counts and an exact tail.

These classes partition the declared row-floor family. They are proposal
classes, not necessary distance conditions. Costs may be channel costs or ranks.
"""

import heapq
import math

def _validate(n, k, floor, costs):
    if n < 2 or n & (n - 1) or not 1 <= k <= n:
        raise ValueError('Expected a power-of-two length and a nonzero dimension')
    if floor < 1 or floor & (floor - 1) or floor > n or len(costs) != n:
        raise ValueError('Invalid polar row floor or cost vector')
    if any(not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError('Costs must be finite and nonnegative')

def layer_label(profile, floor):
    low = [i for i in profile if 1 << i.bit_count() == floor]
    if not low:
        raise ValueError('The profile must attain the declared row floor')
    return (len(low), sum(1 << i.bit_count() == 2 * floor for i in profile), max(low))

def structural_signature(n, profile, floor):
    return dict(layer_counts_and_last=layer_label(profile, floor),
                suffix_dimensions=[sum(i >= n - length for i in profile)
                                   for length in (n // 2, n // 4, n // 8) if length])

def _domains(n, floor, last, costs):
    key = lambda i: (costs[i], i)
    return (sorted((i for i in range(last) if 1 << i.bit_count() == floor), key=key),
            sorted((i for i in range(n) if 1 << i.bit_count() == 2 * floor), key=key),
            sorted((i for i in range(n) if 1 << i.bit_count() > 2 * floor), key=key))

def ranked_layer_profiles(n, k, floor, count, next_count, last, costs):
    """Enumerate one class completely in nondecreasing additive cost order."""
    _validate(n, k, floor, costs)
    if not 0 <= last < n or 1 << last.bit_count() != floor:
        return
    domains = _domains(n, floor, last, costs)
    sizes = (count - 1, next_count, k - count - next_count)
    if any(q < 0 or q > len(domain) for q, domain in zip(sizes, domains)):
        return
    initial = tuple(tuple(range(q)) for q in sizes)

    def cost(state):
        return math.fsum([costs[last]] + [costs[domain[j]]
                         for domain, ranks in zip(domains, state) for j in ranks])

    heap, seen = [(cost(initial), initial)], {initial}
    while heap:
        total, state = heapq.heappop(heap)
        for side, domain in enumerate(domains):
            ranks = state[side]
            for j, value in enumerate(ranks):
                stop = ranks[j + 1] if j + 1 < len(ranks) else len(domain)
                if value + 1 >= stop:
                    continue
                advanced = ranks[:j] + (value + 1,) + ranks[j + 1:]
                child = state[:side] + (advanced,) + state[side + 1:]
                if child not in seen:
                    seen.add(child)
                    heapq.heappush(heap, (cost(child), child))
        profile = tuple(sorted([last] + [domain[j]
                               for domain, ranks in zip(domains, state) for j in ranks]))
        yield dict(profile=profile, cost=total, class_label=(count, next_count, last),
                   rank_state=state)

def layered_seed_profiles(n, k, floor, costs):
    """Yield the cost-minimal member of every nonempty class, without a grid.

    For fixed lowest-layer count and last index, the cost as a function of the
    next-layer count is discrete convex. A heap merges these one-dimensional
    frontiers without materializing the full Cartesian product of classes.
    """
    _validate(n, k, floor, costs)
    key = lambda i: (costs[i], i)
    low = [i for i in range(n) if 1 << i.bit_count() == floor]
    middle = sorted((i for i in range(n) if 1 << i.bit_count() == 2 * floor), key=key)
    high = sorted((i for i in range(n) if 1 << i.bit_count() > 2 * floor), key=key)
    upper = sorted(middle + high, key=key)

    def prefix(domain):
        out = [0.0]
        for i in domain:
            out.append(out[-1] + costs[i])
        return out

    pm, ph = prefix(middle), prefix(high)
    middle_counts = [0]
    for i in upper:
        middle_counts.append(middle_counts[-1] + (1 << i.bit_count() == 2 * floor))
    pools, prefixes = {}, {}
    heap, seen = [], set()

    def push(count, next_count, last):
        remaining = k - count
        if not max(0, remaining - len(high)) <= next_count <= min(len(middle), remaining):
            return
        state = count, next_count, last
        if state in seen:
            return
        seen.add(state)
        total = costs[last] + prefixes[last][count - 1] + pm[next_count] + ph[remaining - next_count]
        heapq.heappush(heap, (total, state))

    for index, last in enumerate(low):
        pools[last] = sorted(low[:index], key=key)
        prefixes[last] = prefix(pools[last])
        for count in range(max(1, k - len(upper)), min(k, index + 1) + 1):
            push(count, middle_counts[k - count], last)
    while heap:
        _, (count, next_count, last) = heapq.heappop(heap)
        push(count, next_count - 1, last)
        push(count, next_count + 1, last)
        profile = tuple(sorted([last] + pools[last][:count - 1] + middle[:next_count]
                               + high[:k - count - next_count]))
        yield dict(profile=profile, cost=math.fsum(costs[i] for i in profile),
                   class_label=(count, next_count, last))

def obstruction_neighbors(n, profile, target, length, protected_tail, costs, limit=32):
    """Propose scoped repairs, including transfers between row-weight layers.

    A caller must supply evidence of a failed domain. These proposals neither
    exclude the parent profile nor claim that a modified profile is feasible.
    """
    original = set(profile)
    floor = min(1 << i.bit_count() for i in profile)
    start, stop = n - length, n - protected_tail
    moves = []
    for old in profile:
        if not start <= old < stop or 1 << old.bit_count() >= target:
            continue
        for new in range(max(0, n - 2 * length), stop):
            if new in original or 1 << new.bit_count() < floor:
                continue
            if 1 << old.bit_count() == floor and 1 << new.bit_count() > floor:
                if sum(1 << i.bit_count() == floor for i in profile) == 1:
                    continue
            # Move an active low-weight source earlier, or remove it from that layer.
            if new >= old and new.bit_count() <= old.bit_count():
                continue
            moves.append((costs[new] - costs[old], old, new))
    rows = []
    for delta, old, new in heapq.nsmallest(limit, moves):
        candidate = tuple(sorted(original - {old} | {new}))
        rows.append(dict(profile=candidate, removed=old, added=new,
                         delta_cost=delta, mode='proved_domain_layer_repair',
                         feedback_length=length, preserved_tail=protected_tail,
                         structural_signature=structural_signature(n, candidate, floor),
                         full_profile_excluded=False))
    return rows
