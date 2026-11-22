"""Profile-only distance ceilings from repeated suffixes and packing bounds."""

from math import comb

def suffix_bounds(n, profile):
    result = []
    length = n
    while length:
        local = [i-(n-length) for i in profile if i >= n-length]
        k = len(local)
        if k:
            even = 0 not in local
            effective_length = length-int(even)
            capacity = 1 << (effective_length-k)
            candidates = []
            for distance in range(2 if even else 1, length+1, 2 if even else 1):
                radius = (distance-1-int(even))//2
                volume = sum(comb(effective_length, j) for j in range(radius+1))
                if volume <= capacity:
                    candidates.append(distance)
            bound = max(candidates)
            record = dict(length=length, dimension=k, repetitions=n//length,
                even_for_every_causal_support=even, punctured_length=effective_length,
                packing_capacity=capacity, local_upper_bound=bound,
                full_upper_bound=(n//length)*bound, profile=local)
            next_distance = bound+(2 if even else 1)
            if next_distance <= length:
                radius = (next_distance-1-int(even))//2
                record.update(first_excluded_local_distance=next_distance,
                    excluded_radius=radius,
                    excluded_ball_volume=sum(comb(effective_length,j) for j in range(radius+1)))
            else:
                record['terminal_reason'] = 'Local distance cannot exceed local length'
            result.append(record)
        length //= 2
    return result
