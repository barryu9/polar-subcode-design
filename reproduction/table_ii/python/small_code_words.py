from verify_causal_flag_theory import polar_transform

def all_words(profile, columns):
    result = []
    for message in range(1 << len(profile)):
        u = sum(((message >> q) & 1) << i for q, i in enumerate(profile))
        for j, support in enumerate(columns):
            if (u & support).bit_count() & 1:
                u |= 1 << j
        result.append((u, polar_transform(u, len(columns)).bit_count()))
    return result
