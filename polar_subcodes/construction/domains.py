"""Choose a released-row domain and retained core from the profile."""

def initial_structure(n, profile, distance):
    profile = tuple(sorted(profile))
    if (n < 8 or n & (n-1) or len(set(profile)) != len(profile)
            or not profile or any(i < 0 or i >= n for i in profile)
            or not 1 <= distance <= n):
        raise ValueError("Invalid length, profile, or distance")
    baseline = min(1 << i.bit_count() for i in profile)
    rows = tuple(j for j in range(profile[0]+1, n)
                 if j not in profile and (1 << j.bit_count()) >= baseline)
    core = tuple(i for i in profile if (1 << i.bit_count()) >= distance)
    coordinates = tuple((q, i) for q, j in enumerate(rows)
                        for i in profile if i < j and i not in core)
    return dict(baseline=baseline, rows=rows, core=core,
                coordinates=coordinates)
