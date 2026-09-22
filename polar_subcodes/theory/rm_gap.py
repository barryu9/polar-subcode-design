"""Check containment before applying the classical RM weight gap."""

def rm_gap_membership(profile, columns, target):
    n = len(columns)
    m = n.bit_length()-1
    threshold = min(i.bit_count() for i in profile)
    baseline = 1 << threshold
    order = m-threshold
    if n != 1 << m or not 2 <= order <= m-2 or not baseline < target <= 3*baseline//2:
        raise ValueError('Unsupported minimum-to-second-weight interval')
    active = set(profile) | {j for j, c in enumerate(columns) if c}
    if any(j.bit_count() < threshold for j in active):
        raise ValueError('Candidate is outside the claimed RM mother')
    return dict(N=n, order=order, variables=m, minimum_weight=baseline,
        next_possible_weight=3*baseline//2, member=True,
        excluded_open_interval=[baseline, 3*baseline//2],
        justification='All active polar inputs have index weight >= m-r; the RM low-weight theorem excludes (d,3d/2).',
        theorem_doi='10.1109/TIT.1970.1054545',
        accessible_source='https://arxiv.org/pdf/2301.13497', source_location='Theorem 2; Section 5.2 for RM(3,8)')
