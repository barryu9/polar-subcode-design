"""Independent radius-zero/one checks of saved Plotkin construction trials.

The complete constituent catalogues are supplied by the enumeration backends.
This auditor uses direct syndrome elimination, not the construction quotient API.
"""

from collections import Counter
import json
import time

def transform(value, n):
    bits = [(value >> i) & 1 for i in range(n)]
    step = 1
    while step < n:
        for start in range(0, n, 2 * step):
            for j in range(step):
                bits[start+j] ^= bits[start+j+step]
        step *= 2
    return sum(bit << i for i, bit in enumerate(bits))

def reduce_by(value, basis):
    for pivot in sorted(basis, reverse=True):
        if value >> pivot & 1:
            value ^= basis[pivot]
    return value

def audit_pair(n, profile, columns, catalogue, cutoff):
    half = n // 2
    info = set(profile)
    left_info = {i for i in info if i < half}
    right_info = {i-half for i in info if i >= half}
    right_checks = [(1 << j) ^ (columns[half+j] >> half)
                    for j in range(half) if j not in right_info]

    def syndrome(word):
        u = transform(word, half)
        return sum(((u & check).bit_count() & 1) << q
                   for q, check in enumerate(right_checks))

    h_columns = [syndrome(1 << j) for j in range(half)]
    violations, ranks, radii, weights = [], Counter(), Counter(), Counter()
    seen = set()
    for u, weight in catalogue:
        assert u not in seen and 0 < u < 1 << half
        seen.add(u)
        assert all(((u >> j) ^ (u & columns[j]).bit_count()) % 2 == 0
                   for j in range(half) if j not in left_info)
        a = transform(u, half)
        assert a.bit_count() == weight and 0 < weight <= cutoff
        radius = (cutoff-weight) // 2
        if radius > 1:
            raise ValueError('This independent auditor supports only radius zero or one')
        cross = sum(((u & columns[half+j]).bit_count() & 1) << j
                    for j in range(half))
        target = syndrome(transform(cross, half))
        basis = {}
        for j in range(half):
            if a >> j & 1:
                value = reduce_by(h_columns[j], basis)
                if value:
                    basis[value.bit_length()-1] = value
        label = reduce_by(target, basis)
        bad = label == 0
        if radius == 1:
            bad |= any(label == reduce_by(h_columns[j], basis)
                       for j in range(half) if not (a >> j & 1))
        if bad:
            violations.append(u)
        ranks[len(basis)] += 1
        radii[radius] += 1
        weights[weight] += 1
    return dict(cosets=len(seen), violations=violations, support_image_ranks=dict(ranks),
                outside_radii=dict(radii), left_weights=dict(weights),
                method='Direct parity checks and binary span membership')

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def audit(folder):
    started = time.perf_counter()
    result = load(folder/'result.json')
    assert result['status'] == 'verified_feasible'
    details = []
    previous = None
    lower = None
    for record in result['stages']:
        n, k = record['N'], record['K']
        stage = folder/f'n{n}'
        design = load(stage/'support.json')
        info, columns = design['profile'], design['columns']
        assert len(info) == k and len(columns) == n
        mask = sum(1 << i for i in info)
        assert all(c >= 0 and not (c >> j) and not (c & ~mask)
                   and not (j in info and c) for j, c in enumerate(columns))
        if record['status'] == 'static_sufficient':
            assert not any(columns)
            lower = min(1 << i.bit_count() for i in info)
            details.append(dict(N=n, distance_lower_bound=lower, proof='Static polar distance'))
        elif record['status'] == 'local_domain_completed':
            checked = load(stage/'independent_enumeration.json')
            assert checked['profile'] == info and checked['columns'] == columns
            assert checked['cpp']['exact'] and checked['python']['exact']
            assert checked['cpp']['spectrum'] == checked['python']['spectrum'] == {'0':1}
            lower = record['target']
            details.append(dict(N=n, distance_lower_bound=lower, proof='Two complete enumeration backends'))
        else:
            half = n//2
            assert previous['N'] == half
            assert previous['profile'] == [i-half for i in info if i >= half]
            assert previous['columns'] == [c >> half for c in columns[half:]]
            assert not any(columns[:half])
            left_dp = min(1 << i.bit_count() for i in info if i < half)
            if record['status'] == 'zero_outer_sufficient':
                assert not any(c & ((1 << half)-1) for c in columns[half:])
                lower = min(left_dp, 2*lower)
                details.append(dict(N=n, distance_lower_bound=lower, proof='Plotkin distance bound'))
            else:
                cutoff = record['target']-1
                assert 2*lower > cutoff
                catalogue = load(stage/'outer'/'left_catalogue.json')
                counted = Counter(w for _, w in catalogue)
                verified = load(stage/'outer'/'left_verified.json')
                assert verified['profile'] == [i for i in info if i < half]
                assert verified['columns'] == columns[:half]
                assert verified['cpp']['exact'] and verified['python']['exact']
                assert verified['cpp']['spectrum'] == verified['python']['spectrum']
                assert {int(w):a for w,a in verified['cpp']['spectrum'].items()} == {0:1, **counted}
                pair = audit_pair(n, info, columns, catalogue, cutoff)
                assert not pair['violations']
                lower = record['target']
                details.append(dict(N=n, distance_lower_bound=lower, **pair))
        assert lower >= record['target']
        previous = design
    final = load(folder/'support.json')
    assert final['columns'] == previous['columns'] and final['profile'] == previous['profile']
    words = [(i, transform((1 << i) | sum(((c >> i) & 1) << j
              for j,c in enumerate(final['columns'])), final['N'])) for i in final['profile']]
    i, witness = min(words, key=lambda row: row[1].bit_count())
    upper = witness.bit_count()
    assert lower <= upper
    report = dict(passed=True, distance_lower_bound=lower, distance_upper_bound=upper,
                  dmin=lower if lower == upper else None, stages=details,
                  witness=dict(information_position=i, word_hex=hex(witness), weight=upper),
                  independent_of_construction_quotient_code=True,
                  scope='Complete saved left catalogues, direct syndrome exclusions, and recursive distance bounds',
                  seconds=time.perf_counter()-started)
    (folder/'independent_syndrome_audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report
