"""Exact low-weight counts from punctured right-code cosets, without an RM domain."""
from collections import Counter
from itertools import combinations

from polar_subcodes.algebra import check_rows
from polar_subcodes.algebra import polar_transform
from polar_subcodes.algebra import solve


def causal_containment_domain(profile, mandatory_inputs, length):
    """Canonical supports whose code contains the specified polar-input vectors."""
    rows, dimension, non_rm_dimension = {}, 0, 0
    degree = min(i.bit_count() for i in profile)
    for j in range(length):
        if j in profile:
            continue
        prior = tuple(i for i in profile if i < j)
        equations = [(sum(((u >> i) & 1) << q for q, i in enumerate(prior)),
                      (u >> j) & 1) for u in mandatory_inputs]
        solution = solve(equations, len(prior))
        if solution is None:
            return dict(consistent=False, obstruction_row=j)
        origin, basis = solution
        dimension += len(basis)
        if j.bit_count() < degree:
            non_rm_dimension += len(basis)
        rows[j] = dict(prior=prior, origin=origin, basis=basis)
    return dict(consistent=True, dimension=dimension,
                non_rm_row_freedom=non_rm_dimension, rows=rows)


def small_error_counts(columns, syndrome, radius):
    """Count distinct-coordinate supports with repeated syndrome columns allowed."""
    if not 0 <= radius <= 4:
        raise ValueError('The explicit relative-spectrum counter supports radius at most four')
    counts = Counter(columns)
    result = [int(syndrome == 0)]
    if radius >= 1:
        result.append(counts[syndrome])
    if radius >= 2:
        ordered = sum(count*counts[label ^ syndrome] for label, count in counts.items())
        result.append((ordered-(len(columns) if syndrome == 0 else 0))//2)
    if radius >= 3:
        ordered = sum(ca*cb*counts[a ^ b ^ syndrome]
                      for a, ca in counts.items() for b, cb in counts.items())
        # Remove the three repeated-index events, with inclusion-exclusion.
        distinct = ordered-(3*len(columns)-2)*counts[syndrome]
        assert distinct >= 0 and distinct % 6 == 0
        result.append(distinct//6)
    if radius >= 4:
        pairs = Counter(a ^ b for a, b in combinations(columns, 2))
        ordered = sum(count*pairs[label ^ syndrome] for label, count in pairs.items())
        # Remove pair-pairs sharing one coordinate or both coordinates.
        # Each four-coordinate support has six ordered pair partitions.
        distinct = ordered-2*(len(columns)-2)*pairs[syndrome]
        if syndrome == 0:
            distinct -= len(columns)*(len(columns)-1)//2
        assert distinct >= 0 and distinct % 6 == 0
        result.append(distinct//6)
    return result


class RelativePlotkinSpectrum:
    def __init__(self, profile, columns):
        self.n = len(columns)
        if self.n < 2 or self.n & (self.n-1):
            raise ValueError('A nontrivial power-of-two code length is required')
        self.half, self.columns = self.n//2, tuple(columns)
        self.left_profile = tuple(i for i in profile if i < self.half)
        self.right_profile = tuple(i-self.half for i in profile if i >= self.half)
        self.left_columns = tuple(columns[:self.half])
        self.right_columns = tuple(value >> self.half for value in columns[self.half:])
        checks = check_rows(self.right_profile, self.right_columns)
        self.redundancy = len(checks)
        self.check_columns = [sum(((row >> t) & 1) << q for q, row in enumerate(checks))
                              for t in range(self.half)]

    def inspect(self, left_input, cutoff):
        if left_input <= 0 or left_input >> self.half:
            raise ValueError('Expected a nonzero left-half input')
        if any(((left_input & ((1 << j) ^ self.left_columns[j])).bit_count() & 1)
               for j in range(self.half) if j not in self.left_profile):
            raise ValueError('Input is not in the left projection code')
        a = polar_transform(left_input, self.half)
        weight = a.bit_count()
        if weight > cutoff:
            return dict(left_weight=weight, spectrum={})
        radius = (cutoff-weight)//2
        if radius > 4:
            raise ValueError('This diagnostic needs a small punctured-coset radius')
        inside = [t for t in range(self.half) if a >> t & 1]
        outside = [t for t in range(self.half) if not a >> t & 1]
        _, annihilator = solve([(self.check_columns[t], 0) for t in inside], self.redundancy)
        kernel_dimension = len(inside)-self.redundancy+len(annihilator)
        assert kernel_dimension >= 0

        def project(label):
            return sum(((label & basis).bit_count() & 1) << q
                       for q, basis in enumerate(annihilator))

        right_input = sum(((value & left_input).bit_count() & 1) << q
                          for q, value in enumerate(self.columns[self.half:]))
        cross_word = polar_transform(right_input, self.half)
        projected_columns = [project(self.check_columns[t]) for t in outside]
        syndrome = 0
        for t, column in zip(outside, projected_columns):
            if cross_word >> t & 1:
                syndrome ^= column
        counts = small_error_counts(projected_columns, syndrome, radius)
        spectrum = {weight+2*t: count*(1 << kernel_dimension)
                    for t, count in enumerate(counts) if count}
        return dict(left_input=left_input, left_word=a, left_weight=weight,
            right_shortening_dimension=kernel_dimension,
            punctured_redundancy=len(annihilator), checked_radius=radius,
            syndrome=syndrome, small_error_counts=counts, spectrum=spectrum,
            separated=not any(counts))

    def constraint(self, left_input, cutoff, materialize=True, max_radius=3):
        """Linear syndrome map and its distinct forbidden small-error labels."""
        a = polar_transform(left_input, self.half)
        weight = a.bit_count()
        radius = (cutoff-weight)//2
        if not 0 <= radius <= max_radius:
            raise ValueError('Critical left input exceeds the requested radius limit')
        if materialize and radius > 3:
            raise ValueError('Use an external syndrome-ball builder for radius above three')
        inside = [t for t in range(self.half) if a >> t & 1]
        outside = [t for t in range(self.half) if not a >> t & 1]
        _, basis = solve([(self.check_columns[t], 0) for t in inside], self.redundancy)

        def project(label):
            return sum(((label & row).bit_count() & 1) << q for q, row in enumerate(basis))

        errors = [project(self.check_columns[t]) for t in outside]
        forbidden = {0} if materialize else None
        if materialize and radius >= 1:
            forbidden.update(errors)
        if materialize and radius >= 2:
            forbidden.update(a ^ b for q, a in enumerate(errors) for b in errors[q+1:])
        if materialize and radius >= 3:
            forbidden.update(a ^ b ^ c for a, b, c in combinations(errors, 3))
        return dict(left_input=left_input, left_weight=weight, radius=radius,
                    syndrome_map=[project(1 << q) for q in range(self.redundancy)],
                    error_columns=errors, forbidden=forbidden, rank=len(basis),
                    kernel_dimension=len(inside)-self.redundancy+len(basis))

    def spectrum(self, left_words, right_histogram, cutoff):
        total = Counter({2*int(w): count for w, count in right_histogram.items()
                         if 2*int(w) <= cutoff})
        contribution = Counter()
        details = []
        for u, weight in left_words:
            if not weight:
                continue
            item = self.inspect(u, cutoff)
            assert item['left_weight'] == weight
            total.update(item['spectrum'])
            contribution.update(item['spectrum'])
            details.append(item)
        return dict(spectrum=dict(sorted(total.items())),
                    right_only_spectrum={2*int(w): count for w, count in right_histogram.items()
                                         if 2*int(w) <= cutoff},
                    cross_spectrum=dict(sorted(contribution.items())), details=details,
                    bad_left_cosets=sum(not item['separated'] for item in details),
                    left_cosets=len(details),
                    kernel_dimension_histogram=dict(sorted(Counter(
                        item['right_shortening_dimension'] for item in details).items())))
