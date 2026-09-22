"""A priori causal realization of distance-protected polar support domains.

The distance ingredient uses a Reed--Muller top-degree Moore check. It is
related to known RM subcode/sum-free constructions, not a claim to a new
general RM distance bound. Profile compatibility is checked before support
construction, using independent binary linear systems for each source.
"""
from fractions import Fraction
from functools import lru_cache
from math import comb, gcd

from polar_subcodes.algebra import gf_mul
from polar_subcodes.theory.interfaces import echelon
from polar_subcodes.construction.domains import initial_structure
from polar_subcodes.algebra import solve


def polynomial_remainder(a, b):
    if b <= 0:
        raise ValueError('Positive polynomial divisor required')
    while a.bit_length() >= b.bit_length():
        a ^= b << (a.bit_length()-b.bit_length())
    return a


def polynomial_gcd(a, b):
    while b:
        a, b = b, polynomial_remainder(a, b)
    return a


def is_irreducible(polynomial, degree):
    if degree < 1 or polynomial.bit_length() != degree+1 or not polynomial & 1:
        return False
    n, x, power = 1 << degree, polynomial_remainder(2, polynomial), polynomial_remainder(2, polynomial)
    for k in range(1, degree+1):
        power = gf_mul(power, power, n, polynomial)
        if k <= degree//2 and polynomial_gcd(power ^ x, polynomial) != 1:
            return False
    return power == x


@lru_cache(maxsize=None)
def first_irreducible(degree):
    if degree < 1:
        raise ValueError('Positive field degree required')
    return next(p for p in range((1 << degree)+1, 1 << (degree+1), 2)
                if is_irreducible(p, degree))


def gf_power(value, exponent, degree, polynomial):
    result = 1
    while exponent:
        if exponent & 1:
            result = gf_mul(result, value, 1 << degree, polynomial)
        value = gf_mul(value, value, 1 << degree, polynomial)
        exponent >>= 1
    return result


def field_determinant(matrix, degree, polynomial):
    rows = [list(row) for row in matrix]
    size, result = len(rows), 1
    if any(len(row) != size for row in rows):
        raise ValueError('Square field matrix required')
    for j in range(size):
        pivot = next((k for k in range(j, size) if rows[k][j]), None)
        if pivot is None:
            return 0
        rows[j], rows[pivot] = rows[pivot], rows[j]
        value = rows[j][j]
        result = gf_mul(result, value, 1 << degree, polynomial)
        inverse = gf_power(value, (1 << degree)-2, degree, polynomial)
        for k in range(j+1, size):
            factor = gf_mul(rows[k][j], inverse, 1 << degree, polynomial)
            if factor:
                for t in range(j, size):
                    rows[k][t] ^= gf_mul(factor, rows[j][t], 1 << degree, polynomial)
    return result


def moore_value(vectors, degree, polynomial):
    rows = [list(vectors)]
    for _ in range(1, len(vectors)):
        rows.append([gf_mul(x, x, 1 << degree, polynomial) for x in rows[-1]])
    return field_determinant(rows, degree, polynomial)


@lru_cache(maxsize=None)
def moore_layer(degree, order):
    if not 2 <= order <= degree-2:
        raise ValueError('This distance-gap construction needs 2 <= order <= degree-2')
    n, polynomial = 1 << degree, first_irreducible(degree)
    columns = [0]*n
    layer = tuple(i for i in range(n) if i.bit_count() == degree-order)
    for i in layer:
        columns[i] = moore_value([1 << j for j in range(degree) if not (i >> j) & 1],
                                degree, polynomial)
    core = tuple(i for i in range(n) if i.bit_count() > degree-order)
    return dict(n=n, degree=degree, order=order, polynomial=polynomial,
                layer=layer, core=core, check_columns=tuple(columns),
                check_rank=len(echelon(columns)),
                baseline=1 << (degree-order), target=3*(1 << (degree-order-1)))


def causal_check_domain(n, profile, destinations, edges, check_columns):
    """Exact test for C_s contained in the auxiliary check kernel.

    Frozen coordinates outside destinations are zero. Only the supplied
    (destination, information-source) coefficients can be nonzero. Failure
    excludes this auxiliary subdomain, not all distance-achieving supports.
    """
    profile, destinations, edges = tuple(profile), tuple(destinations), tuple(edges)
    if (n < 1 or n & (n-1) or len(check_columns) != n
            or len(set(profile)) != len(profile) or len(set(destinations)) != len(destinations)
            or set(profile) & set(destinations)
            or any(not 0 <= i < n for i in (*profile, *destinations))
            or any(column < 0 for column in check_columns)
            or len(set(edges)) != len(edges)
            or any(j not in destinations or i not in profile or i >= j for j, i in edges)):
        raise ValueError('Invalid causal coordinate domain or check map')
    width = max((column.bit_length() for column in check_columns), default=0)
    result = dict(compatible=True, free_dimension=0, source_tests=[],
                  coefficient_count=len(edges), columns=[0]*n,
                  scope='Containment in the stated auxiliary kernel, not general distance feasibility')
    directions = []
    for i in sorted(profile):
        tail = tuple(sorted(j for j, source in edges if source == i))
        rank = len(echelon(check_columns[j] for j in tail))
        equations = [(sum(((check_columns[j] >> bit) & 1) << q for q, j in enumerate(tail)),
                      (check_columns[i] >> bit) & 1) for bit in range(width)]
        solution = solve(equations, len(tail))
        test = dict(source=i, tail=tail, rank=rank, compatible=solution is not None)
        result['source_tests'].append(test)
        if solution is None:
            result['compatible'] = False
            continue
        origin, basis = solution
        for q, j in enumerate(tail):
            result['columns'][j] |= ((origin >> q) & 1) << i
        directions.extend(dict(source=i, destinations=[j for q, j in enumerate(tail) if b >> q & 1])
                          for b in basis)
        result['free_dimension'] += len(basis)
    if not result['compatible']:
        result.pop('columns')
        result['free_dimension'] = None
        result['count'] = 0
    else:
        result['count'] = 1 << result['free_dimension']
        result['directions'] = directions
    return result


def first_domain_moore_test(degree, order, profile):
    model = moore_layer(degree, order)
    profile = tuple(sorted(profile))
    if not set(profile) <= set(model['core']) | set(model['layer']):
        raise ValueError('Profile lies outside the selected RM mother')
    structure = initial_structure(model['n'], profile, model['target'])
    edges = tuple((structure['rows'][q], i) for q, i in structure['coordinates'])
    result = causal_check_domain(model['n'], profile, structure['rows'], edges, model['check_columns'])
    result.update(profile=profile, target=model['target'],
                  distance_lower_bound=model['target'] if result['compatible'] else None,
                  polynomial=model['polynomial'],
                  auxiliary_rank=model['check_rank'], structure=structure)
    return result


def canonical_moore_code(degree, order):
    model = moore_layer(degree, order)
    pivots, independent = [], []
    for j in reversed(model['layer']):
        if len(echelon((*independent, model['check_columns'][j]))) > len(independent):
            independent.append(model['check_columns'][j])
            pivots.append(j)
    profile = tuple(sorted((*model['core'], *(j for j in model['layer'] if j not in pivots))))
    result = first_domain_moore_test(degree, order, profile)
    if not result['compatible'] or result['count'] != 1:
        raise AssertionError('The reverse-pivot profile did not realize the full check kernel')
    result.update(n=model['n'], k=len(profile), degree=degree, order=order,
                  frozen_layer=sorted(pivots), baseline=model['baseline'])
    return result


def gaussian_binomial(n, k):
    numerator = denominator = 1
    for j in range(k):
        numerator *= (1 << (n-j))-1
        denominator *= (1 << (k-j))-1
    value, remainder = divmod(numerator, denominator)
    if remainder:
        raise AssertionError('Nonintegral Gaussian binomial')
    return value


def positive_rate_bound(t):
    if t < 2:
        raise ValueError('The family uses t >= 2')
    degree, order, h, d = 2*t+1, t+1, 1 << (2*t+1), 1 << t
    if gcd(degree, order) != 1:
        raise AssertionError('Moore-map surjectivity hypothesis failed')
    left_dimension = h//2
    right_dimension = h//2+comb(degree, order)-degree
    f = h-right_dimension
    minimum_words = (1 << t)*gaussian_binomial(degree, t)
    radius = d//2-1
    volume = sum(comb(h-2*d, j) for j in range(radius+1))
    bound = minimum_words*(1 << (2*d-1))*volume
    probability_bound = Fraction(bound, 1 << f)
    return dict(t=t, degree=degree, right_order=order, n=2*h,
                k=left_dimension+right_dimension, left_dimension=left_dimension,
                right_dimension=right_dimension, right_redundancy=f,
                unprecoded_distance=2*d, target=3*d,
                guaranteed_distance=3*d if bound < (1 << f) else None,
                right_target=3*d//2, left_minimum_words=minimum_words,
                outer_radius=radius, total_column_bound=bound,
                bound_numerator=probability_bound.numerator,
                bound_denominator=probability_bound.denominator,
                sufficient=bound < (1 << f),
                coarse_log_bound=t*t+t-1+(1 << (t-1))*(2*t+5),
                coarse_redundancy_lower_bound=1 << (2*t-1))
