"""Finite checks of all three theorems, three propositions and two corollaries.

Run python -m theory.check_results from the repository root. To follow one
argument, append its name, e.g. theorem2. Bit j of a packed word is coordinate
j in the paper. XOR adds vectors over GF(2), and bit_count() gives Hamming
weight. Each check prints its summary after its comparisons have passed.
"""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import ceil, comb, prod

from polar_subcodes.algebra import all_subspaces, leading
from polar_subcodes.code import Design, PolarSubcode

from algorithms.algorithm1_directory import list_inputs, outside_masks, right_checks, xor_selected
from algorithms.algorithm2_outer import critical_conditions, reverse_source_basis, complete_outer
from algorithms.algorithm3_fixed_profile import causal_pairs, common_conditions, make_code


def require(predicate, message):
    """Raise even if invoked with Python's -O flag."""
    if not predicate:
        raise AssertionError(message)


def independent_words(code):
    """Return (u,x) using the entrywise rule (G_N)[i,j]=1 iff j is a submask of i."""
    for bits in product((0, 1), repeat=code.k):
        # Place the message at its information positions. At frozen position j,
        # the parity of the selected source bits gives the dynamic frozen bit.
        source = sum(bit << i for bit, i in zip(bits, code.profile))
        u = source | sum(((mask & source).bit_count() % 2) << j
                         for j, mask in enumerate(code.columns))
        x = sum((sum(1 for i in range(code.n) if u >> i & 1 and (j & i) == j) % 2) << j
                for j in range(code.n))
        yield u, x


def configurations(n, profile):
    """Keep the information set fixed and try both values of every causal coefficient."""
    design = Design(n, profile)
    edges = causal_pairs(design)
    for bits in product((0, 1), repeat=len(edges)):
        yield make_code(design, edges, bits)


def join(left, right, outer):
    """Assemble the parent code from the constituent checks and T in (15).

    outer[i] is column i of T. Its bit r couples the i-th left information
    bit to the r-th frozen expression of the right constituent.
    """
    h = left.n
    full_profile = left.profile + tuple(h+i for i in right.profile)
    columns = list(left.columns) + [c << h for c in right.columns]
    for r, j in enumerate(Design(h, right.profile).rows):
        columns[h+j] |= sum(((value >> r) & 1) << i for i, value in zip(left.profile, outer))
    return PolarSubcode(2*h, full_profile, tuple(columns))


def proposition1():
    # Here i_D = 4: row 3 has weight four, row 4 has weight two. For words
    # through D = 2 the leading input position is at least 4. Shifting each
    # frozen-expression mask keeps exactly the source coefficients in E_D.
    profile, cutoff = (3, 4, 7), 2
    threshold = min(i for i in profile if 1 << i.bit_count() <= cutoff)
    by_retained = {}
    for code in configurations(8, profile):
        key = tuple(c >> threshold for c in code.columns)
        words = {x for _, x in independent_words(code) if 0 < x.bit_count() <= cutoff}
        require(by_retained.setdefault(key, words) == words, "cutoff word set changed")
    return f"{len(by_retained)} retained-coefficient classes, all member word sets agree"


def theorem1():
    # A nonzero baseline and a non-contiguous set of forbidden weights test both
    # the increment convention and the weight-specific directory multiplicities.
    base = PolarSubcode(8, (1, 3, 6), (0, 0, 2, 0, 8, 2, 0, 64))
    free = ((2, 1), (4, 1), (4, 3), (7, 6))
    released = {j for j, _ in free}
    weights = {1, 2, 4}
    sources = {i for _, i in free}
    source_mask = sum(1 << i for i in sources)
    # Drop released check rows, keeping the fixed baseline checks on all others.
    relaxed = PolarSubcode(base.n, tuple(sorted(set(base.profile) | released)),
                           tuple(0 if j in released else c for j, c in enumerate(base.columns)))
    directory = Counter()
    for u, x in independent_words(relaxed):
        # A surviving input must start at an original information position.
        # Store (source pattern q, baseline check residual beta, weight w).
        # These masks keep their original coordinate labels so that two rows
        # contributing to the same released check add at the same bit.
        if x.bit_count() in weights and leading(u) in base.profile:
            residual = sum((((u >> j) ^ (base.columns[j] & u).bit_count()) & 1) << j
                           for j in released)
            directory[u & source_mask, residual, x.bit_count()] += 1
    for bits in product((0, 1), repeat=len(free)):
        columns = list(base.columns)
        for bit, (j, i) in zip(bits, free):
            columns[j] ^= bit << i
        trial = PolarSubcode(8, base.profile, tuple(columns))
        direct = Counter(x.bit_count() for _, x in independent_words(trial)
                         if x.bit_count() in weights)
        predicted = Counter()
        for (pattern, residual, weight), multiplicity in directory.items():
            # The same increment matrix V acts on every source pattern.
            # When V q^T = beta, all M_w(q, beta) inputs of this entry
            # satisfy the released checks and contribute to A_w.
            value = 0
            for bit, (j, i) in zip(bits, free):
                if bit and pattern >> i & 1:
                    value ^= 1 << j
            if value == residual:
                predicted[weight] += multiplicity
        require(predicted == direct, "Theorem 1 multiplicities disagree")
    return "16 shared assignments checked against exact codewords, including all three layers"


def proposition2():
    """Verify that Algorithm 1 emits each input through D exactly once."""
    cases = 0
    for n in (1, 2, 4):
        for k in range(n + 1):
            for profile in combinations(range(n), k):
                for code in configurations(n, profile):
                    oracle = list(independent_words(code))
                    for cutoff in range(-1, n + 2):
                        expected = sorted(u for u, x in oracle if x.bit_count() <= cutoff)
                        require(sorted(list_inputs(code, cutoff)) == expected, "missing or duplicate input")
                        cases += 1
    return f"{cases} complete input lists, including zero, negative cutoffs and full-rate codes"


def small_pair():
    """Two length-four constituents, two right frozen positions, and D = 2."""
    left = PolarSubcode(4, (0, 1), (0, 0, 0, 2))
    right = PolarSubcode(4, (0, 1), (0, 0, 1, 3))
    return left, right, 2


def decode_theta(theta, f, k):
    """Unpack vec(T) into k source columns, each containing f check bits."""
    mask = (1 << f) - 1
    return tuple((theta >> (i*f)) & mask for i in range(k))


def joint(conditions, outer):
    """Stack Q_a T m in critical-message order to form L_D theta, as in (22)."""
    return tuple(c.project(xor_selected(outer, c.message)) for c in conditions)


def theta_data():
    """Tabulate both sides of the quotient for all 16 outer matrices.

    observations comes from the projected checks. spectra counts directly
    encoded parent codewords at weights 0 through D, including A_0 = 1.
    """
    left, right, cutoff = small_pair()
    conditions = critical_conditions(left, right, cutoff)
    f = right.n - right.k
    observations, spectra = {}, {}
    for t in range(1 << (f * left.k)):
        outer = decode_theta(t, f, left.k)
        observations[t] = joint(conditions, outer)
        counts = Counter(x.bit_count() for _, x in independent_words(join(left, right, outer)))
        spectra[t] = tuple(counts[w] for w in range(cutoff + 1))
    return left, right, cutoff, conditions, observations, spectra


def theorem2():
    left, right, cutoff, conditions, observations, spectra = theta_data()
    # Count each admissible outside vector with its inside kernel multiplicity.
    # Widen the cutoff here to test every coefficient of the full WEF in part (i).
    columns = right_checks(right)
    all_messages = critical_conditions(left, right, left.n * 2)
    for t in observations:
        outer = decode_theta(t, right.n - right.k, left.k)
        # For the zero left message, each right codeword appears in both
        # halves of the parent word. Its weight therefore doubles.
        prediction = Counter(2*x.bit_count() for _, x in independent_words(right))
        for condition in all_messages:
            a = left.encode(condition.message)
            outside = [j for j in range(left.n) if not a >> j & 1]
            value = condition.project(xor_selected(outer, condition.message))
            for z in outside_masks(outside, len(outside)):
                if condition.project(xor_selected(columns, z)) == value:
                    prediction[a.bit_count()+2*z.bit_count()] += 1 << len(condition.quotient.kernel)
        direct = Counter(x.bit_count() for _, x in independent_words(join(left, right, outer)))
        require(prediction == direct, "quotient formula mismatches full spectrum")
        # Return to D = 2 for the distance test. This right code has minimum
        # distance 2, so the zero-message branch already meets d_star = 3.
        permitted = all(y not in c.forbidden for y, c in zip(observations[t], conditions))
        require(permitted == (sum(spectra[t][1:]) == 0), "distance equivalence")

    kernel = {t for t in spectra if observations[t] == observations[0]}
    invariant = {v for v in spectra if all(spectra[t] == spectra[t ^ v] for t in spectra)}
    require(kernel == invariant == {t for t in spectra if spectra[t] == spectra[0]},
            "universal spectral directions")
    # Every binary linear map partitions the domain by its kernel; exhaust
    # every subspace that could be such a kernel, including rank-zero maps.
    width = (right.n - right.k) * left.k
    rho = width - (len(kernel).bit_length() - 1)
    for basis in all_subspaces(width):
        fiber = [xor_selected(basis, u) for u in range(1 << len(basis))]
        determines_spectrum = all(
            len({spectra[t ^ delta] for delta in fiber}) == 1 for t in spectra)
        if determines_spectrum:
            require(width - len(basis) >= rho, "smaller linear sufficient map")
    return f"full spectra and all linear kernels checked on {1<<width} outer assignments, rank {rho}"


def proposition3():
    left, right, _, conditions, observations, spectra = theta_data()
    f = right.n - right.k
    domains = [(0, tuple(1 << r for r in range(f))) for _ in left.profile]
    retained = reverse_source_basis(domains, conditions, f)
    kernel = {t for t, y in observations.items() if y == observations[0]}
    directions = [b << (i*f) for i, block in enumerate(retained) for b in block]
    reps = [xor_selected(directions, bits) for bits in range(1 << len(directions))]
    feasible = {t for t, counts in spectra.items() if not any(counts[1:])}
    # A representative fixes one joint observation. Adding any member of
    # ker L_D leaves it fixed; taking all feasible representatives recovers
    # every feasible outer matrix, including the discarded coordinates.
    require({t ^ k for t in reps if t in feasible for k in kernel} == feasible,
            "feasible fibers do not reconstruct every original assignment")
    require(len(reps) * len(kernel) == len(observations), "rank-nullity")
    for i, block in enumerate(retained):
        # L_i contains exactly the messages whose last nonzero bit is i.
        # Compare the whole tuple of observations from column t_i, since
        # one choice of that column must serve every message in the group.
        group = [c for c in conditions if c.message.bit_length() == i + 1]
        original = {tuple(c.project(t) for c in group) for t in range(1 << f)}
        reduced = {tuple(c.project(xor_selected(block, bits)) for c in group)
                   for bits in range(1 << len(block))}
        require(original == reduced, "lost a joint source-wise choice")
    # Part (iii) allows linked column parameters and a nonzero affine origin.
    origin, linked = 1, (0b0101, 0b1010, 0b0010)
    affine = [origin ^ xor_selected(linked, u) for u in range(1 << len(linked))]
    multiplicities = Counter(observations[t] for t in affine)
    equal_sizes = set(multiplicities.values())
    require(len(equal_sizes) == 1, "unequal affine fibers")
    good_images = {observations[t] for t in affine if t in feasible}
    require(sum(t in feasible for t in affine) == equal_sizes.pop() * len(good_images),
            "affine count")
    restricted = [(1, (3,)), (2, (1, 2))]
    bases = reverse_source_basis(restricted, conditions, f)
    for i, (base, original_basis) in enumerate(restricted):
        group = [c for c in conditions if c.message.bit_length() == i + 1]
        def images(basis):
            return {tuple(c.project(base ^ xor_selected(basis, b)) for c in group)
                    for b in range(1 << len(basis))}
        require(images(original_basis) == images(bases[i]), "restricted source choices")
    return f"{len(reps)} representatives; current-source images and linked affine fibers checked"


def corollary1():
    _, _, _, conditions, observations, spectra = theta_data()
    # Enumerate *all* affine subspaces of the four-parameter example.
    checked = set()
    for basis in all_subspaces(4):
        offsets = [xor_selected(basis, u) for u in range(1 << len(basis))]
        for base in observations:
            domain = frozenset(base ^ x for x in offsets)
            if domain in checked:
                continue
            checked.add(domain)
            # Every point must meet the distance target. One attainable
            # forbidden syndrome for one message rules out this all-points claim.
            projected = all({observations[t][j] for t in domain}.isdisjoint(c.forbidden)
                            for j, c in enumerate(conditions))
            direct = all(not any(spectra[t][1:]) for t in domain)
            require(projected == direct, "all-points affine distance test")
    return f"all {len(checked)} affine domains checked"


def check_mean_tree(observations, forbidden, feasible):
    """Compare the conditional mean Phi(pi) in (36) with direct violation counts."""
    width = (len(observations) - 1).bit_length()
    counts = {t: sum(y in bad for y, bad in zip(row, forbidden))
              for t, row in observations.items()}
    guaranteed = 0
    means = {}
    for depth in reversed(range(width + 1)):
        # The low 'depth' bits encode the partial assignment pi. Work from
        # leaves toward the root so both child means are available below.
        for bits in range(1 << depth):
            domain = [t for t in observations if t % (1 << depth) == bits]
            mean = Fraction(sum(counts[t] for t in domain), len(domain))
            from_images = Fraction(0)
            for j, bad in enumerate(forbidden):
                # The remaining bits map affinely onto this attainable
                # syndrome set with equal preimage sizes. The intersection
                # fraction is the violation probability for this condition.
                image = {observations[t][j] for t in domain}
                from_images += Fraction(len(image & bad), len(image))
            require(mean == from_images, "wrong conditional violation probability")
            means[depth, bits] = mean
            successes = sum(t in feasible for t in domain)
            require(successes >= ceil(len(domain) * max(0, 1-mean)), "count lower bound")
            if mean < 1:
                # Along a minimum-mean path Phi stays below one. At a leaf
                # Phi equals the integer number of violations, hence is zero.
                value = bits
                for j in range(depth, width):
                    value = min((value, value | (1 << j)), key=lambda b: means[j+1, b])
                require(counts[value] == 0 and value in feasible, "strict-mean greedy completion")
                guaranteed += 1
    return len(means), guaranteed


def theorem3():
    # Fixed constituents: compare the strict bound and chosen descendant
    # against direct encoding. A mean >= 1 makes no claim of failure.
    _, _, _, conditions, observations, spectra = theta_data()
    feasible = {t for t, counts in spectra.items() if not any(counts[1:])}
    outer = check_mean_tree(observations, [c.forbidden for c in conditions], feasible)

    # Common-supercode form: now an internal coefficient of the left code
    # changes together with two outer coefficients. These source observations
    # are correlated, but the conditional *sum* still counts violations.
    design = Design(8, (1, 3, 7))
    free = ((2, 1), (4, 1), (6, 3))
    checks = common_conditions(design, free, 3, lambda: None)
    observations, feasible = {}, set()
    for bits in range(1 << len(free)):
        trial = make_code(design, free, [(bits >> i) & 1 for i in range(len(free))])
        observations[bits] = tuple(xor_selected(cols, bits) for cols, _ in checks)
        if all(not x or x.bit_count() > 3 for _, x in independent_words(trial)):
            feasible.add(bits)
    common = check_mean_tree(observations, [bad for _, bad in checks], feasible)
    require(outer[1] and common[1], "fixture should exercise both strict-mean forms")
    return (f"outer/common: {outer[0]}/{common[0]} prefixes, {outer[1]}/{common[1]} "
            "strict-mean completions; source count checked with Cor. 2")


def corollary2():
    # Use Example 7, where the uniform count is nontrivial. Exhaust all 2^14
    # outer matrices via quotient tests rather than expanding their 2^11 words.
    left = PolarSubcode(16, (3, 5), (0,) * 16)
    right = PolarSubcode(16, (3, 5, 9, 10, 11, 12, 13, 14, 15), (0,) * 16)
    right_words = {x for _, x in independent_words(right)}
    require({x for _, x in independent_words(left)} <= right_words and
            min(x.bit_count() for x in right_words if x) == 4, "containment hypothesis")
    cutoff, f = 5, 7
    conditions = critical_conditions(left, right, cutoff)
    lambdas = [0] * left.k
    upper = [0] * left.k
    for c in conditions:
        a = left.encode(c.message)
        weight = a.bit_count()
        # The factor stores rank(P_inside), equal to w_a - 1 here.
        # The quotient rank r_a in (46) is f minus that stored rank.
        require(len(c.quotient.kernel) == 1 and c.quotient.rank == weight - 1,
                "retained-constituent rank")
        i = c.message.bit_length() - 1
        lambdas[i] += len(c.forbidden) << c.quotient.rank
        volume = sum(comb(left.n-weight, j) for j in range((cutoff-weight)//2 + 1))
        upper[i] += (1 << (weight-1)) * volume
    require(all(l <= u < 1 << f for l, u in zip(lambdas, upper)), "uniform source bound")
    prefixes = [()]
    for i in range(left.k):
        group = [c for c in conditions if c.message.bit_length() == i+1]
        next_level = []
        for prefix in prefixes:
            # Assign the next full column t_i. Every earlier accepted prefix
            # must have at least 2^f - lambda_i extensions, even if their
            # identities differ from one prefix to the next.
            continuations = [(*prefix, t) for t in range(1 << f) if all(
                c.project(xor_selected((*prefix, t), c.message)) not in c.forbidden
                for c in group)]
            require(len(continuations) >= (1 << f) - lambdas[i], "source-wise extension")
            next_level.extend(continuations)
        prefixes = next_level
    lower = prod((1 << f)-u for u in upper)
    require((upper, lower, len(prefixes)) == ([8, 16], 13440, 13472),
            "Example 7 uniform bounds and exact count")
    require(complete_outer(left, right, cutoff) in set(prefixes), "constructed matrix infeasible")
    return f"Lambda={upper}, lower bound {lower}, exact {len(prefixes)}"


CHECKS = (proposition1, theorem1, theorem2, proposition2, proposition3,
          corollary1, theorem3, corollary2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="e.g. theorem1 corollary2 (default: all)")
    requested = parser.parse_args().names
    available = {f.__name__: f for f in CHECKS}
    for name in requested or available:
        if name not in available:
            parser.error(f"Choose from {', '.join(available)}")
        print(f"{name}: {available[name]()}")
