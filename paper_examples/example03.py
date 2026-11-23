"""Example 3. Enumerate the common supercode before grouping its residuals."""
from collections import Counter
from itertools import product
from polar_subcodes.code import PolarSubcode
from paper_examples.running import PROFILE, code, printed_bits, support


def run():
    # Releasing checks 8 and 9 makes six input positions free.
    free = tuple(sorted((*PROFILE, 8, 9)))
    mother = PolarSubcode(16, free, (0,) * 16)
    directory = []
    multiplicities = Counter()
    for msg in range(1 << len(free)):
        u, x = mother.input_word(msg), mother.encode(msg)
        if x.bit_count() != 2:
            continue
        # q records the source bits (u3,u4,u5). The two checks released from
        # the zero-coefficient baseline have residual beta = (u8,u9).
        q = (u >> 3) & 7
        beta = (u >> 8) & 3
        directory.append((support(u), support(x), printed_bits(q, 3), printed_bits(beta, 2)))
        # A causal codeword input must start at an information position.
        # Keep all nine rows for comparison with the paper's table, and apply
        # the leading-position filter before counting residual multiplicities.
        if support(u)[0] in PROFILE:
            multiplicities[q, beta] += 1
    assert len(directory) == 9 and sum(multiplicities.values()) == 6
    feasible = []
    for config in product((0, 1), repeat=4):
        alpha, b0, c, b1 = config
        # A directory word survives exactly when its residual equals V q^T.
        # For source patterns 010 and 011, that increment is (b0,c) and
        # (b0+b1,c), respectively. Each matching input contributes once to A_2.
        matched = multiplicities[2, b0 | (c << 1)]
        matched += multiplicities[6, (b0 ^ b1) | (c << 1)]
        family = code(*config)
        counts = Counter(family.encode(m).bit_count() for m in range(16))
        assert counts[2] == matched == 2 * (c == 0) + (b0 ^ b1 == c)
        if matched == 0:
            assert min(w for w in counts if w > 0) == 4
            feasible.append(config)
    assert len(feasible) == 4
    return {"directory_(input,output,q,beta)": directory, "feasible": feasible}


if __name__ == "__main__":
    print(run())
