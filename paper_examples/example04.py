"""Example 4. Project away inside bits and recover every compatible lift."""
from collections import Counter
from itertools import combinations
from polar_subcodes.algebra import check_rows, polar_transform, solve
from paper_examples.running import printed_bits, support


def run():
    # P acts on the right codeword x_R. Its seven rows correspond to global
    # frozen positions 8,9,10,11,13,14,15. columns[i] is P's column at local
    # right-codeword position i, packed as a seven-bit check value.
    checks = check_rows((4,), (0,) * 8)
    columns = [sum(((row >> i) & 1) << k for k, row in enumerate(checks))
               for i in range(8)]
    summaries = {}
    for u_left in (1 << 4, (1 << 4) | (1 << 5)):
        a = polar_transform(u_left, 8)
        inside = support(a)
        outside = [i for i in range(8) if i not in inside]
        # A basis for all checks annihilating the inside columns gives Q_a.
        # At D=3 the outside bits are zero, so Q_a T m = 0 is exactly the
        # condition for an inside assignment to realize the chosen residual.
        _, q_rows = solve([(columns[i], 0) for i in inside], 7)
        residuals = Counter()
        for xi in range(4):
            rhs = (columns[inside[0]] if xi & 1 else 0) ^ (columns[inside[1]] if xi & 2 else 0)
            residuals[rhs] += 1
        # Match Example 3's release of just positions 8 and 9: the other
        # five residual bits must vanish. n is the number of inside lifts.
        restricted = {printed_bits(rhs, 2): n for rhs, n in residuals.items() if rhs >> 2 == 0}
        label = printed_bits(u_left >> 3, 3)
        summaries[label] = {"inside": inside, "rank": len(q_rows), "residual_counts": restricted}
        if label == "010":
            assert len(q_rows) == 6 and restricted == {"00": 2, "10": 2}
        else:
            assert len(q_rows) == 5 and restricted == {"00": 1, "11": 1}
            continue

        # With c=1 and alpha=b0=b1=0 the packed rhs is e_1. At D=4,
        # enumerate outside assignments of weight 0 or 1 before lifting.
        lifts = []
        for weight in (0, 1):
            for z in combinations(outside, weight):
                rhs = 2
                for pos in z:
                    rhs ^= columns[pos]
                if any((row & rhs).bit_count() % 2 for row in q_rows):
                    continue
                # Undo the projection for this accepted z. Every compatible
                # inside assignment xi yields one full word (a + x_R, x_R).
                for xi in range(4):
                    x_right = sum(1 << pos for pos in z)
                    check = 0
                    for bit, pos in enumerate(inside):
                        if (xi >> bit) & 1:
                            x_right |= 1 << pos
                            check ^= columns[pos]
                    if check == rhs:
                        lifts.append((z, printed_bits(xi, 2), printed_bits(a ^ x_right, 8),
                                      printed_bits(x_right, 8)))
        assert lifts == [((1,), "10", "01001000", "11000000"),
                         ((1,), "01", "11000000", "01001000")]
        summaries[label]["D=4_lifts_(outside,inside,left,right)"] = lifts
    return summaries


if __name__ == "__main__":
    print(run())
