"""Example 2. Decide which coefficients can be omitted at cutoff D=3."""
from itertools import product
from paper_examples.running import PROFILE, code, support


def run():
    cutoff = 3
    first = min(i for i in PROFILE if 2**i.bit_count() <= cutoff)
    assert first == 4
    for b0, c, b1 in product((0, 1), repeat=3):
        low_sets = []
        for alpha in (0, 1):
            family = code(alpha, b0, c, b1)
            low = set()
            for msg in range(1, 16):
                x = family.encode(msg)
                if x.bit_count() <= cutoff:
                    assert (family.input_word(msg) & (1 << 3)) == 0
                    low.add(x)
            low_sets.append(low)
        assert low_sets[0] == low_sets[1]
    # Set u4=u5=1. Although g5 alone is heavier than D, b1 changes this sum.
    outputs = [support(code(0, 0, 1, b1).encode(0b0110)) for b1 in (0, 1)]
    assert outputs == [[0, 5, 8, 9], [5, 9]]
    # On the extra domain alpha=b1, lift b1=1 by restoring alpha=1.
    lifts = [(alpha, 0, 1, 1) for alpha in (0, 1) if alpha == 1]
    return {"first_possible_source": first, "output_supports": outputs,
            "admissible_lift_of_(b0,c,b1)=(0,1,1)": lifts}


if __name__ == "__main__":
    print(run())
