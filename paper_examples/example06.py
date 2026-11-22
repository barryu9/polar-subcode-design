"""Example 6. Complete the two retained coefficients in physical source order."""
from fractions import Fraction
from itertools import product
from paper_examples.running import code


def run():
    def violations(c, b1):
        return (int(c == 0), int((c ^ b1) == 0))

    joint = Fraction(sum(sum(violations(c, b1)) for c, b1 in product((0, 1), repeat=2)), 4)
    # Each condition is checked at its last nonzero source. Later choices
    # cannot change a condition already settled.
    first = [violations(c, 0)[0] for c in (0, 1)]
    c = first.index(min(first))
    second = [violations(c, b1)[1] for b1 in (0, 1)]
    b1 = second.index(min(second))
    means = [Fraction(sum(counts), 2) for counts in (first, second)]
    assert joint == 1 and means == [Fraction(1, 2)] * 2
    constructed = code(0, 0, c, b1)
    distance = min(constructed.encode(m).bit_count() for m in range(1, 16))
    assert (c, b1, distance) == (1, 0, 4)
    return {"joint_mean": str(joint), "group_means": list(map(str, means)),
            "violations_for_children": [first, second], "c,b1": [c, b1],
            "nonzero_frozen_expression": "u9 = u4", "dmin": distance}


if __name__ == "__main__":
    print(run())
