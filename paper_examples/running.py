"""Shared data of Example 1. Bit j of an integer denotes physical position j."""
from polar_subcodes.code import PolarSubcode

PROFILE = (3, 4, 5, 12)


def code(alpha, b0, c, b1):
    """Build the four-coefficient family in Fig. 1 at one assignment."""
    # Bit i in columns[j] selects information bit u_i in the expression for u_j.
    # Thus u8 = alpha*u3 + b0*u4 + b1*u5 and u9 = c*u4, with sums in GF(2).
    # The packed message uses the order in PROFILE, so its fourth bit selects u12.
    columns = [0] * 16
    columns[8] = (alpha << 3) | (b0 << 4) | (b1 << 5)
    columns[9] = c << 4
    return PolarSubcode(16, PROFILE, tuple(columns))


def support(word):
    return [i for i in range(word.bit_length()) if (word >> i) & 1]


def printed_bits(word, width):
    """The paper prints physical position zero on the left."""
    return "".join(str((word >> i) & 1) for i in range(width))
