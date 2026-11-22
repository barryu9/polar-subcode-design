"""Shared data of Example 1. Bit j of an integer denotes physical position j."""
from polar_subcodes.code import PolarSubcode

PROFILE = (3, 4, 5, 12)


def code(alpha, b0, c, b1):
    """Return the canonical N=16 code; coefficient order follows Example 5."""
    columns = [0] * 16
    columns[8] = (alpha << 3) | (b0 << 4) | (b1 << 5)
    columns[9] = c << 4
    return PolarSubcode(16, PROFILE, tuple(columns))


def support(word):
    return [i for i in range(word.bit_length()) if (word >> i) & 1]


def printed_bits(word, width):
    """The paper prints physical position zero on the left."""
    return "".join(str((word >> i) & 1) for i in range(width))
