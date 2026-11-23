"""Example 1. Construct H, B, and B G_16 for the four-coefficient family."""
from itertools import product
from polar_subcodes.algebra import polar_transform
from paper_examples.running import code, support


def run():
    for coefficients in product((0, 1), repeat=4):
        family = code(*coefficients)
        # H acts on the polar input u. A unit message selects one generator
        # row of B, so the parity of its overlap with every row of H must be zero.
        input_rows = [family.input_word(1 << i) for i in range(family.k)]
        assert all((row & check).bit_count() % 2 == 0
                   for row in input_rows for check in family.checks)
        # For the unit message at source 3, u8 = alpha is the only possible
        # frozen contribution. Applying G_16 turns this input row into a codeword.
        first_row = (1 << 3) | (coefficients[0] << 8)
        assert input_rows[0] == first_row
        assert family.encode(1) == polar_transform(first_row, 16)
    # All four coefficients set to one make every variable term in Fig. 1 visible.
    family = code(1, 1, 1, 1)
    return {
        "profile": family.profile,
        "H (row supports)": [support(row) for row in family.checks],
        "B (row supports)": [support(family.input_word(1 << i)) for i in range(4)],
        "B G_16 (row supports)": [support(family.encode(1 << i)) for i in range(4)],
    }


if __name__ == "__main__":
    print(run())
