"""Binary coordinates used to compute the conditional means in Example 8."""


def xor_mask(columns, mask):
    value = 0
    while mask:
        bit = mask & -mask
        value ^= columns[bit.bit_length()-1]
        mask ^= bit
    return value


def suffix_coordinates(columns):
    """Use rightmost independent variables so each pivot depends on earlier bits."""
    pivots, indices = {}, []
    for index in reversed(range(len(columns))):
        value, coordinates = columns[index], 0
        while value:
            p = value.bit_length()-1
            if p not in pivots:
                coordinates ^= 1 << len(indices)
                pivots[p] = value, coordinates
                indices.append(index)
                break
            other, label = pivots[p]
            value, coordinates = value ^ other, coordinates ^ label

    def encode(value):
        label = 0
        while value:
            p = value.bit_length()-1
            if p not in pivots:
                raise ValueError('Syndrome outside the map image')
            other, coordinates = pivots[p]
            value, label = value ^ other, label ^ coordinates
        return label
    return [encode(value) for value in columns], {index: q for q, index in enumerate(indices)}, encode
