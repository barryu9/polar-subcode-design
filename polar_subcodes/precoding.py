"""PAC, CRC and canonical precoding using Section II's polar input coordinates.

Bit i in an integer is position i in the natural SC order. A payload is read
from low bit to high bit; for the CRC this is also the order entering the LFSR.
"""
from .algebra import leading, rref, polar_transform
from .code import Design, PolarSubcode

PAC_TAPS = (1, 0, 1, 1, 0, 1, 1)


def _check_taps(taps):
    if not taps or taps[0] != 1 or any(type(t) is not int or t not in (0,1) for t in taps):
        raise ValueError("PAC taps must be binary in increasing delay order, with g[0]=1")


def spread(message, profile):
    """Place successive message bits into the selected polar-input positions."""
    if type(message) is not int or message < 0 or message >> len(profile):
        raise ValueError("Message does not fit the profile")
    return sum(((message >> q) & 1) << i for q, i in enumerate(profile))


def pac_input(n, profile, message, taps=PAC_TAPS):
    """Rate-profile v, then return u = v*g (causal convolution truncated at N)."""
    profile = Design(n, tuple(profile)).profile
    _check_taps(taps)
    v = spread(message, profile)
    u = 0
    for delay, tap in enumerate(taps):
        if tap:
            u ^= v << delay
    return u & ((1 << n)-1)


def pac_code(n, profile, taps=PAC_TAPS):
    """Express v[j]=0 at frozen positions as causal equations for u[j].

    At an information position u[j] is free and v[j] can be recovered using
    earlier v. Thus encoding PAC payloads uses pac_input(), while canonical
    encoding with the same payload integers spans the same set of codewords.
    """
    profile = Design(n, tuple(profile)).profile
    _check_taps(taps)
    info = set(profile)
    v_masks = []
    columns = [0] * n
    for j in range(n):
        past = 0
        for delay in range(1, min(j+1, len(taps))):
            if taps[delay]:
                past ^= v_masks[j-delay]
        if j in info:
            v_masks.append((1 << j) ^ past)
        else:
            columns[j] = past
            v_masks.append(0)
    return PolarSubcode(n, profile, tuple(columns))


def unpack_pac(u, profile, taps):
    """Invert u=v*g to recover the PAC information bits at the selected positions."""
    _check_taps(taps)
    v = 0
    for j in range(max(profile, default=-1)+1):
        bit = (u >> j) & 1
        for delay in range(1, min(j+1,len(taps))):
            if taps[delay]:
                bit ^= (v >> (j-delay)) & 1
        v |= bit << j
    return sum(((v >> i) & 1) << q for q, i in enumerate(profile))


def crc_remainder(message, length, polynomial):
    """Stream 'length' bits into the zero-initialized MSB-first CRC register."""
    if type(polynomial) is not int or polynomial < 3 or not polynomial & 1:
        raise ValueError("CRC polynomial must include the leading and constant coefficients")
    if type(length) is not int or length < 0 or type(message) is not int or message < 0 or message >> length:
        raise ValueError("Message must fit in length bits")
    width = polynomial.bit_length()-1
    state = 0
    for j in range(length):
        state = (state << 1) | ((message >> j) & 1)
        if state >> width:
            state ^= polynomial
    return state


def append_crc(message, payload, polynomial):
    """Append the high-to-low coefficients of the remainder of M(X) X^width."""
    if type(message) is not int or message < 0 or message >> payload:
        raise ValueError("Message exceeds the payload length")
    width = polynomial.bit_length()-1
    remainder = crc_remainder(message, payload+width, polynomial)
    trailer = sum(((remainder >> (width-1-j)) & 1) << j for j in range(width))
    return message | (trailer << payload)


def from_input_basis(n, rows):
    """Convert independent polar-input basis rows, e.g. a published Z, to Section II."""
    rows = tuple(rows)
    if any(type(row) is not int or row <= 0 or row >> n for row in rows):
        raise ValueError("A basis row must be a nonzero N-bit integer")
    reduced = rref(rows)
    if len(reduced) != len(rows):
        raise ValueError("Input generator rows must be independent")
    profile = tuple(map(leading, reduced))
    columns = [0]*n
    for pivot, row in zip(profile,reduced):
        entries = row ^ (1 << pivot)
        while entries:
            position = leading(entries)
            columns[position] |= 1 << pivot
            entries ^= 1 << position
    return PolarSubcode(n, profile, tuple(columns))
