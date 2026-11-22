"""Reference-informed diagnostics; none of the replays is a fresh search."""

def binary_rank(rows):
    pivots = {}
    for value in rows:
        while value:
            pivot = value.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)
