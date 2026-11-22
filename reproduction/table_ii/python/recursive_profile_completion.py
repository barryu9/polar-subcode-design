"""Exact small constituents and complete projected-coset outer completion.

The leaf distance restrictions define an explicit recursive subfamily. They are
not necessary conditions for every parent code. Unsolved prefixes are retained.
"""

def coordinates(n, profile):
    return tuple((j, i) for j in range(n) if j not in profile for i in profile if i < j)
