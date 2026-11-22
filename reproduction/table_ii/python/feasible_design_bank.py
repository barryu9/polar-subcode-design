"""Verified polar-subcode candidates and exact multi-solution separation."""

import hashlib
import json
from verify_causal_flag_theory import polar_transform, rref

def validate_design(n, profile, columns):
    """Canonical expressions use only earlier information positions."""
    if n < 2 or n & (n-1):
        raise ValueError('N must be a power of two')
    profile = tuple(sorted(profile))
    if not profile or len(set(profile)) != len(profile) or not all(0 <= i < n for i in profile):
        raise ValueError('Invalid information profile')
    if len(columns) != n or any(type(c) is not int for c in columns):
        raise ValueError('Expected N integer expression masks')
    mask = sum(1 << i for i in profile)
    if any(c < 0 or c >> j or c & ~mask or (j in profile and c)
           for j, c in enumerate(columns)):
        raise ValueError('Noncausal or noncanonical expression')
    return profile

def generator_rows(n, profile, columns):
    profile = validate_design(n, profile, columns)
    return tuple(polar_transform((1 << i) | sum(((c >> i) & 1) << j
                 for j, c in enumerate(columns)), n) for i in profile)

def code_fingerprint(n, rows):
    """Identify the row space with fixed physical coordinate order."""
    canonical = rref(rows)
    payload = dict(N=n, rows=[hex(row) for row in canonical])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
