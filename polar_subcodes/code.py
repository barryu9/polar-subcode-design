"""Canonical profiles and causal dynamic frozen expressions.

Indices are zero based. Bit i of columns[j] is the coefficient of u_i in
u_j. Only earlier information bits may appear. Encoding is x = u F^{tensor m}
without an additional bit-reversal permutation.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from .algebra import polar_transform


def integer(value):
    """Read lossless decimal/hexadecimal masks, rejecting floats and booleans."""
    if type(value) is int:
        return value
    if isinstance(value, str):
        return int(value, 16 if value.startswith('0x') else 10)
    raise ValueError('Expected an integer or a decimal/hexadecimal string')


@dataclass(frozen=True)
class Design:
    n: int
    profile: tuple

    def __post_init__(self):
        if type(self.n) is not int or self.n < 1 or self.n & (self.n-1):
            raise ValueError('N must be a positive power of two')
        if (any(type(i) is not int or not 0 <= i < self.n for i in self.profile)
                or tuple(sorted(set(self.profile))) != tuple(self.profile)):
            raise ValueError('Profile must be a sorted, distinct information set')
        object.__setattr__(self, 'profile', tuple(self.profile))

    @property
    def rows(self):
        return tuple(j for j in range(self.n) if j not in self.profile)

    @property
    def d_P(self):
        return min((1 << i.bit_count() for i in self.profile), default=None)


@dataclass(frozen=True)
class PolarSubcode:
    n: int
    profile: tuple
    columns: tuple

    def __post_init__(self):
        design = Design(self.n, tuple(self.profile))
        columns = tuple(integer(c) for c in self.columns)
        mask = sum(1 << i for i in design.profile)
        if len(columns) != self.n or any(c < 0 or c >> j or c & ~mask or
                (j in design.profile and c) for j, c in enumerate(columns)):
            raise ValueError('Support must use earlier information bits at frozen positions only')
        object.__setattr__(self, 'profile', design.profile)
        object.__setattr__(self, 'columns', columns)

    @property
    def k(self):
        return len(self.profile)

    @property
    def d_P(self):
        return Design(self.n, self.profile).d_P

    @property
    def checks(self):
        return tuple((1 << j) ^ c for j, c in enumerate(self.columns) if j not in self.profile)

    @property
    def supports(self):
        return tuple(tuple(i for i in self.profile if c >> i & 1) for c in self.columns)

    def input_word(self, message):
        if type(message) is not int or message < 0 or message >> self.k:
            raise ValueError('Message must be an unsigned K-bit integer')
        u = sum(((message >> q) & 1) << i for q, i in enumerate(self.profile))
        return u | sum(((c & u).bit_count() & 1) << j for j, c in enumerate(self.columns))

    def encode(self, message):
        return polar_transform(self.input_word(message), self.n)

    def as_dict(self):
        return dict(schema_version=1, index_base=0, transform='F_tensor_no_bit_reversal',
                    N=self.n, K=self.k, profile=list(self.profile),
                    columns=[str(c) for c in self.columns])

    def describe(self, include_static=False):
        lines = [f'N = {self.n}, K = {self.k}, d_P = {self.d_P}',
                 'Information positions (zero based):', ', '.join(map(str, self.profile)),
                 'Encoding: x = u F^{tensor log2(N)} (no bit reversal)',
                 'Dynamic frozen expressions (XOR over GF(2)):']
        for j, support in enumerate(self.supports):
            if j not in self.profile and (support or include_static):
                lines.append(f'u[{j}] = '+(' XOR '.join(f'u[{i}]' for i in support) or '0'))
        if not any(self.columns):
            lines.append('(All frozen positions are zero.)')
        elif not include_static:
            lines.append('All other frozen positions are zero.')
        return '\n'.join(lines)


def load_code(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('index_base', 0) != 0 or data.get('transform', 'F_tensor_no_bit_reversal') != 'F_tensor_no_bit_reversal':
        raise ValueError('Unsupported indexing or transform convention')
    n, profile = data['N'], tuple(data['profile'])
    if data.get('K', len(profile)) != len(profile):
        raise ValueError('K does not match the information set')
    columns = data.get('columns')
    equations = data.get('dynamic_expressions')
    if columns is None:
        columns = [0]*n
        for j, sources in (equations or {}).items():
            if len(set(sources)) != len(sources) or any(type(i) is not int or i < 0 for i in sources):
                raise ValueError('Invalid or repeated expression source')
            index = integer(j)
            if not 0 <= index < n:
                raise ValueError('Expression position outside the code')
            columns[index] = sum(1 << i for i in sources)
    code = PolarSubcode(n, profile, tuple(columns))
    if equations is not None:
        expected = {str(j): list(s) for j, s in enumerate(code.supports) if s}
        actual = {str(integer(j)): sorted(s) for j, s in equations.items() if s}
        if expected != actual:
            raise ValueError('columns and dynamic_expressions disagree')
    return code


def load_profile(path):
    """Read profile data only. Never initialize a search from stored support."""
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if 'columns' in data or 'dynamic_expressions' in data:
        raise ValueError('Construction input must be a profile, not a saved support')
    return load_code(path)
