"""Causal support-graph generator; no witness bank, distance mask or teacher.

Rebuild a typed bipartite graph after each frozen row. Its edge states are
selected/rejected/undecided. A MADE head autoregresses within the current row.
Exact partial effective generator rows are state, never spectrum features.
"""

from dataclasses import dataclass
import numpy as np

PROFILE64 = (13,14,15,19,23,25,26,27,28,29,30,31,37,38,39,41,
             42,43,44,45,47,51,53,54,55,57,58,59,60,61,62,63)

@dataclass(frozen=True)
class Design:
    n: int = 64
    profile: tuple = PROFILE64

    def __post_init__(self):
        if self.n < 4 or self.n > 256 or self.n & (self.n-1):
            raise ValueError('Power-of-two N in [4,256] required')
        if tuple(sorted(set(self.profile))) != self.profile or not 1 < len(self.profile) < self.n:
            raise ValueError('Sorted distinct information profile required')
        if any(i < 0 or i >= self.n for i in self.profile):
            raise ValueError('Information coordinate out of range')

    @property
    def rows(self):
        return tuple(j for j in range(self.n) if j not in self.profile and j > self.profile[0])

    @property
    def legal(self):
        return np.asarray([[i < j for i in self.profile] for j in self.rows], dtype=np.float32)

    def code(self, bits):
        bits = np.asarray(bits)
        if bits.shape != self.legal.shape or np.any((bits != 0) & (bits != 1)) or np.any(bits > self.legal):
            raise ValueError('Invalid complete causal support')
        supports = [[] for _ in range(self.n)]
        for r,j in enumerate(self.rows):
            supports[j] = [i for k,i in enumerate(self.profile) if bits[r,k]]
        return dict(actions=[0 if j in self.profile else 2 if supports[j] else 1 for j in range(self.n)],
                    supports=supports, action_scheme='direct_sparse')
