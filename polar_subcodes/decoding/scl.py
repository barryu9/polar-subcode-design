"""General causal polar-subcode SCL decoding, with min-sum path metrics.

The public API decodes arbitrary transmitted messages. It does not use the
all-zero-path survival shortcut used by some research evaluation drivers.
All list paths enforce the full dynamic expression, without a support cap.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np

from .full import decode_full_path_batch, full_scl_backend_name


@lru_cache(maxsize=16)
def parameters(n):
    llr_layer = np.zeros(n,dtype=np.int64)
    bit_layer = np.zeros(n,dtype=np.int64)
    for i in range(1,n):
        llr_layer[i] = (i & -i).bit_length()-1
    for i in range(n):
        value = i//2
        while value & 1:
            bit_layer[i] += 1
            value >>= 1
    return SimpleNamespace(N=n,lambda_offset=2**np.arange(n.bit_length(),dtype=np.int64),
                           llr_layer_vec=llr_layer,bit_layer_vec=bit_layer)


def decode(code,llr,list_size=64,backend='auto'):
    """Return decoded polar inputs, shape [frames,N], from channel LLRs.

    Positive LLR favors bit zero. The C++ and NumPy implementations use the
    same min-sum f operation and max-log path increment. No CRC is applied.
    """
    data = np.asarray(llr,dtype=np.float64)
    if data.ndim==1:
        data = data.reshape(1,-1)
    if data.ndim!=2 or data.shape[1]!=code.n or not np.all(np.isfinite(data)):
        raise ValueError('LLRs must be a finite [frames,N] array')
    if type(list_size) is not int or list_size<1:
        raise ValueError('List size must be a positive integer')
    if not len(data):
        return np.empty((0,code.n),dtype=np.int8)
    if code.n==1:
        return (data<0).astype(np.int8) if code.k else np.zeros(data.shape,dtype=np.int8)
    actions = [0 if i in code.profile else 2 if code.columns[i] else 1 for i in range(code.n)]
    _,_,_,decoded = decode_full_path_batch(parameters(code.n),actions,code.supports,data,
        list_size,early_stop=False,terminal_margin=0,backend=backend,
        max_support_size=None,return_decoded_u=True)
    return decoded
