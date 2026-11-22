"""Actual full-path SCL selection with paired noise and independent confirmation.

GA and spectra do not determine the winner. The selection metric is the error
of the terminal minimum-metric path, not genie-aided list survival.
"""

import math
import numpy as np
from feasible_design_bank import code_fingerprint, generator_rows, validate_design
from polar_code.pc_funcs import pc_init
from polar_code.SCL_full_decoder import decode_full_path_batch, simulate_awgn_llr_batch

def construction(n, profile, columns):
    profile = validate_design(n, profile, columns)
    information = set(profile)
    actions = tuple(0 if j in information else 2 if c else 1 for j, c in enumerate(columns))
    supports = tuple(tuple(i for i in profile if c >> i & 1) for c in columns)
    return actions, supports

def wilson(errors, frames):
    z = 1.959963984540054
    p = errors/frames
    denominator = 1+z*z/frames
    center = (p+z*z/(2*frames))/denominator
    half = z*math.sqrt((p*(1-p)+z*z/(4*frames))/frames)/denominator
    return [max(0., center-half), min(1., center+half)]

def noiseless_check(design, list_size, backend):
    n, k = design['N'], design['K']
    profile, columns = design['profile'], design['columns']
    rng = np.random.default_rng(991)
    actions, supports = construction(n, profile, columns)
    inputs = []
    codewords = []
    from verify_causal_flag_theory import polar_transform
    for message in rng.integers(0, 2, size=(4, k)):
        u = sum(int(bit) << i for bit, i in zip(message, profile))
        u |= sum((sum(int(bit)*(c >> i & 1) for bit, i in zip(message, profile)) % 2) << j
                 for j, c in enumerate(columns))
        x = polar_transform(u, n)
        inputs.append([(u >> j) & 1 for j in range(n)])
        codewords.append([(x >> j) & 1 for j in range(n)])
    llr = rng.uniform(100., 1000., size=(4, n))*(1-2*np.asarray(codewords))
    params = pc_init(n, k, 0, np.asarray(((1, 0), (1, 1)), dtype=int))
    decoded = decode_full_path_batch(params, actions, supports, llr, list_size,
        early_stop=False, terminal_margin=0., max_support_size=None,
        backend=backend, return_decoded_u=True)[3]
    if not np.array_equal(np.asarray(decoded), np.asarray(inputs)):
        raise AssertionError('Noiseless random-message decoding failed')
