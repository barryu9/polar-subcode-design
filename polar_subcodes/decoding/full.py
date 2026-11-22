# -*- coding: utf-8 -*-
"""One-shot full-path SCL decoding for a fixed causal construction.

The optional implementation is the CPU C++ backend exposed by
``_scl_step_decoder_cpp.decode_full_pxb``.  A NumPy implementation is kept as
a correctness fallback and deliberately does not call the step-decoder API.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np

from .helpers import (
    _clone_action0_batch,
    _propagate_llr_batch,
    _update_c_batch,
    scl_cpp_available,
)

try:  # pragma: no cover - availability depends on the local build
    from . import _scl_step_decoder_cpp as _SCL_CPP
except Exception:  # pragma: no cover
    _SCL_CPP = None


def _requested_backend(backend: str | None) -> str:
    value = (
        str(backend)
        if backend is not None
        else os.environ.get("SCL_FULL_BACKEND", "auto")
    ).strip().lower()
    if value in ("c++", "native"):
        value = "cpp"
    if value == "python":
        value = "numpy"
    if value not in ("auto", "numpy", "cpp"):
        raise ValueError("backend must be one of 'auto', 'numpy', or 'cpp'")
    return value


def full_scl_backend_name(backend: str | None = None) -> str:
    """Return the effective backend name without executing a decode."""
    requested = _requested_backend(backend)
    if requested == "auto":
        return "cpp" if _SCL_CPP is not None else "numpy"
    return requested


def simulate_awgn_llr_batch(
    N: int,
    K: int,
    EbN0_dB: float,
    frame_seeds: Sequence[int],
) -> np.ndarray:
    """Generate the seeded all-zero-codeword LLR batch used by evaluation."""
    N = int(N)
    K = int(K)
    if N <= 0 or K <= 0 or K >= N:
        raise ValueError("N and K must satisfy 0 < K < N")
    seeds = [int(value) for value in frame_seeds]
    rate = K / N
    ebn0_linear = 10.0 ** (float(EbN0_dB) / 10.0)
    noise_std = float(np.sqrt(1.0 / (2.0 * rate * ebn0_linear)))
    if not seeds:
        return np.empty((0, N), dtype=np.float64)
    noise = np.vstack([
        np.random.default_rng(seed).normal(0.0, noise_std, N)
        for seed in seeds
    ])
    return np.ascontiguousarray(2.0 * (1.0 + noise) / (noise_std ** 2))


def pack_construction_expressions(
    actions: Sequence[int],
    supports: Sequence[Sequence[int]],
    max_support_size: int | None = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack one fixed construction into C++-friendly action/expression arrays."""
    action_values = np.asarray([int(x) for x in actions], dtype=np.int8).reshape(-1)
    if action_values.size == 0:
        raise ValueError("construction cannot be empty")
    if len(supports) != int(action_values.size):
        raise ValueError("actions and supports must have the same length")
    if np.any((action_values < 0) | (action_values > 32)):
        raise ValueError("actions must be in the range 0..32")

    # The native interface uses information/static/dynamic opcodes 0/1/2.
    # Historical dynamic IDs are normalized at this internal boundary.
    decoder_action_values = np.where(action_values >= 2, 2, action_values).astype(np.int8)
    support_values = []
    for phi, (action, support) in enumerate(zip(action_values, supports)):
        current = tuple(int(x) for x in support)
        if len(set(current)) != len(current):
            raise ValueError(f"duplicate support index at position {phi}")
        if any(index < 0 or index >= phi for index in current):
            raise ValueError(f"noncausal support at position {phi}: {current}")
        if max_support_size is not None and len(current) > int(max_support_size):
            raise ValueError("support size exceeds the configured bound")
        support_values.append(
            current if int(action) >= 2 else tuple()
        )

    width = max(1, max(len(row) for row in support_values))
    expression_indices = np.full(
        (int(action_values.size), width), -1, dtype=np.int64
    )
    for phi, support in enumerate(support_values):
        if support:
            expression_indices[phi, :len(support)] = np.asarray(
                support, dtype=np.int64
            )
    return np.ascontiguousarray(decoder_action_values), np.ascontiguousarray(expression_indices)


def _validate_inputs(
    actions: np.ndarray,
    expressions: np.ndarray,
    llr: np.ndarray,
    L: int,
) -> None:
    N = int(actions.size)
    if expressions.ndim != 2 or expressions.shape[0] != N:
        raise ValueError("expressions must have shape [N,S]")
    if expressions.shape[1] < 1:
        raise ValueError("expressions must have at least one padded slot")
    if llr.ndim != 2 or llr.shape[1] != N:
        raise ValueError("llr must have shape [B,N]")
    if int(L) <= 0:
        raise ValueError("L must be positive")
    valid = expressions >= 0
    row_indices = np.broadcast_to(
        np.arange(N, dtype=np.int64)[:, None], expressions.shape
    )
    if np.any(expressions[valid] >= row_indices[valid]):
        raise ValueError("dynamic expression contains a noncausal index")


def _correct_path_lost(u: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Return one flag per frame for whether the all-zero path disappeared."""
    has_correct = np.any(active & (np.sum(u, axis=1) == 0), axis=1)
    return ~has_correct


def _final_failed(
    pm: np.ndarray,
    u: np.ndarray,
    active: np.ndarray,
    terminal_margin: float,
) -> np.ndarray:
    B = pm.shape[0]
    correct = active & (np.sum(u, axis=1) == 0)
    has_correct = np.any(correct, axis=1)
    correct_index = np.argmax(correct, axis=1)
    rows = np.arange(B, dtype=np.int64)
    correct_pm = pm[rows, correct_index]
    best_pm = np.min(np.where(active, pm, np.inf), axis=1)
    return (~has_correct) | (correct_pm > best_pm + float(terminal_margin))


def _decode_numpy(
    pc_params,
    actions: np.ndarray,
    expressions: np.ndarray,
    llr: np.ndarray,
    L: int,
    early_stop: bool,
    terminal_margin: float,
    return_decoded_u: bool,
    return_list: bool = False,
):
    """Decode a fixed path without calling the step-decoder entry point."""
    B, N = llr.shape
    L = int(L)
    pm = np.zeros((B, L), dtype=np.float64)
    p = np.zeros((B, N - 1, L), dtype=np.float64)
    c = np.zeros((B, N - 1, 2 * L), dtype=np.int8)
    u = np.zeros((B, N, L), dtype=np.int8)
    active = np.zeros((B, L), dtype=bool)
    active[:, 0] = True
    failed = np.zeros(B, dtype=bool)
    first_loss = np.full(B, -1, dtype=np.int64)
    steps = np.zeros(B, dtype=np.int64)

    for phi, action in enumerate(actions):
        live = ~failed if early_stop else np.ones(B, dtype=bool)
        if not np.any(live):
            break

        # Keep already-terminated rows out of the SCL state transition while
        # retaining their result in the output arrays.
        active[~live] = False
        _propagate_llr_batch(
            pc_params, int(phi), llr, p, c, L
        )
        p_val = p[:, 0, :]

        if int(action) == 0:
            max_float = np.finfo(np.float64).max
            pm_pair = np.full((B, 2, L), max_float, dtype=np.float64)
            hard = p_val < 0
            pm_pair[:, 0, :] = np.where(
                active, pm + np.abs(p_val) * hard, pm_pair[:, 0, :]
            )
            pm_pair[:, 1, :] = np.where(
                active, pm + np.abs(p_val) * (~hard), pm_pair[:, 1, :]
            )
            active_count = np.sum(active, axis=1).astype(np.int64)
            valid = active_count > 0
            middle = np.maximum(1, np.minimum(2 * active_count, L))
            # A threshold can select more than L paths when metrics tie.
            # Stable candidate order breaks ties by (bit, parent path).
            order = np.argsort(pm_pair.reshape(B, 2 * L),axis=1,kind='stable')
            selected = np.zeros((B,2*L),dtype=bool)
            np.put_along_axis(selected,order,np.arange(2*L)[None,:]<middle[:,None],axis=1)
            compare = selected.reshape(B,2,L)
            compare[~valid] = False
            _clone_action0_batch(
                int(phi), pm, p, c, u, active, pm_pair, compare,
                pc_params, L,
            )
        else:
            safe = np.where(expressions[phi] >= 0, expressions[phi], 0)
            values = np.take(u, safe, axis=1)
            values = np.where(expressions[phi][None, :, None] >= 0, values, 0)
            target = np.bitwise_xor.reduce(values, axis=1).astype(np.int8)
            hard = (p_val < 0).astype(np.int8)
            mismatch = hard != target
            pm[...] = pm + np.where(active, np.abs(p_val) * mismatch, 0.0)
            for path in range(L):
                rows = np.flatnonzero(active[:, path])
                if rows.size == 0:
                    continue
                values_path = target[rows, path]
                u[rows, int(phi), path] = values_path
                c[rows, 0, 2 * path + (int(phi) % 2)] = values_path
                _update_c_batch(
                    c, int(phi), int(pc_params.N), pc_params.lambda_offset,
                    pc_params.bit_layer_vec, rows,
                    np.full(rows.size, path, dtype=np.int64),
                )

        lost_now = _correct_path_lost(u, active) & live
        new_loss = lost_now & (first_loss < 0)
        first_loss[new_loss] = int(phi)
        failed |= lost_now
        steps[live] = int(phi + 1)
        if early_stop:
            active[lost_now] = False

    if return_list:
        # Keep every surviving candidate for terminal CRC selection.
        return u.transpose(0, 2, 1), pm, active
    completed = steps == N
    final_failed = np.zeros(B, dtype=bool)
    if np.any(completed):
        final_failed[completed] = _final_failed(
            pm[completed], u[completed], active[completed], terminal_margin
        )
    failed |= final_failed
    steps[steps == 0] = N
    if return_decoded_u:
        masked_pm = np.where(active, pm, np.inf)
        best_path = np.argmin(masked_pm, axis=1)
        rows = np.arange(B, dtype=np.int64)[:, None]
        bits = np.arange(N, dtype=np.int64)[None, :]
        decoded_u = np.ascontiguousarray(u[rows, bits, best_path[:, None]])
        return failed, first_loss, steps, decoded_u
    return failed, first_loss, steps


def decode_full_path_batch(
    pc_params,
    actions: Sequence[int],
    supports: Sequence[Sequence[int]],
    llr: np.ndarray,
    L: int,
    early_stop: bool = True,
    terminal_margin: float = 1e-4,
    backend: str | None = None,
    max_support_size: int | None = 3,
    return_decoded_u: bool = False,
):
    """Decode one complete fixed construction for many AWGN frames.

    Returns ``(failed, first_loss_position, decoded_steps)`` with one value per
    frame.  ``first_loss_position`` is -1 when the all-zero path survives all
    SCL steps and the frame only fails the final PM-best test.
    """
    action_array, expression_array = pack_construction_expressions(
        actions,
        supports,
        max_support_size=max_support_size,
    )
    llr_array = np.ascontiguousarray(np.asarray(llr, dtype=np.float64))
    _validate_inputs(action_array, expression_array, llr_array, int(L))

    requested = _requested_backend(backend)
    use_cpp = requested == "cpp" or (requested == "auto" and _SCL_CPP is not None)
    if use_cpp:
        if _SCL_CPP is None:
            raise ImportError(
                "full-path C++ SCL backend is not built; run "
                "python tools/build_native.py --scl"
            )
        return _SCL_CPP.decode_full_pxb(
            int(pc_params.N),
            np.ascontiguousarray(pc_params.lambda_offset, dtype=np.int64),
            np.ascontiguousarray(pc_params.llr_layer_vec, dtype=np.int64),
            np.ascontiguousarray(pc_params.bit_layer_vec, dtype=np.int64),
            action_array,
            expression_array,
            llr_array,
            int(L),
            bool(early_stop),
            float(terminal_margin),
            bool(return_decoded_u),
        )

    return _decode_numpy(
        pc_params,
        action_array,
        expression_array,
        llr_array,
        int(L),
        bool(early_stop),
        float(terminal_margin),
        bool(return_decoded_u),
    )


__all__ = [
    "decode_full_path_batch",
    "full_scl_backend_name",
    "pack_construction_expressions",
    "simulate_awgn_llr_batch",
]
