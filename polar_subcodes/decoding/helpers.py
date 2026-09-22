"""Vectorized min-sum SCL state updates shared with the native decoder."""
import numpy as np
from typing import Optional
try:
    from . import _scl_step_decoder_cpp as _SCL_CPP
except ImportError:
    _SCL_CPP = None

def scl_cpp_available() -> bool:
    """Return whether the optional CPU C++ SCL extension is importable."""
    return _SCL_CPP is not None


def _update_c_batch(
    C: np.ndarray,
    phi: int,
    N: int,
    lambda_offset: np.ndarray,
    bit_layer_vec: np.ndarray,
    batch_indices: np.ndarray,
    path_indices: np.ndarray,
) -> None:
    """Apply the standard odd-bit partial-sum update for selected paths."""
    if batch_indices.size == 0 or phi % 2 != 1 or phi == N - 1:
        return

    layer = int(bit_layer_vec[phi])
    b = np.asarray(batch_indices, dtype=np.int64)
    paths = np.asarray(path_indices, dtype=np.int64)
    cols_left = 2 * paths
    cols_right = cols_left + 1

    for i_layer in range(layer):
        index_1 = int(lambda_offset[i_layer])
        index_2 = int(lambda_offset[i_layer + 1])
        for beta in range(index_1, 2 * index_1):
            val_sum = (
                C[b, beta - 1, cols_left]
                + C[b, beta - 1, cols_right]
            )
            C[b, beta + index_1 - 1, cols_right] = val_sum % 2
            C[b, beta + index_2 - 1, cols_right] = C[
                b, beta - 1, cols_right
            ]

    index_1 = int(lambda_offset[layer])
    index_2 = int(lambda_offset[layer + 1])
    for beta in range(index_1, 2 * index_1):
        val_sum = C[b, beta - 1, cols_left] + C[b, beta - 1, cols_right]
        C[b, beta + index_1 - 1, cols_left] = val_sum % 2
        C[b, beta + index_2 - 1, cols_left] = C[b, beta - 1, cols_right]


def _propagate_llr_batch(
    pc_params,
    phi: int,
    llr: np.ndarray,
    P: np.ndarray,
    C: np.ndarray,
    L: int,
) -> None:
    """Propagate the LLR tree over the batch and list axes."""
    N = int(pc_params.N)
    n = int(np.log2(N))
    lambda_offset = pc_params.lambda_offset
    llr_layer_vec = pc_params.llr_layer_vec

    # Updating inactive paths is harmless: their PM values exclude them, and
    # an inactive path is overwritten from an active source before reuse.
    for l_index in range(int(L)):
        layer = int(llr_layer_vec[phi])
        if phi == 0:
            index_1 = int(lambda_offset[n - 1])
            beta = np.arange(index_1, dtype=np.int64)
            val1 = llr[:, beta]
            val2 = llr[:, beta + index_1]
            P[:, beta + index_1 - 1, l_index] = (
                np.sign(val1) * np.sign(val2) * np.minimum(np.abs(val1), np.abs(val2))
            )
            for i_layer in range(n - 2, -1, -1):
                index_1 = int(lambda_offset[i_layer])
                index_2 = int(lambda_offset[i_layer + 1])
                beta = np.arange(index_1, dtype=np.int64)
                val1 = P[:, beta + index_2 - 1, l_index]
                val2 = P[:, beta + index_1 + index_2 - 1, l_index]
                P[:, beta + index_1 - 1, l_index] = (
                    np.sign(val1) * np.sign(val2) * np.minimum(np.abs(val1), np.abs(val2))
                )
        elif phi == N // 2:
            index_1 = int(lambda_offset[n - 1])
            beta = np.arange(index_1, dtype=np.int64)
            x_tmp = C[:, beta + index_1 - 1, 2 * l_index]
            P[:, beta + index_1 - 1, l_index] = (
                (1 - 2 * x_tmp) * llr[:, beta] + llr[:, beta + index_1]
            )
            for i_layer in range(n - 2, -1, -1):
                index_1 = int(lambda_offset[i_layer])
                index_2 = int(lambda_offset[i_layer + 1])
                beta = np.arange(index_1, dtype=np.int64)
                val1 = P[:, beta + index_2 - 1, l_index]
                val2 = P[:, beta + index_1 + index_2 - 1, l_index]
                P[:, beta + index_1 - 1, l_index] = (
                    np.sign(val1) * np.sign(val2) * np.minimum(np.abs(val1), np.abs(val2))
                )
        else:
            index_1 = int(lambda_offset[layer])
            index_2 = int(lambda_offset[layer + 1])
            beta = np.arange(index_1, dtype=np.int64)
            x_tmp = C[:, beta + index_1 - 1, 2 * l_index]
            P[:, beta + index_1 - 1, l_index] = (
                (1 - 2 * x_tmp)
                * P[:, beta + index_2 - 1, l_index]
                + P[:, beta + index_1 + index_2 - 1, l_index]
            )
            for i_layer in range(layer - 1, -1, -1):
                index_1 = int(lambda_offset[i_layer])
                index_2 = int(lambda_offset[i_layer + 1])
                beta = np.arange(index_1, dtype=np.int64)
                val1 = P[:, beta + index_2 - 1, l_index]
                val2 = P[:, beta + index_1 + index_2 - 1, l_index]
                P[:, beta + index_1 - 1, l_index] = (
                    np.sign(val1) * np.sign(val2) * np.minimum(np.abs(val1), np.abs(val2))
                )


def _clone_action0_batch(
    phi: int,
    PM: np.ndarray,
    P: np.ndarray,
    C: np.ndarray,
    u: np.ndarray,
    activepath: np.ndarray,
    PM_pair: np.ndarray,
    compare: np.ndarray,
    pc_params,
    L: int,
    info_rows: Optional[np.ndarray] = None,
) -> None:
    """Apply the SCL information-bit split with a vectorized frame axis."""
    B = PM.shape[0]
    N = int(pc_params.N)
    lambda_offset = pc_params.lambda_offset
    bit_layer_vec = pc_params.bit_layer_vec
    active_before = activepath.copy()
    active_after = activepath.copy()
    if info_rows is None:
        info_rows = np.ones(B, dtype=bool)
    else:
        info_rows = np.asarray(info_rows, dtype=bool)

    kill = info_rows[:, None] & ~(compare[:, 0, :] | compare[:, 1, :])
    active_after[kill] = False

    # The legacy decoder pops the largest available killed path index first.
    # Precomputing one clone target per (frame, source path) preserves that
    # order while the numerical state updates remain vectorized over frames.
    clone_targets = np.full((B, int(L)), -1, dtype=np.int64)
    for frame in range(B):
        if not info_rows[frame]:
            continue
        available = list(np.flatnonzero(kill[frame]))
        for l_index in range(int(L)):
            if not active_before[frame, l_index]:
                continue
            state = int(compare[frame, 0, l_index]) * 2 + int(compare[frame, 1, l_index])
            if state == 3 and available:
                clone_targets[frame, l_index] = int(available.pop())

    activepath[...] = active_after

    for l_index in range(int(L)):
        state1 = (
            info_rows & active_after[:, l_index]
            & (~compare[:, 0, l_index]) & compare[:, 1, l_index]
        )
        state2 = (
            info_rows & active_after[:, l_index]
            & compare[:, 0, l_index] & (~compare[:, 1, l_index])
        )

        for state_mask, value, pm_column in (
            (state1, 1, 1),
            (state2, 0, 0),
        ):
            b = np.flatnonzero(state_mask)
            if b.size == 0:
                continue
            PM[b, l_index] = PM_pair[b, pm_column, l_index]
            u[b, phi, l_index] = value
            C[b, 0, 2 * l_index + (phi % 2)] = value
            _update_c_batch(
                C, phi, N, lambda_offset, bit_layer_vec,
                b, np.full(b.size, l_index, dtype=np.int64),
            )

        clone_mask = clone_targets[:, l_index] >= 0
        b = np.flatnonzero(clone_mask)
        if b.size == 0:
            continue
        destination = clone_targets[b, l_index]

        # Advanced indexing pairs each frame with its own destination path.
        P[b, :, destination] = P[b, :, l_index]
        u[b, :, destination] = u[b, :, l_index]
        C[b, :, 2 * destination] = C[b, :, 2 * l_index]
        C[b, :, 2 * destination + 1] = C[b, :, 2 * l_index + 1]
        activepath[b, destination] = True

        PM[b, l_index] = PM_pair[b, 0, l_index]
        PM[b, destination] = PM_pair[b, 1, l_index]
        u[b, phi, l_index] = 0
        u[b, phi, destination] = 1
        C[b, 0, 2 * l_index + (phi % 2)] = 0
        C[b, 0, 2 * destination + (phi % 2)] = 1

        _update_c_batch(
            C, phi, N, lambda_offset, bit_layer_vec,
            b, np.full(b.size, l_index, dtype=np.int64),
        )
        _update_c_batch(
            C, phi, N, lambda_offset, bit_layer_vec,
            b, destination,
        )
