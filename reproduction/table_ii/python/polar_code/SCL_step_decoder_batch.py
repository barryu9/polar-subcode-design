"""Vectorized SCL step decoder for one construction trace and many frames.

The batch axis is the AWGN-frame axis.  All frames receive the same action
and causal support at a step, while PM/P/C/u and active paths remain
independent for every frame.  The implementation intentionally contains no
reliability profile or channel-dependent policy feature.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from . import _scl_step_decoder_cpp as _SCL_CPP
except ImportError:  # The optional extension is built separately in place.
    _SCL_CPP = None


def _cpp_arrays_compatible(
    llr: np.ndarray,
    PM: np.ndarray,
    P: np.ndarray,
    C: np.ndarray,
    u: np.ndarray,
    activepath: np.ndarray,
) -> bool:
    """Check the zero-copy NumPy layout expected by the C++ backend."""
    arrays = (llr, PM, P, C, u, activepath)
    dtypes = (np.float64, np.float64, np.float64, np.int8, np.int8, np.bool_)
    return all(
        isinstance(array, np.ndarray)
        and array.flags.c_contiguous
        and array.flags.writeable
        and array.dtype == dtype
        for array, dtype in zip(arrays, dtypes)
    )


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


def _soft_rank_batch(
    PM: np.ndarray,
    u: np.ndarray,
    activepath: np.ndarray,
    sigma: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-frame path-loss flags and continuous potentials."""
    active = np.asarray(activepath, dtype=bool)
    path_sums = np.sum(u, axis=1)
    correct = active & (path_sums == 0)
    active_count = np.sum(active, axis=1)
    has_correct = np.any(correct, axis=1)
    lost = ~has_correct

    correct_index = np.argmax(correct, axis=1).astype(np.int64)
    row = np.arange(PM.shape[0], dtype=np.int64)
    pm_correct = PM[row, correct_index]
    competitors = active.copy()
    competitors[row, correct_index] = False
    competitor_pm = np.min(np.where(competitors, PM, np.inf), axis=1)
    raw_gap = competitor_pm - pm_correct
    potential = np.zeros(PM.shape[0], dtype=np.float64)
    valid_gap = has_correct & (active_count > 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        potential[valid_gap] = (
            raw_gap[valid_gap]
            / (float(sigma) + np.abs(raw_gap[valid_gap]))
        )
    potential = np.where(has_correct & (active_count <= 1), 1.0, potential)
    potential = np.where(lost, -1.0, potential)
    return lost.astype(bool), potential.astype(np.float64)


def _branch_prefix_statistics(
    PM_pair: np.ndarray,
    activepath: np.ndarray,
    info_rows: np.ndarray,
) -> np.ndarray:
    """Compute diagnostic entropy of information candidates before pruning.

    ``PM_pair`` is the same 2L candidate buffer already needed by ordinary
    SCL.  This helper intentionally does not create or advance another list:
    it only summarizes the candidates while they are still available, then
    the caller proceeds with the normal L-path pruning.

    The returned columns are ``[H_pre, candidate_count]``.  Non-information
    rows receive ``NaN, 0`` because no pre-pruning candidate set exists.
    """
    PM_pair = np.asarray(PM_pair)
    activepath = np.asarray(activepath, dtype=bool)
    info_rows = np.asarray(info_rows, dtype=bool)
    M, _, L = PM_pair.shape
    # PM_pair is laid out as [candidate_bit, parent_path], i.e. all zero
    # candidates followed by all one candidates.  Keep the mask in the same
    # order so inactive parents never enter the entropy calculation.
    values = PM_pair.reshape(M, 2 * L)
    valid = np.concatenate((activepath, activepath), axis=1) & info_rows[:, None]
    finite_values = np.where(valid, values, np.inf)
    minimum = np.min(finite_values, axis=1)
    finite_rows = info_rows & np.isfinite(minimum)
    shifted = np.where(
        valid,
        np.clip(-(values - minimum[:, None]), -745.0, 0.0),
        -np.inf,
    )
    weights = np.where(valid, np.exp(shifted), 0.0)
    weight_sum = np.sum(weights, axis=1)
    probabilities = np.divide(
        weights,
        weight_sum[:, None],
        out=np.zeros_like(weights),
        where=weight_sum[:, None] > 0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(
            np.where(probabilities > 0.0, probabilities * np.log2(probabilities), 0.0),
            axis=1,
        )
    entropy = np.where(finite_rows, entropy, np.nan).astype(np.float64)
    counts = np.where(
        finite_rows,
        2.0 * np.sum(activepath, axis=1),
        0.0,
    ).astype(np.float64)
    return np.stack((entropy, counts), axis=1)


def _surviving_prefix_statistics(
    PM: np.ndarray,
    activepath: np.ndarray,
    selected_rows: np.ndarray,
) -> np.ndarray:
    """Compute entropy over current surviving paths after an F/D update."""
    PM = np.asarray(PM, dtype=np.float64)
    activepath = np.asarray(activepath, dtype=bool)
    selected_rows = np.asarray(selected_rows, dtype=bool)
    valid = activepath & selected_rows[:, None]
    finite_values = np.where(valid, PM, np.inf)
    minimum = np.min(finite_values, axis=1)
    finite_rows = selected_rows & np.isfinite(minimum)
    shifted = np.where(
        valid,
        np.clip(-(PM - minimum[:, None]), -745.0, 0.0),
        -np.inf,
    )
    weights = np.where(valid, np.exp(shifted), 0.0)
    weight_sum = np.sum(weights, axis=1)
    probabilities = np.divide(
        weights,
        weight_sum[:, None],
        out=np.zeros_like(weights),
        where=weight_sum[:, None] > 0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(
            np.where(probabilities > 0.0, probabilities * np.log2(probabilities), 0.0),
            axis=1,
        )
    entropy = np.where(finite_rows, entropy, 0.0).astype(np.float64)
    counts = np.where(
        finite_rows,
        np.sum(activepath, axis=1),
        0.0,
    ).astype(np.float64)
    return np.stack((entropy, counts), axis=1)


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


def pc_SCL_step_decoder_pxb(
    pc_params,
    phi: int,
    actions: Sequence[int],
    expression_indices: np.ndarray,
    llr: np.ndarray,
    L: int,
    PM: np.ndarray,
    P: np.ndarray,
    C: np.ndarray,
    u: np.ndarray,
    activepath: np.ndarray,
    pbrs_sigma: float = 10.0,
    backend: Optional[str] = None,
    return_prefix_stats: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Decode one bit for P traces x B frames in one unified kernel.

    The first axis of every decoder array is a flattened ``P*B`` row axis.
    ``actions[m]`` selects information splitting when it is zero and the
    fixed/dynamic-frozen branch otherwise.  ``expression_indices[m, :]`` is a
    padded list of causal information-bit indices; its XOR is the dynamic
    frozen expression supplied by the caller.  Invalid entries are ``-1``.

    This API deliberately does not infer the expression from an action label:
    the environment owns construction semantics and passes the expression
    explicitly, while the kernel only evaluates it on each SCL path.

    ``backend`` may be ``"numpy"``, ``"cpp"`` or ``"auto"``.  The default
    is controlled by ``SCL_KERNEL_BACKEND`` and otherwise uses ``"auto"``;
    when the optional extension is available, auto selects the CPU C++
    implementation for contiguous float64/int8/bool decoder buffers.  When
    ``return_prefix_stats`` is true, a fourth return value has shape ``[M, 4]``
    and contains ``[H_surv, surviving_count, H_pre, expanded_count]``.
    ``H_surv`` is always computed after processing and pruning from the real
    surviving SCL-L paths and is suitable for PBRS.  ``H_pre`` is computed
    only for information rows from the existing 2L branch candidates and is
    diagnostic only.  No independent decoder state is advanced.
    """
    actions = np.asarray(actions, dtype=np.int8).reshape(-1)
    expression_indices = np.asarray(expression_indices, dtype=np.int64)
    llr = np.asarray(llr)
    PM = np.asarray(PM)
    P = np.asarray(P)
    C = np.asarray(C)
    u = np.asarray(u)
    activepath = np.asarray(activepath, dtype=bool)

    requested_backend = (
        str(backend)
        if backend is not None
        else os.environ.get("SCL_KERNEL_BACKEND", "auto")
    ).strip().lower()
    if requested_backend in ("c++", "native"):
        requested_backend = "cpp"
    if requested_backend not in ("auto", "numpy", "python", "cpp"):
        raise ValueError(
            "backend must be one of 'auto', 'numpy', 'python', or 'cpp'"
        )
    if requested_backend == "python":
        requested_backend = "numpy"

    cpp_compatible = _cpp_arrays_compatible(
        llr, PM, P, C, u, activepath
    )
    use_cpp = requested_backend == "cpp" or (
        requested_backend == "auto" and _SCL_CPP is not None
    )
    if use_cpp:
        if _SCL_CPP is None:
            raise ImportError(
                "SCL C++ backend is not built; run "
                "python setup_scl_cpp.py build_ext --inplace"
            )
        if not cpp_compatible:
            if requested_backend == "cpp":
                raise ValueError(
                    "C++ backend requires contiguous writable decoder arrays "
                    "with llr/PM/P=float64, C/u=int8, activepath=bool"
                )
        else:
            return _SCL_CPP.step_pxb(
                int(pc_params.N),
                np.ascontiguousarray(pc_params.lambda_offset, dtype=np.int64),
                np.ascontiguousarray(pc_params.llr_layer_vec, dtype=np.int64),
                np.ascontiguousarray(pc_params.bit_layer_vec, dtype=np.int64),
                int(phi),
                actions,
                expression_indices,
                llr,
                int(L),
                PM,
                P,
                C,
                u,
                activepath,
                float(pbrs_sigma),
                bool(return_prefix_stats),
            )

    if actions.ndim != 1:
        raise ValueError("actions must have shape [P*B]")
    M = actions.shape[0]
    if expression_indices.ndim != 2 or expression_indices.shape[0] != M:
        raise ValueError("expression_indices must have shape [P*B,S]")
    if expression_indices.shape[1] < 1:
        raise ValueError("expression_indices needs at least one padded slot")
    if llr.shape[0] != M or PM.shape != (M, int(L)):
        raise ValueError("batched P*B arrays have inconsistent leading dimensions")
    if P.shape[0] != M or C.shape[0] != M or u.shape[0] != M:
        raise ValueError("batched P*B decoder arrays have inconsistent leading dimensions")
    if activepath.shape != (M, int(L)):
        raise ValueError("activepath shape does not match PM")
    if np.any((actions < 0) | (actions > 4)):
        raise ValueError("actions must be in the range 0..4")

    valid_expression = expression_indices >= 0
    if np.any(expression_indices[valid_expression] >= int(phi)):
        raise ValueError("dynamic frozen expression is not causal")
    if np.any(expression_indices[valid_expression] >= int(pc_params.N)):
        raise ValueError("dynamic frozen expression index is out of range")

    _, previous_potential = _soft_rank_batch(PM, u, activepath, sigma=pbrs_sigma)
    _propagate_llr_batch(pc_params, int(phi), llr, P, C, int(L))

    info_rows = actions == 0
    PM_pair = np.full(
        (M, 2, int(L)),
        np.finfo(PM.dtype).max if np.issubdtype(PM.dtype, np.floating) else np.finfo(np.float64).max,
        dtype=PM.dtype if np.issubdtype(PM.dtype, np.floating) else np.float64,
    )
    p_val = P[:, 0, :]
    hard = p_val < 0
    PM_pair[:, 0, :] = np.where(
        info_rows[:, None] & activepath,
        PM + np.abs(p_val) * hard,
        PM_pair[:, 0, :],
    )
    PM_pair[:, 1, :] = np.where(
        info_rows[:, None] & activepath,
        PM + np.abs(p_val) * (~hard),
        PM_pair[:, 1, :],
    )
    active_count = np.sum(activepath, axis=1).astype(np.int64)
    valid_rows = active_count > 0
    middle = np.maximum(1, np.minimum(2 * active_count, int(L)))
    PM_sort = np.sort(PM_pair.reshape(M, 2 * int(L)), axis=1)
    PM_cv = PM_sort[np.arange(M), middle - 1]
    compare = PM_pair <= PM_cv[:, None, None]
    compare[~valid_rows] = False
    pre_pruning_stats = (
        _branch_prefix_statistics(PM_pair, activepath, info_rows)
        if return_prefix_stats
        else None
    )
    _clone_action0_batch(
        int(phi), PM, P, C, u, activepath, PM_pair, compare,
        pc_params, int(L), info_rows=info_rows,
    )

    noninfo_rows = ~info_rows
    if np.any(noninfo_rows):
        safe_indices = np.where(valid_expression, expression_indices, 0)
        gather_indices = np.broadcast_to(
            safe_indices[:, :, None],
            (M, expression_indices.shape[1], int(L)),
        )
        expression_values = np.take_along_axis(u, gather_indices, axis=1)
        expression_values = np.where(
            valid_expression[:, :, None], expression_values, 0
        )
        target = np.bitwise_xor.reduce(expression_values, axis=1).astype(np.int8)
        mismatch = (p_val < 0).astype(np.int8) != target
        PM[...] = PM + np.where(
            noninfo_rows[:, None] & activepath,
            np.abs(p_val) * mismatch,
            0.0,
        )

        for l_index in range(int(L)):
            row_mask = noninfo_rows & activepath[:, l_index]
            b = np.flatnonzero(row_mask)
            if b.size == 0:
                continue
            values = target[b, l_index]
            u[b, int(phi), l_index] = values
            C[b, 0, 2 * l_index + (int(phi) % 2)] = values
            _update_c_batch(
                C, int(phi), int(pc_params.N), pc_params.lambda_offset,
                pc_params.bit_layer_vec, b,
                np.full(b.size, l_index, dtype=np.int64),
            )

    lost, current_potential = _soft_rank_batch(PM, u, activepath, sigma=pbrs_sigma)
    if return_prefix_stats:
        assert pre_pruning_stats is not None
        surviving_stats = _surviving_prefix_statistics(
            PM, activepath, np.ones(M, dtype=bool)
        )
        prefix_stats = np.concatenate(
            (surviving_stats, pre_pruning_stats), axis=1
        )
        return lost, current_potential, previous_potential, prefix_stats
    return lost, current_potential, previous_potential


__all__ = ["pc_SCL_step_decoder_pxb"]
