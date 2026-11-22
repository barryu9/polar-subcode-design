"""Resumable fixed-frame SCL comparison with shared noise across saved codes.

The native decoder releases the GIL, so independent frame chunks can run in
threads. Each completed chunk is committed atomically and reused on restart.
Only completed chunks contribute to reported statistics.
"""

import hashlib
import math
from pathlib import Path
import time
import numpy as np
from polar_code.SCL_full_decoder import decode_full_path_batch, simulate_awgn_llr_batch
from select_designs_scl import construction, noiseless_check, wilson

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def evaluate_chunk(start, count, codes, params, config):
    started = time.perf_counter()
    labels = config['labels']
    rows = {label: dict(failure_offsets=[], path_pruning_failures=0,
                       terminal_selection_failures=0, decoded_steps=0) for label in labels}
    joint = np.zeros(1 << len(codes), dtype=np.int64)
    for offset in range(0, count, config['batch_size']):
        size = min(config['batch_size'], count-offset)
        seeds = range(config['seed']+start+offset, config['seed']+start+offset+size)
        llr = simulate_awgn_llr_batch(params.N, config['K'], config['EbN0_dB'], seeds)
        outcomes = np.zeros(size, dtype=np.int64)
        for index, (actions, supports) in enumerate(codes):
            _, loss, steps, decoded = decode_full_path_batch(
                params, actions, supports, llr, config['L'], early_stop=False,
                terminal_margin=0., max_support_size=None, backend='cpp', return_decoded_u=True)
            if np.any(steps != params.N):
                raise AssertionError('A full-path trial stopped before the final decision')
            failures = np.any(decoded != 0, axis=1)
            if np.any((loss >= 0) & ~failures):
                raise AssertionError('A pruned transmitted path was reported as successful')
            record = rows[labels[index]]
            record['failure_offsets'].extend((offset+np.flatnonzero(failures)).tolist())
            record['path_pruning_failures'] += int(np.sum(loss >= 0))
            record['terminal_selection_failures'] += int(np.sum(failures & (loss < 0)))
            record['decoded_steps'] += int(steps.sum())
            outcomes |= failures.astype(np.int64) << index
        joint += np.bincount(outcomes, minlength=len(joint))
    return dict(start=start, frames=count, codes=rows, joint_counts=joint.tolist(),
                elapsed_seconds=time.perf_counter()-started)

def validate_chunk(chunk, start, count, labels):
    if chunk['start'] != start or chunk['frames'] != count or set(chunk['codes']) != set(labels):
        raise ValueError('Checkpoint range or roster mismatch')
    outcomes = np.zeros(count, dtype=np.int64)
    for i, label in enumerate(labels):
        row = chunk['codes'][label]
        offsets = row['failure_offsets']
        if offsets != sorted(set(offsets)) or any(not 0 <= j < count for j in offsets):
            raise ValueError('Invalid failure offsets')
        outcomes[offsets] |= 1 << i
        if row['path_pruning_failures']+row['terminal_selection_failures'] != len(offsets):
            raise ValueError('Inconsistent error counts')
    expected = np.bincount(outcomes, minlength=1 << len(labels)).tolist()
    if chunk['joint_counts'] != expected:
        raise ValueError('Joint outcomes disagree with retained error traces')

def aggregate(chunks, labels):
    frames = sum(c['frames'] for c in chunks)
    if frames == 0:
        return dict(frames_per_code=0, codes={}, paired_comparisons=[])
    joint = np.sum([c['joint_counts'] for c in chunks], axis=0).astype(np.int64)
    codes, pairs = {}, []
    for i, label in enumerate(labels):
        errors = sum(int(value) for mask, value in enumerate(joint) if mask >> i & 1)
        codes[label] = dict(errors=errors, FER=errors/frames, Wilson_95pct=wilson(errors, frames),
            path_pruning_failures=sum(c['codes'][label]['path_pruning_failures'] for c in chunks),
            terminal_selection_failures=sum(c['codes'][label]['terminal_selection_failures'] for c in chunks),
            low_error_count_warning=errors < 100)
        for j in range(i):
            a_only = sum(int(v) for mask, v in enumerate(joint) if (mask >> j & 1) and not (mask >> i & 1))
            b_only = sum(int(v) for mask, v in enumerate(joint) if (mask >> i & 1) and not (mask >> j & 1))
            delta = (a_only-b_only)/frames
            se = math.sqrt(max(0., (a_only+b_only)/frames-delta**2)/(frames-1)) if frames > 1 else None
            pairs.append(dict(a=labels[j], b=label, a_only_errors=a_only, b_only_errors=b_only,
                              FER_a_minus_b=delta,
                              normal_95pct=[delta-1.95996398454*se, delta+1.95996398454*se] if se is not None else None,
                              few_discordant_pairs_warning=a_only+b_only < 30))
    return dict(frames_per_code=frames, codes=codes, paired_comparisons=pairs,
                joint_counts=joint.tolist())
