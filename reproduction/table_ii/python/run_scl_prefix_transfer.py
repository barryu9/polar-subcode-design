"""Apply spectrum-neutral prefix completion to saved short polar subcodes.

Each profile is held fixed. A separate validation set chooses a completion
scope before matched-noise holdout testing. Empty prefix domains are recorded
as inapplicable rather than interpreted as failed optimization.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

os.environ.setdefault('OMP_NUM_THREADS', '1')

os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import evaluate_scl_code_suite as suite
from polar_code import _scl_step_decoder_cpp
from polar_code.pc_funcs import pc_init
from run_diverse_profile_suite import identities, read, write
from select_designs_scl import construction, noiseless_check

def evaluate(designs, directory, frames, snr, seed, workers):
    directory.mkdir(parents=True, exist_ok=True)
    n, k = designs[0]['N'], designs[0]['K']
    if any((d['N'], d['K']) != (n, k) for d in designs):
        raise ValueError('The FER roster has inconsistent dimensions')
    labels = [d['label'] for d in designs]
    config = dict(N=n, K=k, L=32, EbN0_dB=snr, seed=seed, frames_per_code=frames,
        batch_size=128, chunk_size=1000, labels=labels,
        design_ids=[d['name'] for d in designs],
        native_decoder_sha256=suite.sha256(_scl_step_decoder_cpp.__file__),
        channel='BPSK AWGN; all-zero codeword; no CRC; rate K/N',
        decoder='Existing native min-sum full-path SCL; terminal minimum-metric decision',
        early_stop=False, support_size_cap=None, shared_noise=True,
        frame_seed='seed + global zero-based frame index')
    manifest = directory/'manifest.json'
    if manifest.exists() and read(manifest)['configuration'] != config:
        raise ValueError('The existing evaluation protocol differs')
    write(manifest, dict(configuration=config, roster=designs))
    if (directory/'result.json').exists():
        saved = read(directory/'result.json')
        if saved['complete']:
            return saved
    codes = [construction(n, d['profile'], d['columns']) for d in designs]
    params = pc_init(n, k, 0, np.asarray(((1, 0), (1, 1)), dtype=int))
    checkpoint = directory/'chunks'
    checkpoint.mkdir(exist_ok=True)
    chunks, pending = [], []
    for start in range(0, frames, 1000):
        count, path = min(1000, frames-start), checkpoint/f'{start:09d}.json'
        if path.exists():
            chunk = read(path)
            suite.validate_chunk(chunk, start, count, labels)
            chunks.append(chunk)
        else:
            pending.append((start, count, path))
    started, last = time.perf_counter(), 0.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(suite.evaluate_chunk, s, c, codes, params, config): (s, c, p)
                 for s, c, p in pending}
        for task in as_completed(tasks):
            start, count, path = tasks[task]
            chunk = task.result()
            suite.validate_chunk(chunk, start, count, labels)
            write(path, chunk)
            chunks.append(chunk)
            elapsed = time.perf_counter()-started
            if elapsed-last >= 30 or len(chunks)*1000 >= frames:
                progress = suite.aggregate(chunks, labels)
                write(directory/'progress.json', dict(**progress, configuration=config))
                print(json.dumps(dict(stage=directory.name, case=directory.parent.name,
                    frames=progress['frames_per_code'], target=frames,
                    errors={l: r['errors'] for l, r in progress['codes'].items()})), flush=True)
                last = elapsed
    result = suite.aggregate(chunks, labels)
    if result['frames_per_code'] != frames:
        raise AssertionError('The fixed-frame evaluation is incomplete')
    for label in labels:
        errors = np.zeros(frames, dtype=bool)
        for chunk in chunks:
            errors[chunk['start']+np.asarray(chunk['codes'][label]['failure_offsets'], dtype=int)] = True
        if int(errors.sum()) != result['codes'][label]['errors']:
            raise AssertionError('Frame outcomes disagree with the aggregate')
        np.save(directory/(label+'_errors.npy'), errors)
    result.update(configuration=config, complete=True, seconds=time.perf_counter()-started)
    write(directory/'result.json', result)
    return result
