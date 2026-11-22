"""Complete Plotkin counting with exact radius-eight subset convolutions."""

from collections import Counter
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import time
from extended_exact_spectrum import measure
from native_paths import library_path
from feasible_design_bank import generator_rows
from obstruction_support_search import enumerate_words
from plotkin_relative_structure import RelativePlotkinSpectrum
from run_diverse_profile_suite import write
from verify_causal_flag_theory import polar_transform

ROOT = Path(__file__).resolve().parents[1]

LIBRARY = library_path('_relative_subset_counter_v2')

def complete_catalogue(n, profile, columns, cutoff, work):
    if n <= 256:
        return enumerate_words(n, profile, columns, cutoff, work)
    from wide_word_catalogue import enumerate_words as wide_enumerate
    return wide_enumerate(n, profile, columns, cutoff, work)

def build():
    source = ROOT/'cpp/relative_subset_counter.cpp'
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < source.stat().st_mtime:
        compiler = os.environ.get('CXX') or shutil.which('g++') or shutil.which('clang++')
        if not compiler and os.name == 'nt':
            compiler = str(Path(os.environ.get('MW_MINGW64_LOC', 'C:/MinGW64'))/'bin/g++.exe')
        if not compiler:
            raise RuntimeError("Install GCC/Clang or set CXX to a C++17 compiler.")
        subprocess.run([compiler, '-O3', '-std=c++17', '-shared', str(source),
                        *(['-static-libgcc', '-static-libstdc++'] if os.name == 'nt' else ['-fPIC']), '-o', str(LIBRARY)], check=True)
    return LIBRARY

def subset_counts(columns, syndrome, radius):
    if not 0 <= radius <= 8 or len(columns) > 1024 or (len(columns)>128 and radius>4):
        raise ValueError('Expected radius <= 8, or radius <= 4 for more than 128 columns')
    if any(v < 0 or v.bit_length() > 128 for v in [syndrome, *columns]):
        raise ValueError('Expected unsigned syndromes of at most 128 bits')
    library = ctypes.CDLL(str(build()))
    ptr = ctypes.POINTER(ctypes.c_uint64)
    library.relative_counts.argtypes = [ctypes.c_int, ptr, ptr, ctypes.c_uint64,
                                       ctypes.c_uint64, ctypes.c_int, ptr]
    library.relative_counts.restype = ctypes.c_int
    mask = (1 << 64)-1
    low = (ctypes.c_uint64*len(columns))(*(v & mask for v in columns))
    high = (ctypes.c_uint64*len(columns))(*(v >> 64 for v in columns))
    output = (ctypes.c_uint64*(radius+1))()
    status = library.relative_counts(len(columns), low, high, syndrome & mask,
                                     syndrome >> 64, radius, output)
    if status:
        raise ArithmeticError(f'Exact subset counter failed with status {status}')
    return list(output)

def primal_catalogue(n, profile, columns, cutoff):
    rows = generator_rows(n, profile, columns)
    inputs = [(1 << i) | sum(((c >> i) & 1) << j for j, c in enumerate(columns))
              for i in profile]
    words, u, x = [], 0, 0
    for ordinal in range(1, 1 << len(profile)):
        q = (ordinal & -ordinal).bit_length()-1
        u ^= inputs[q]
        x ^= rows[q]
        weight = x.bit_count()
        if weight <= cutoff:
            words.append((u, weight))
    return words

def inspect_fast(model, left_input, cutoff):
    """Use quotient residues directly, without constructing an annihilator."""
    left = polar_transform(left_input, model.half)
    weight = left.bit_count()
    basis = {}
    inside = [i for i in range(model.half) if left >> i & 1]
    for i in inside:
        value = model.check_columns[i]
        while value:
            pivot = value.bit_length()-1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    ordered = sorted(basis, reverse=True)

    def residue(value):
        for pivot in ordered:
            if value >> pivot & 1:
                value ^= basis[pivot]
        return value

    cross_input = sum(((value & left_input).bit_count() & 1) << i
                      for i,value in enumerate(model.columns[model.half:]))
    cross_word = polar_transform(cross_input, model.half)
    syndrome = 0
    for i in range(model.half):
        if cross_word >> i & 1:
            syndrome ^= model.check_columns[i]
    syndrome = residue(syndrome)
    radius = (cutoff-weight)//2
    if radius == 0:
        counts = [int(syndrome == 0)]
    else:
        labels = [residue(model.check_columns[i]) for i in range(model.half)
                  if not (left >> i & 1)]
        counts = subset_counts(labels, syndrome, radius)
    kernel = weight-len(basis)
    return dict(left_input=left_input,left_weight=weight,radius=radius,
                projected_rank=model.redundancy-len(basis),shortening_dimension=kernel,
                spectrum={weight+2*r:count << kernel for r,count in enumerate(counts) if count})

def count_recursive(profile, columns, cutoff, output, work):
    n = len(columns)
    folder = output/f'n{n}_cutoff{cutoff}'
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'result.json'
    if result_path.exists():
        import json
        return {int(w): a for w, a in json.loads(result_path.read_text())['spectrum'].items()}
    if len(profile) <= 20:
        words = primal_catalogue(n, profile, columns, cutoff)
        counts = Counter(w for _, w in words)
        counts[0] = 1
    elif n <= 64:
        counts = Counter(measure(n, profile, columns, cutoff, work)['spectrum'])
    else:
        model = RelativePlotkinSpectrum(profile, columns)
        right = count_recursive(model.right_profile, model.right_columns, cutoff//2, output, work)
        counts = Counter({2*w: a for w, a in right.items()})
        # Causal polar cosets retain their profile row-weight lower bound even
        # when frozen expressions inside the left projection are nonzero.
        left_lower = min((1 << i.bit_count() for i in model.left_profile),default=n+1)
        if left_lower > cutoff:
            write(result_path,dict(N=n,K=len(profile),cutoff=cutoff,spectrum=dict(counts),exact=True,
                method='Exact repeated right subcode by the left causal-coset bound'))
            return dict(sorted(counts.items()))
        if len(model.left_profile) <= 20:
            left = primal_catalogue(n//2, model.left_profile, model.left_columns, cutoff)
        else:
            left, _ = complete_catalogue(n//2, model.left_profile, model.left_columns, cutoff, work)
        write(folder/'left_catalogue.json', dict(words=left, complete=True))
        done = 0
        details = folder/'cosets.jsonl'
        if details.exists():
            import json
            for line in details.read_text().splitlines():
                record = json.loads(line)
                counts.update({int(w): a for w, a in record['spectrum'].items()})
                done += 1
        with details.open('a', encoding='utf-8') as stream:
            last_progress = time.perf_counter()
            for index, (u, _) in enumerate(left[done:], done+1):
                item = inspect_fast(model, u, cutoff)
                counts.update(item['spectrum'])
                import json
                stream.write(json.dumps(item)+'\n')
                if time.perf_counter()-last_progress >= 10 or index == len(left):
                    stream.flush()
                    write(folder/'progress.json', dict(completed=index, total=len(left),
                                                      partial_spectrum=dict(counts), exact=False))
                    print(f'Exact relative spectrum N={n}: {index}/{len(left)}', flush=True)
                    last_progress = time.perf_counter()
    write(result_path, dict(N=n, K=len(profile), cutoff=cutoff, spectrum=dict(counts), exact=True,
                            method='Complete constituent catalogues and exact subset convolutions'))
    return dict(sorted(counts.items()))
