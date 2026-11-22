"""Exact recursive Plotkin spectrum with radius-four pair-syndrome counting."""

from collections import Counter
import json
import time
from audit_profile_distance_ceiling import dual_macwilliams_spectrum
from obstruction_support_search import enumerate_words
from plotkin_relative_structure import RelativePlotkinSpectrum
from run_manuscript_n256_reduced_joint import verify
from search_checkpoint_io import atomic_json

def count_code(profile, columns, cutoff, output, work):
    n = len(columns)
    path = output / f"n{n}_cutoff{cutoff}"
    path.mkdir(exist_ok=True)
    started = time.perf_counter()
    if n-len(profile) <= 16:
        dual = dual_macwilliams_spectrum(profile, columns, max_dual_dimension=16)
        counts = {w: count for w, count in dual["spectrum"].items() if w <= cutoff}
        direct = verify(n, profile, columns, cutoff, work)
        if counts != direct["cpp"]["spectrum"]:
            raise AssertionError("Dual and direct terminal spectra disagree")
        atomic_json(path / "result.json", dict(
            exact=True, n=n, k=len(profile), cutoff=cutoff, spectrum=counts,
            method="complete dual enumeration with integer MacWilliams transform",
            dual_spectrum=dual["dual_spectrum"], direct=direct["cpp"],
            seconds=time.perf_counter()-started,
        ))
        print(f"BASE N={n}, K={len(profile)}, spectrum={counts}", flush=True)
        return counts
    model = RelativePlotkinSpectrum(profile, columns)
    right = count_code(model.right_profile, model.right_columns, cutoff//2, output, work)
    left_verified = verify(n//2, model.left_profile, model.left_columns, cutoff, work)
    left_words, catalogue = enumerate_words(
        n//2, model.left_profile, model.left_columns, cutoff, work
    )
    left_counts = Counter(w for _, w in left_words)
    left_counts[0] = 1
    if dict(left_counts) != left_verified["cpp"]["spectrum"]:
        raise AssertionError("Left catalogue and independent histogram disagree")
    atomic_json(path / "left_verified.json", left_verified)
    atomic_json(path / "left_catalogue.json", dict(words=left_words, audit=catalogue))
    print(f"LEFT N={n//2}, K={len(model.left_profile)}, spectrum={dict(left_counts)}",
          flush=True)
    counts = Counter({2*w: value for w, value in right.items() if 2*w <= cutoff})
    cross = Counter()
    per_weight = {}
    last_print = time.perf_counter()
    details = (path / "cosets.jsonl").open("w", encoding="utf-8")
    try:
        for index, (u, weight) in enumerate(left_words, 1):
            result = model.inspect(u, cutoff)
            if result["left_weight"] != weight:
                raise AssertionError("Catalogue and polar transform disagree")
            counts.update(result["spectrum"])
            cross.update(result["spectrum"])
            per_weight.setdefault(weight, Counter()).update(result["spectrum"])
            details.write(json.dumps(result) + "\n")
            now = time.perf_counter()
            if now-last_print >= 10 or index == len(left_words):
                details.flush()
                atomic_json(path / "progress.json", dict(
                    exact=index == len(left_words), completed=index, total=len(left_words),
                    partial_spectrum=dict(sorted(counts.items())),
                    seconds=now-started,
                ))
                print(f"N={n}: {index}/{len(left_words)} left cosets; "
                      f"partial spectrum={dict(sorted(counts.items()))}", flush=True)
                last_print = now
    finally:
        details.close()
    counts = dict(sorted(counts.items()))
    atomic_json(path / "result.json", dict(
        exact=True, n=n, k=len(profile), cutoff=cutoff, spectrum=counts,
        repeated_right_spectrum={2*w: value for w, value in right.items() if 2*w <= cutoff},
        cross_spectrum=dict(sorted(cross.items())),
        cross_by_left_weight={w: dict(sorted(hist.items())) for w, hist in per_weight.items()},
        left_catalogue_spectrum=dict(sorted(left_counts.items())),
        completed_left_cosets=len(left_words), seconds=time.perf_counter()-started,
    ))
    return counts
