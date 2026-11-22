"""Example 9. Follow two profile exchanges and solve the parent exclusion tests.

The default run solves both 32-bit domains and rebuilds every parent forbidden
set. --verify-spectrum also calls the exact native counter on the saved longer
suffixes and the completed N=256 code.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import json
from math import comb
from pathlib import Path
from types import SimpleNamespace

from polar_subcodes.algebra import polar_transform
from polar_subcodes.code import Design, PolarSubcode, load_code
from polar_subcodes.construction.domains import initial_structure
from polar_subcodes.construction.source_patterns import PatternSearch, SourcePatternProblem
from polar_subcodes.spectrum.relative import RelativePlotkinSpectrum

DATA = Path(__file__).with_name("data")


def solve_short_domain(profile):
    # Only even codewords are possible. Enumerating physical weights 2 and 4
    # is small at length 32; inverse polar transformation recovers their inputs.
    # The transform is its own inverse, so a unit output at i has input row g_i.
    structure = initial_structure(32, profile, 6)
    free = sum(1 << i for i in (*profile, *structure["rows"]))
    inverse_rows = [polar_transform(1 << i, 32) for i in range(32)]
    low = []
    for weight in (2, 4):
        for positions in combinations(range(32), weight):
            u = 0
            for j in positions:
                u ^= inverse_rows[j]
            if u & ~free == 0:
                low.append((u, weight))
    catalogue = SimpleNamespace(design=Design(32, tuple(profile)), rows=structure["rows"],
                                domain=structure["coordinates"], columns=[0] * 32, words=low)
    problem = SourcePatternProblem(catalogue)
    search = PatternSearch(problem)
    while search.status == "pending":
        search.step()
    return {"variables": len(catalogue.domain), "status": search.status,
            "nonzero_patterns": sum(s != 0 for s in problem.scopes)}


def run(verify_spectrum=False, rebuild=False):
    final = load_code(DATA / "example09_code.json")
    saved = json.loads((DATA / "example09_trace.json").read_text(encoding="utf-8"))
    # The quick walkthrough reads the historical GA ordering. The full
    # replay computes its own RM–GA order at rate 1/4 and Eb/N0 = 2 dB.
    if rebuild:
        from polar_subcodes.construction.profiles import reliability_order
        priority = reliability_order(256, "rm_ga", rate=0.25, ebn0_db=2)
    else:
        rank = {i: k for k, i in enumerate(saved["ga_order"])}
        priority = sorted(range(256), key=lambda i: (-i.bit_count(), rank[i]))
    eligible = [i for i in priority if 1 << i.bit_count() >= 32]
    # The length-16 sphere bound packs balls of size 17. At length 32
    # triple-error syndromes represent at most floor(32/3) disjoint triples.
    triples_per_syndrome = 32 // 3
    occupied_syndromes = 32 + (comb(32, 3) + triples_per_syndrome - 1) // triples_per_syndrome
    caps = {8: 7, 16: (2**16 // 17).bit_length() - 1,
            32: 31 - (occupied_syndromes - 1).bit_length()}
    assert caps == {8: 7, 16: 11, 32: 21}
    seed, skipped = [], []
    for i in eligible:
        if any(sum(j >= 256 - n for j in [*seed, i]) > k for n, k in caps.items()):
            skipped.append(i)
        else:
            seed.append(i)
        if len(seed) == 64:
            break
    assert skipped == [248, 244, 242, 241, 227]
    assert [i for i in seed if i not in eligible[:64]] == [179, 121, 173, 199, 118]
    first = (set(seed) - {233}) | {227}
    second = (first - {211}) | {158}
    profile = tuple(sorted(second))
    assert profile == final.profile
    short = []
    for p in (seed, first):
        short.append(solve_short_domain(sorted(i - 224 for i in p if i >= 224)))
    assert [r["status"] for r in short] == ["unsat", "sat"]
    assert [r["variables"] for r in short] == [20, 23]
    assert short[0]["nonzero_patterns"] == 19
    assert [sum(i >= 192 for i in p) for p in (first, second)] == [37, 36]
    assert final.columns[224 + 9] >> 224 == sum(1 << i for i in (3, 5, 6))

    # Remove only cross-half terms to expose the fixed constituent pair.
    if rebuild:
        from paper_examples.initial_domain import complete_initial_domain
        # Solve the larger suffix domains from their own zero baselines.
        # The verified 32-bit profile guides the exchange. All internal
        # coefficients remain available when solving each larger suffix.
        for n, target in ((64, 12), (128, 24)):
            start = 256 - n
            local = tuple(sorted(i - start for i in second if i >= start))
            suffix = complete_initial_domain(n, local, target, "product")
            assert suffix.columns == tuple(c >> start for c in final.columns[start:])
        fixed = (0,) * 128 + tuple(c << 128 for c in suffix.columns)
    else:
        fixed = tuple(c if j < 128 else (c >> 128) << 128 for j, c in enumerate(final.columns))
    structure = initial_structure(256, profile, 48)
    coordinates = [(structure["rows"][q], i) for q, i in structure["coordinates"]
                   if i < 128 <= structure["rows"][q]]
    sources = sorted({i for _, i in coordinates})
    destinations = sorted({j for j, _ in coordinates})
    assert sources == [118, 121, 122, 124] and len(coordinates) == 4 * 12
    model = RelativePlotkinSpectrum(profile, fixed)
    left = PolarSubcode(128, model.left_profile, model.left_columns)
    frozen = [i for i in range(128) if i not in model.right_profile]
    packed_destinations = [frozen.index(j - 128) for j in destinations]
    forbidden = defaultdict(set)
    critical = 0
    for message in range(1, 1 << left.k):
        if left.encode(message).bit_count() != 32:
            continue
        critical += 1
        u = left.input_word(message)
        scope = sum(((u >> i) & 1) << k for k, i in enumerate(sources))
        # For weight 32 there are no outside errors. A 12-bit increment is
        # forbidden precisely when its Q_a image is zero for this message.
        event = model.constraint(u, 32)
        labels = [0]
        for j in packed_destinations:
            col = event["syndrome_map"][j]
            labels += [v ^ col for v in labels]
        forbidden[scope].update(v for v, y in enumerate(labels) if y in event["forbidden"])
    assert critical == 44 and len(forbidden) == 11 and 0 not in forbidden
    assert {len(bad) for bad in forbidden.values()} == {32}
    mean = sum(Fraction(len(bad), 1 << len(destinations)) for bad in forbidden.values())
    assert mean == Fraction(11, 128)
    if rebuild:
        # Match the recorded deterministic source-pattern ordering. Each
        # variable assigns a 12-bit column. The 11 forbidden sets above
        # are sufficient to construct the parent without reading its support.
        catalogue = SimpleNamespace(design=Design(256, tuple(sorted(second))),
                                    rows=destinations, columns=fixed,
                                    domain=[(k, i) for k in range(len(destinations)) for i in sources])
        group_counts = {}
        for scope, bad in forbidden.items():
            # SourcePatternProblem packs sources in decreasing physical order.
            reverse_scope = sum(((scope >> k) & 1) << (len(sources) - 1 - k)
                                for k in range(len(sources)))
            group_counts.update({(reverse_scope, value): 1 for value in bad})
        problem = SourcePatternProblem(catalogue, projected_groups=group_counts, max_width=12)
        search = PatternSearch(problem, "reverse", "product")
        while search.status == "pending":
            search.step()
        assert search.status == "sat"
        constructed = list(fixed)
        for j, update in zip(destinations, problem.deltas(search.solution)):
            constructed[j] ^= update
        assert tuple(constructed) == final.columns
    for scope, bad in forbidden.items():
        u = sum(1 << i for k, i in enumerate(sources) if scope >> k & 1)
        value = sum(((final.columns[j] & u).bit_count() % 2) << k for k, j in enumerate(destinations))
        assert value not in bad
    # The verified right suffix supplies the zero-left condition; RM(3,8)
    # has no weights strictly between its minimum 32 and its next weight 48.
    assert all(i.bit_count() >= 5 for i in final.profile)
    assert all(j.bit_count() >= 5 for j, c in enumerate(final.columns) if c)
    source = final.profile.index(121)
    assert final.input_word(1 << source) == (1 << 121) | (1 << 155)
    assert final.encode(1 << source).bit_count() == 48
    suffixes = []
    for item in saved["suffix_certificates"]:
        n, target = item["N"], item["target"]
        start = final.n - n
        profile = tuple(i - start for i in final.profile if i >= start)
        coefficients = tuple(c >> start for c in final.columns[start:])
        assert len(profile) == item["K"]
        domain = initial_structure(n, profile, target)
        assert len(domain["coordinates"]) == item["variables"]
        if verify_spectrum:
            from polar_subcodes.spectrum.backend import count
            # Each suffix lies in an RM code of minimum 2*target/3, where the
            # next possible weight is target. Check membership and the first layer.
            minimum = 2 * target // 3
            active = set(profile) | {j for j, c in enumerate(coefficients) if c}
            assert all(1 << j.bit_count() >= minimum for j in active)
            assert count(PolarSubcode(n, profile, coefficients), minimum, 500_000_000)["spectrum"] == {0: 1}
        suffixes.append([n, len(profile), target, item["variables"]])
    if verify_spectrum:
        assert count(final, 32, 500_000_000)["spectrum"] == {0: 1}
    return {"skipped": skipped, "exchanges": [(233, 227), (211, 158)], "length_32_trials": short,
            "suffixes_(N,K,target,variables)": suffixes,
            "parent_sources": sources, "editable_rows": destinations, "critical_messages": critical,
            "forbidden_counts_by_pattern": {s: len(bad) for s, bad in sorted(forbidden.items())},
            "initial_mean": str(mean), "witness_input_support": [121, 155], "witness_weight": 48,
            "longer_suffix_verification": "recomputed" if verify_spectrum else "archived exact checks"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construct", action="store_true", help="Rebuild the two larger suffixes and parent")
    parser.add_argument("--verify-spectrum", action="store_true")
    args = parser.parse_args()
    print(run(args.verify_spectrum, args.construct))
