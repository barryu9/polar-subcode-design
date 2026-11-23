"""Recompute Table II's information-set columns directly from the shipped profiles.

All eight rows use the same 2.25 dB comparison SNR. The proposed profile
determines d_P, and each profile in the pair determines its own P_GA(I).
The minimum distance and its multiplicity depend on the full code and are
obtained by the separate exact-spectrum calculations in Section VI.
"""
import csv
from functools import lru_cache
import math
from pathlib import Path

from polar_subcodes.code import load_profile
from .table_ii.python.polar_code.rp_funcs import rp_GA

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=6)
def error_probabilities(n, payload):
    """Return the GA bit-channel error estimates at the common comparison SNR."""
    return rp_GA(n, payload, 2.25).pe


def profile_statistic(code):
    # P_GA(I) = 1 - product_{i in I}(1 - p_i^GA) in Section VI.A.
    # It ranks the selected information channels. Changing only the dynamic
    # frozen expressions leaves this value unchanged. log1p/expm1 preserve
    # accuracy when a bit-channel error estimate is close to zero.
    errors = error_probabilities(code.n, code.k)
    return -math.expm1(sum(math.log1p(-float(errors[i])) for i in code.profile))


def main():
    with (ROOT / "results/table_ii.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case = row["case"]
            proposed = load_profile(ROOT / "configs/proposed" / (case + ".profile.json"))
            if proposed.n == 1024:
                # At length 1,024, Table II pairs the distance-24 rows with
                # Number-polar (13,1,4) and the distance-32 row with (15,6,1).
                suffix = "15_6_1" if case.endswith("_d32") else "13_1_4"
                ref_id = "n1024_k512_number_polar_" + suffix
            else:
                ref_id = case + ("_sc_ml" if proposed.n == 64 else "_recurscode")
            reference = load_profile(ROOT / "configs/reference" / (ref_id + ".profile.json"))
            # d_P is the minimum polar-row weight over the Proposed information
            # set. It is the minimum distance before adding frozen expressions.
            values = [f"{profile_statistic(code):.6f}" for code in (proposed,reference)]
            if values != [row["P_GA_proposed"], row["P_GA_reference"]] or proposed.d_P != int(row["d_P"]):
                raise ValueError(f"Profile statistics disagree with Table II for {case}")
            print(f"{case:18s} d_P={proposed.d_P:2d}  P_GA={values[0]}  reference={values[1]}")


if __name__ == "__main__":
    main()
