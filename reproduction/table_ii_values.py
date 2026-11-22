"""Recompute Table II's information-set columns directly from the shipped profiles.

The comparison is at a common 2.25 dB, independent of the SNR originally
used to select a particular profile. The eight printed multiplicities and
minimum distances have their own exact-spectrum evidence and cold workflows.
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
    return rp_GA(n, payload, 2.25).pe


def profile_statistic(code):
    # Eq. (64) in Section VI.A: log1p/expm1 retain small probabilities.
    errors = error_probabilities(code.n, code.k)
    return -math.expm1(sum(math.log1p(-float(errors[i])) for i in code.profile))


def main():
    with (ROOT / "results/table_ii.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case = row["case"]
            proposed = load_profile(ROOT / "configs/proposed" / (case + ".profile.json"))
            if proposed.n == 1024:
                suffix = "15_6_1" if case.endswith("_d32") else "13_1_4"
                ref_id = "n1024_k512_number_polar_" + suffix
            else:
                ref_id = case + ("_sc_ml" if proposed.n == 64 else "_recurscode")
            reference = load_profile(ROOT / "configs/reference" / (ref_id + ".profile.json"))
            values = [f"{profile_statistic(code):.6f}" for code in (proposed,reference)]
            if values != [row["P_GA_proposed"], row["P_GA_reference"]] or proposed.d_P != int(row["d_P"]):
                raise ValueError(f"Profile statistics disagree with Table II for {case}")
            print(f"{case:18s} d_P={proposed.d_P:2d}  P_GA={values[0]}  reference={values[1]}")


if __name__ == "__main__":
    main()
