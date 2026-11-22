"""Replay the Table II Proposed-D32 variant from the generated main archive."""

from reproduce_n1024_variants import distance_32, initialize

if __name__ == "__main__":
    run = initialize(32, __doc__)
    if not run.smoke:
        run.freeze([distance_32(run)])
