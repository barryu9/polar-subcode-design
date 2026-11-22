"""Replay the Table II Proposed-LM variant from the generated main archive."""

from reproduce_n1024_variants import initialize, low_multiplicity

if __name__ == "__main__":
    run = initialize(24, __doc__)
    if not run.smoke:
        run.freeze([low_multiplicity(run)])
