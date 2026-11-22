"""Revisit reliability-improving profiles through their shared suffix tasks.

Only a completed full-causal suffix contradiction excludes a parent profile.
Restricted failures and work limits remain unresolved. Reference coefficients
are never imported; the reference profile is an explicitly labelled control.
"""

from run_manuscript_n256_reduced_joint import verify
from run_rm_ga_profile_polish import canonical_check, read, reliability_cost

def checked_support(n, profile, columns, target, work):
    checked = verify(n, profile, columns, target, work, both=True)
    hist = checked["cpp"]["spectrum"]
    if any(w and w < target and a for w, a in hist.items()):
        raise AssertionError("A forbidden codeword survived independent enumeration")
    proof = canonical_check(n, profile, columns, target)
    proof.update(spectrum=hist, cpp_python_agree=True,
                 dmin=target if hist.get(target) else None)
    return dict(N=n, K=len(profile), profile=profile, columns=columns,
                target=target, verification=proof, initialization="zero",
                reference_support_read=False)
