"""Construct information sets in natural polar-input order (Section VI.A).

For the GA-designed comparison codes the experiment used numerical integration
of phi. The separate Table II reliability *statistic* used Chung's fitted phi.
The design SNR is Eb/N0 per payload bit even when a CRC occupies extra positions.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

from scipy.integrate import quad
from scipy.optimize import brentq


def _log_phi(mean):
    """Log of 1-E[tanh(LLR/2)] for a Gaussian LLR of mean m, variance 2m."""
    if mean < 1e-5:
        return -mean / 2 + mean**2 / 8 - mean**3 / 8

    def sech(x):
        t = math.exp(-x)
        return 2 * t / (1 + t * t)

    if mean <= 1:
        area = quad(lambda t: math.exp(-t*t/4) * sech(math.sqrt(mean)*t/2),
                    0, 16, epsabs=1e-13, epsrel=1e-12)[0]
        return -mean/4 + math.log(area/math.sqrt(math.pi))
    area = quad(lambda t: sech(t/2) * math.exp(-t*t/(4*mean)),
                0, 80, epsabs=1e-13, epsrel=1e-12)[0]
    return -mean/4 + math.log(area/math.sqrt(math.pi*mean))


def _minus_mean(mean):
    # Combining two copies gives phi(m-) = 1 - (1 - phi(m))**2.
    lp = _log_phi(mean)
    target = lp + math.log(2 - math.exp(lp)) if lp < -10 else math.log1p(-math.expm1(lp)**2)
    if target > -1e-12:
        return max(0.0, -2*target)
    return brentq(lambda m: _log_phi(m) - target, 0, mean, xtol=1e-12, rtol=1e-12)


@lru_cache(maxsize=32)
def _ga_means(n, rate, ebn0_db):
    means = [4 * rate * 10**(ebn0_db / 10)]
    while len(means) < n:
        means = [child for mean in means for child in (_minus_mean(mean), 2 * mean)]
    return tuple(means)


def reliability_order(n, method, *, rate=0.5, ebn0_db=2.0, erasure=0.5):
    """Return positions from most to least reliable. Ties follow the experiment."""
    if type(n) is not int or n < 1 or n & (n-1):
        raise ValueError("N must be a positive power of two")
    if method in ("ga", "rm_ga"):
        if not 0 < rate <= 1 or not math.isfinite(ebn0_db) or not -20 <= ebn0_db <= 40:
            raise ValueError("Use a positive rate <= 1 and design Eb/N0 from -20 to 40 dB")
        means = _ga_means(n, rate, ebn0_db)
        # Row weight = 2**popcount(i). RM–GA uses GA only within a weight class.
        return sorted(range(n), key=lambda i: (i.bit_count() if method == "rm_ga" else 0,
                                                means[i], i), reverse=True)
    if method == "bec":
        if not 0 < erasure < 1:
            raise ValueError("Use an erasure probability strictly between 0 and 1")
        z = [erasure]
        while len(z) < n:
            z = [child for p in z for child in (p*(2-p), p*p)]
        beta = 2**0.25
        pw = [sum(beta**j for j in range(n.bit_length()-1) if i >> j & 1) for i in range(n)]
        return sorted(range(n), key=lambda i: (z[i], -pw[i], -i))
    if method == "5g":
        if not 32 <= n <= 1024:
            raise ValueError("Supported 5G mother lengths are 32 through 1024")
        table = json.loads(Path(__file__).with_name("nr_reliability.json").read_text(encoding="utf-8"))
        return [i for i in reversed(table["least_to_most_reliable"]) if i < n]
    raise ValueError("Expected ga, rm_ga, bec or 5g")


def make_profile(n, k, method, *, payload=None, ebn0_db=2.0, erasure=0.5):
    """Pick K positions (payload plus optional CRC) and return them in SC order."""
    if type(k) is not int or not 0 <= k <= n:
        raise ValueError("K must be between zero and N")
    payload = k if payload is None else payload
    if type(payload) is not int or not 0 <= payload <= k or (k and not payload):
        raise ValueError("Payload length must be positive for a nonempty profile")
    order = reliability_order(n, method, rate=(payload/n if n and payload else 0.5),
                              ebn0_db=ebn0_db, erasure=erasure)
    return tuple(sorted(order[:k]))
