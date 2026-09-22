"""Mother GA profiles and explicit output-coordinate selection matrices."""
from copy import deepcopy
from ._fixtures import INFORMATION, PATTERNS


def get_profile(n=64, kind="polar"):
    if n not in INFORMATION or kind not in ("polar", "pac"):
        raise ValueError("Choose N in {64,128,256} and kind polar or pac.")
    return dict(n=n, k=n//2, kind=kind, ebn0_db=2.0, information=tuple(INFORMATION[n]),
                taps=(1,) if kind == "polar" else (1, 0, 1, 1, 0, 1, 1))


def checks_for_profile(profile):
    n, taps = profile["n"], profile["taps"]
    impulse = [1] + [0] * (n - 1)
    for i in range(1, n):
        impulse[i] = sum(taps[j] * impulse[i-j] for j in range(1, min(i+1, len(taps)))) % 2
    information = set(profile["information"])
    return tuple(sum(impulse[i-j] << j for j in range(i+1)) for i in range(n) if i not in information)


def get_rate_matching(n=64, mode="none"):
    if n not in PATTERNS or mode not in PATTERNS[n]:
        raise ValueError("Choose N in {64,128,256}, mode none/puncture/shorten/mixed.")
    return deepcopy(PATTERNS[n][mode])


def selection_matrices(n=64, mode="none"):
    """Return E_T,E_P,E_S as explicit row-major lists of binary rows."""
    cfg = get_rate_matching(n, mode)
    p, s = set(cfg["punctured"]), set(cfg["shortened"])
    rows = lambda indices: [[int(j == i) for j in range(n)] for i in indices]
    return rows(i for i in range(n) if i not in p | s), rows(sorted(p)), rows(sorted(s))
