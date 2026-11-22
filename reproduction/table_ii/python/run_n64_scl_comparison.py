"""Compare saved N=64 constructions using fixed-frame, paired SCL trials.

The roster includes recent designs, published references, and explicitly
identified historical learning baselines. It is not an exhaustive archive scan.
"""

from feasible_design_bank import code_fingerprint, generator_rows, validate_design
from select_designs_scl import construction, noiseless_check

def normalize_json(data):
    """Convert canonical and historical action/support formats without truncation."""
    code = data.get("code", data)
    actions = code.get("actions")
    profile = data.get("profile", data.get("information_positions"))
    if profile is None:
        if actions is None:
            raise ValueError("Missing profile and actions")
        profile = [i for i, action in enumerate(actions) if action == 0]
    columns = data.get("columns")
    if columns is None:
        columns = []
        for support in code["supports"]:
            if len(set(support)) != len(support):
                raise ValueError("Duplicate support index")
            columns.append(sum(1 << i for i in support))
    n, k = data.get("N", len(columns)), data.get("K", len(profile))
    if len(profile) != k:
        raise ValueError("Incorrect dimension")
    profile = list(validate_design(n, profile, columns))
    if actions is not None:
        expected_actions, expected_supports = construction(n, profile, columns)
        if list(actions) != list(expected_actions):
            raise ValueError("Legacy actions do not match canonical expressions")
        if "supports" in code and [list(s) for s in expected_supports] != code["supports"]:
            raise ValueError("Legacy supports do not match canonical expressions")
    identity = code_fingerprint(n, generator_rows(n, profile, columns))
    if data.get("design_id", identity) != identity:
        raise ValueError("Stored fingerprint does not match generator row space")
    return dict(N=n, K=k, profile=profile, columns=columns, design_id=identity)
