"""Exact initial-domain completion used by the optional length-256 replays."""
from types import SimpleNamespace

from polar_subcodes.code import Design, PolarSubcode
from polar_subcodes.construction.domains import initial_structure
from polar_subcodes.construction.source_patterns import SourcePatternProblem, PatternSearch


def complete_initial_domain(n, profile, target, ordering):
    """Release the allowed frozen rows and exclude their low-weight inputs."""
    from polar_subcodes.spectrum.backend import words

    domain = initial_structure(n, profile, target)
    released = domain["rows"]
    # Releasing these frozen checks leaves a common supercode containing
    # every assignment in the initial domain. Its low-weight inputs give the
    # complete directory before conditions are grouped by source pattern.
    mother_profile = tuple(sorted(set(profile) | set(released)))
    mother = PolarSubcode(n, mother_profile, (0,) * n)
    # These initial domains use only rows of weight at least d_P. Their RM
    # supercode has no weights strictly between d_P and 3 d_P / 2.
    cutoff = domain["baseline"]
    assert 2 * target <= 3 * cutoff
    directory, _ = words(mother, cutoff, max_work=500_000_000)
    catalogue = SimpleNamespace(design=Design(n, profile), rows=released,
                                columns=[0] * n, words=directory, domain=domain["coordinates"])
    problem = SourcePatternProblem(catalogue)
    search = PatternSearch(problem, "reverse", ordering)
    while search.status == "pending":
        search.step()
    if search.status != "sat":
        raise ValueError(f"Complete initial domain at length {n} is infeasible")
    columns = [0] * n
    for j, chosen_sources in zip(released, problem.deltas(search.solution)):
        columns[j] = chosen_sources
    return PolarSubcode(n, profile, tuple(columns))
