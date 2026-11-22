"""Exclusive verification timing inside a recursive construction worker."""

import functools
import importlib
import time
from run_diverse_profile_suite import write

class VerificationTiming:
    def __init__(self, path):
        self.path, self.seconds, self.calls, self.depth = path, 0., 0, 0

    def wrap(self, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            outer = self.depth == 0
            self.depth += 1
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.depth -= 1
                if outer:
                    self.seconds += time.perf_counter()-start
                    self.calls += 1
                    write(self.path, dict(verification_seconds=self.seconds,
                                         verification_calls=self.calls))
        return measured

    def finish(self, wall):
        result = dict(worker_wall_seconds=wall, verification_seconds=self.seconds,
                      construction_and_worker_overhead_seconds=max(0., wall-self.seconds),
                      verification_calls=self.calls, exclusive=True,
                      scope='Independent verification calls, not solver-internal enumeration')
        write(self.path, result)
        return result

def install_verification_timing(path):
    timing = VerificationTiming(path)
    for module_name, names in (
        ('structure_feedback_profile_design', ('final_check',)),
        ('run_rm_ga_profile_polish', ('verify', 'final')),
        ('run_profile_only_targets', ('verify', 'final')),
        ('run_manuscript_suffix_construction', ('verify',)),
        ('audit_suffix_construction', ('audit',)),
    ):
        module = importlib.import_module(module_name)
        for name in names:
            if hasattr(module, name):
                setattr(module, name, timing.wrap(getattr(module, name)))
    return timing
