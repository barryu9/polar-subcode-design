"""Utilities for independent, case-specific cold reproductions.

No historical design is an input. Each entry point defines its own exploration
path. Acceptance identities belong exclusively to the separate verifier.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, data):
    from run_diverse_profile_suite import write as atomic_write
    atomic_write(Path(path), data)

def within(path, root):
    return path == root or root in path.parents

def install_guard(run_root):
    """Deny historical data reads in the driver and all Python children."""
    run_root = Path(run_root).resolve()
    repository = ROOT.parents[1]
    # Published comparator JSONs and prior runs are independent acceptance evidence.
    # All support vectors read by a design stage must be from its own workspace.
    forbidden = [repository / name for name in ("configs", "results", "profile", "runs")]
    forbidden.extend(ROOT / name for name in ("results", "reference", "research", "logs", "runs", "output"))
    original_popen = subprocess.Popen

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not within(path, run_root) and any(within(path, p) for p in forbidden):
            raise PermissionError(f"Historical data access denied: {path}")

    class GuardedPopen(original_popen):
        def __init__(self, command, *args, **kwargs):
            if isinstance(command, (tuple, list)) and command:
                command = list(map(str, command))
                executable = Path(command[0])
                # A virtual environment or Conda may return python3.12 (or an
                # explicitly named interpreter) as sys.executable on Unix.
                same_interpreter = executable.is_absolute() and executable.resolve() == Path(sys.executable).resolve()
                python_name = re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", executable.name, re.I)
                if same_interpreter or python_name:
                    if "-c" in command or "-m" in command:
                        raise PermissionError("Cold-design Python children must use an audited script entry")
                    position = next((i for i, arg in enumerate(command[1:], 1)
                                     if arg.endswith(".py")), None)
                    if position is not None and Path(command[position]).name != "cold_reproduction_worker.py":
                        command = command[:position] + [str(ROOT / "python/cold_reproduction_worker.py"),
                            "--run-root", str(run_root), "--script", command[position], "--"] + command[position + 1:]
            super().__init__(command, *args, **kwargs)

    sys.addaudithook(audit)
    subprocess.Popen = GuardedPopen

class Experiment:
    def __init__(self, n, k, target, description, *, configure_parser=None,
                 prepare_inputs=None):
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--output", type=Path)
        parser.add_argument("--smoke", action="store_true",
                            help="Generate inputs only; this does not establish reproduction")
        parser.add_argument("--defer-acceptance", action="store_true",
                            help="Exit after freezing the archive for external acceptance")
        if configure_parser is not None:
            configure_parser(parser)
        args = parser.parse_args()
        self.root = (args.output or ROOT / "results" /
                     f"independent_n{n}_k{k}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
        if self.root.exists():
            raise FileExistsError("Use an empty output directory for a cold reproduction")
        self.root.mkdir(parents=True)
        self.n, self.k, self.target, self.smoke = n, k, target, args.smoke
        self.defer_acceptance = args.defer_acceptance
        self.started = time.perf_counter()
        self.stages = []
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "1"
        inputs = {} if prepare_inputs is None else prepare_inputs(self, args)
        install_guard(self.root)
        protocol = dict(N=n, K=k, target=target,
            initial_candidate_count=0, historical_profile_input=False,
            historical_support_input=False, historical_cache_input=False,
            history_read_guard=True, acceptance_identity_in_design=False,
            scalar_parameters_historically_informed=True,
            entry_point=str(Path(sys.argv[0]).resolve()),
            entry_sha256=hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest(),
            python=sys.version, platform=platform.platform(), cpu=platform.processor(),
            timing_policy="Report total wall time and separately subtract whole timed-out/resource-limited attempts")
        protocol.update(inputs)
        write(self.root / "protocol.json", protocol)
        self.emit("start", output=str(self.root))

    def emit(self, event, **fields):
        row = dict(event=event, elapsed=time.perf_counter() - self.started, **fields)
        print(json.dumps(row), flush=True)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        write(self.root / "progress.json", row)

    @contextlib.contextmanager
    def stage(self, name):
        self.emit("stage_start", stage=name)
        tick = time.perf_counter()
        record = dict(stage=name, excluded_from_effective_time=False)
        try:
            yield record
            record["status"] = "completed"
        except BaseException as error:
            record.update(status="failed", reason=str(error))
            raise
        finally:
            record["seconds"] = time.perf_counter() - tick
            self.stages.append(record)
            write(self.root / "timing.json", self.stages)
            self.emit("stage_end", **record)

    def call(self, name, script, arguments):
        with self.stage(name) as record:
            folder = self.root / "driver_logs"
            folder.mkdir(exist_ok=True)
            command = [sys.executable, "-u", str(ROOT / "python" / script)] + list(map(str, arguments))
            record["command"] = command
            with (folder / f"{len(self.stages):03d}_{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if len(line) < 2500 and not line.lstrip().startswith("{'iteration':"):
                        print(line.rstrip(), flush=True)
                status = process.wait()
            if status:
                raise RuntimeError(f"{script} failed with exit code {status}; inspect {folder}")

    def ga(self, snr=2.25):
        with self.stage(f"channel_input_{snr}"):
            from run_manuscript_n64_suite import channel_data
            from polar_code.rp_funcs import rp_GA
            data = channel_data(self.n, self.k, snr)
            costs = [-math.log1p(-float(p)) for p in rp_GA(self.n, self.k, snr).pe]
            data.update(costs=costs, GA_design_EbN0_dB=snr)
            path = self.root / f"ga_{snr}.json"
            write(path, data)
        return path

    def bank(self, case, *, profiles, layered=True, anchor=False, depth=1,
             seeds=5000, gap=False, joint=0, focus=2, full_necessary=False):
        folder = self.root / "profile_exploration"
        self.call("profile_and_support_exploration", "run_diverse_profile_suite.py", [
            "--output", folder, "--cases", case, "--case-seconds", 14400,
            "--trials", 256, "--trial-seconds", 1800, "--support-seconds", 120,
            "--outer-seconds", 240, "--work", 30000000, "--verification-work", 600000000,
            "--solver-nodes", 200000, "--joint-coefficients", joint,
            "--class-depth", depth, "--layered-seed-limit", seeds,
            "--focus-batch", focus, "--stop-after", profiles, "--snr", 2.25,
            "--cold-run-root", self.root, "--retry-incomplete"] +
            (["--layered-classes"] if layered else []) +
            (["--anchor"] if anchor else []) + (["--gap-first"] if gap else []) +
            (["--full-necessary"] if full_necessary else []))
        case_folder = folder / f"n{self.n}_k{self.k}_dp{dict(zip(range(6),[8,8,32,16,8,16]))[case]}_d{self.target}"
        archive = read(case_folder / "archive.json")
        return [case_folder / "designs" / row["name"] / "code.json" for row in archive]

    def ordered(self, source, name, directions, trials):
        output = self.root / name
        self.call(name, "refine_protected_core_spectrum.py", ["--source", source,
            "--output", output, "--target", self.target, "--directions", directions,
            "--trials", trials, "--seconds", 14400, "--work", 3000000,
            "--unknown-streak", 16])
        return output / "code.json"

    def construct(self, profile, name, *, joint=256):
        # This is the fixed-profile step: keep the information positions and
        # seek causal coefficients meeting d_star. Return a candidate only
        # after distance verification; the caller may try the next profile.
        folder = self.root / name
        source = self.root / "proposed" / f"{name}.json"
        write(source, dict(N=self.n, K=self.k, target=self.target, profile=list(profile)))
        self.call(name, "run_diverse_profile_suite.py", ["--worker", "--profile", source,
            "--output", folder, "--cache", self.root / "construction_cache",
            "--mode", "structured", "--trial-seconds", 1800, "--support-seconds", 120,
            "--outer-seconds", 240, "--solver-nodes", 200000,
            "--joint-coefficients", joint, "--work", 30000000,
            "--verification-work", 600000000, "--cold-run-root", self.root])
        result = read(folder / "result.json")
        if result["status"] != "verified":
            self.emit("construction_unfinished", proposal=name, status=result["status"])
            return None
        path = folder / "support.json"
        write(path, dict(N=self.n, K=self.k, target=self.target, profile=list(profile),
                         columns=result["columns"], verification=result["verification"]))
        return path

    def feedback(self, source, name, length, seed, frames, snr, trials=96,
                 *, catalogue_variants=True):
        with self.stage(name):
            from cold_case_feedback import refine
            return refine(self, source, self.root / name, length, seed, frames, snr, trials,
                          catalogue_variants=catalogue_variants)

    def freeze(self, sources):
        """Collect distinct designs after search, before matching a plotted code.

        The profile identifies information positions. Two codes with that
        profile can have different dynamic frozen expressions, so each entry
        also records the digest of its coefficient columns.
        """
        from run_diverse_profile_suite import identities
        archive = {}
        for path in sources:
            if not within(Path(path).resolve(), self.root):
                raise ValueError("Only same-run sources can enter the frozen archive")
            raw = read(path)
            if "columns" not in raw:
                continue
            ph, sh = identities(raw["N"], raw["profile"], raw["columns"])
            name = ph[:12] + "_" + sh[:12]
            destination = self.root / "designs" / name / "code.json"
            write(destination, dict(raw, name=name, profile_hash=ph, support_hash=sh))
            archive[name] = dict(name=name, profile_hash=ph, support_hash=sh,
                                 code_path=str(destination), source=str(Path(path).resolve()))
        exclusions = []
        for path in self.root.rglob("trials.json"):
            for trial in read(path):
                if trial.get("status") in ("wall_limit", "enumeration_limit", "representation_limit"):
                    exclusions.append(dict(record=str(path), seconds=trial.get("trial_seconds", 0),
                                           status=trial["status"]))
        total = time.perf_counter() - self.started
        excluded = sum(row["seconds"] for row in exclusions)
        write(self.root / "frozen_archive.json", list(archive.values()))
        write(self.root / "summary.json", dict(N=self.n, K=self.k, target=self.target,
            archive_size=len(archive), stages=self.stages, total_wall_seconds=total,
            excluded_attempts=exclusions, excluded_attempt_seconds=excluded,
            effective_seconds=max(0, total - excluded), frozen_before_identity_check=True,
            exact_historical_identity_not_yet_checked=True,
            timing_caveat="Effective time is not ordinary wall time; all completed unsuccessful trials remain counted"))
        self.emit("archive_frozen", codes=len(archive), total_wall_seconds=total,
                  effective_seconds=max(0, total - excluded))
        if not self.defer_acceptance:
            self.call("post_freeze_acceptance", "verify_independent_reproduction.py", ["--run", self.root])
