"""Check fresh design replays and export compact, portable release evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polar_subcodes.code import load_code
from polar_subcodes.experiments.reproduce import CASES
from polar_subcodes.io import hardware, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True,
                        help='Parent directory produced by main.py reproduce all')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for compact evidence')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    hardware=hardware(), packages={name: version(name) for name in
                    ('numpy', 'matplotlib', 'setuptools')}, cases={},
                    timing_scope='Fresh sequential reader-driver replays, not paper timing boundaries')
    for name, (target, _, counted) in CASES.items():
        folder = args.runs/name
        code = load_code(folder/'support.json')
        report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        replay = json.loads((folder/'reproduction.json').read_text(encoding='utf-8'))
        if code != load_code(ROOT/'results'/name/'support.json'):
            raise AssertionError(f'Support mismatch: {name}')
        if not (report['success'] and report['exact'] and report['dmin'] == target
                and report['coefficient_counted'] == counted
                and replay['support_matches_archive']
                and replay['initial_support'] == 'zero'
                and replay['archived_support_read_after_search']):
            raise AssertionError(f'Incomplete construction evidence: {name}')
        destination = args.output/name
        destination.mkdir()
        for filename in ('report.json', 'reproduction.json', 'timing.json',
                         'run.log', 'events.jsonl'):
            source = folder/filename
            if source.exists():
                shutil.copy2(source, destination/filename)
        (destination/'equations.txt').write_text(code.describe()+'\n', encoding='utf-8')
        manifest['cases'][name] = dict(d_P=code.d_P, dmin=report['dmin'],
            Admin=report['Admin'], total_seconds=report['total_seconds'],
            support_matches_archive=True,
            support_sha256=hashlib.sha256((folder/'support.json').read_bytes()).hexdigest())
        print(f'Checked {name}: dmin={target}, Admin={report["Admin"]}', flush=True)

    transcript = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with redirect_stdout(transcript), redirect_stderr(transcript):
        result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
    (args.output/'tests.log').write_text(transcript.getvalue(), encoding='utf-8')
    manifest['tests'] = dict(run=result.testsRun, failures=len(result.failures),
                            errors=len(result.errors), skipped=len(result.skipped),
                            success=result.wasSuccessful())
    sources = [ROOT/'main.py', ROOT/'CMakeLists.txt', ROOT/'requirements.txt']
    for directory in ('polar_subcodes', 'cpp', 'tests', 'tools', 'examples', 'vendor'):
        sources.extend(path for path in (ROOT/directory).rglob('*')
                       if path.suffix in ('.py', '.cpp', '.h', '.hpp'))
    manifest['source_sha256'] = {path.relative_to(ROOT).as_posix():
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(sources))}
    write_json(args.output/'manifest.json', manifest)
    print(f'Tests: {manifest["tests"]}. Evidence: {args.output}', flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
