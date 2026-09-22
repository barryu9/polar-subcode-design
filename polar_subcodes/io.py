"""Portable output, progress logs, and hardware descriptions."""
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True)+'\n', encoding='utf-8')
    temporary.replace(path)


def hardware():
    return dict(cpu=platform.processor(), logical_processors=os.cpu_count(),
                platform=platform.platform(), python=platform.python_version(),
                timing='perf_counter wall time, includes checks and output I/O', gpu_used=False)


class Progress:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.start = time.perf_counter()

    def __call__(self, stage, message, **fields):
        record = dict(elapsed_seconds=time.perf_counter()-self.start,
                      stage=stage, message=message, **fields)
        line = f'[{record["elapsed_seconds"]:8.2f}s] {stage}: {message}'
        if fields:
            line += ' | '+', '.join(f'{key}={value}' for key, value in fields.items())
        print(line, flush=True)
        with (self.folder/'run.log').open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
        with (self.folder/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
