"""Atomic experiment output with bounded retries for transient Windows locks."""

import json
import time
from pathlib import Path

def replace_retry(source, target):
    for attempt in range(10):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt == 9: raise
            time.sleep(.05 * (attempt+1))

def atomic_json(path, value):
    path=Path(path); temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    replace_retry(temp,path)
