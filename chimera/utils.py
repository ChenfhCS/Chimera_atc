import json, os, random, time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_jsonl(path: str, rows):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            if is_dataclass(r):
                r = asdict(r)
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def append_jsonl(path: str, row):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def count_new_tokens(outputs: torch.Tensor, input_len: int) -> int:
    return int(outputs.shape[-1] - input_len)


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
