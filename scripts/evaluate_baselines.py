#!/usr/bin/env python
"""Evaluate baseline models on the 6 datasets.

Supports per-model overrides (dtype, device_map, trust_remote_code) and a
filter so you can run one model at a time, which is what you want at 8-13B
scale. The script reports the task metric AND generated-tokens-per-second
using the same code path that the dynamic-hybrid eval uses, so throughput
numbers are directly comparable.
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from chimera.data import load_eval_dataset  # noqa: E402
from chimera.eval import evaluate_examples, load_hf_model_and_tokenizer  # noqa: E402
from chimera.utils import append_jsonl, ensure_dir, read_yaml, set_seed  # noqa: E402


def _resolve(model_cfg, top, key, default=None):
    """Per-model overrides win over global config defaults."""
    if key in model_cfg and model_cfg[key] is not None:
        return model_cfg[key]
    if key in top and top[key] is not None:
        return top[key]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', default='results/baselines.jsonl')
    ap.add_argument('--max_samples', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_new_tokens', type=int, default=None,
                    help='Override generation budget. If omitted, each dataset uses its '
                         'recommended default (e.g. 4 for ARC, 128 for CNN/DM, 512 for GovReport, '
                         '64 for XSum). Set explicitly only if you need apples-to-apples '
                         'throughput across datasets.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', nargs='+', default=None,
                    help='Optional whitelist of model names to evaluate (matches "name" in the config).')
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='Optional dataset whitelist; overrides the config datasets list.')
    ap.add_argument('--free_after', action='store_true',
                    help='Drop the model and empty the CUDA cache between models. Recommended at 8B+.')
    ap.add_argument('--sample_seed', type=int, default=42,
                    help='Seed for representative sampling of --max_samples rows. '
                         'Set to 0 to disable sampling (take the first N rows = old behavior).')
    ap.add_argument('--stratify', default='length', choices=['length', 'random', 'none'],
                    help='Sampling strategy when sample_seed is set. "length" (default) '
                         'buckets by prompt length and draws evenly across buckets so the '
                         'subset matches the full distribution. Use "random" for plain uniform.')
    ap.add_argument('--input_max_tokens', type=int, default=4096,
                    help='Tokenizer truncation length for input prompts. Lower this to '
                         'speed up long-document evals (arXiv/GovReport prefill is dominated '
                         'by input length). 2048 typically halves wall-time for arXiv.')
    ap.add_argument('--split', default=None,
                    help='Override the per-config split. Accepts HF split arithmetic, e.g. '
                         '"validation[:500]" to only download the first 500 rows. Useful '
                         'when the dataset parquet itself is large (PubMed, arXiv) and you '
                         'only need a small representative subset.')
    args = ap.parse_args()
    set_seed(args.seed)

    cfg = read_yaml(args.config)
    ensure_dir(args.output.rsplit('/', 1)[0] if '/' in args.output else '.')

    datasets = args.datasets or cfg['datasets']
    if args.models:
        wanted = set(args.models)
        models = [m for m in cfg['models'] if m['name'] in wanted]
        missing = wanted - {m['name'] for m in models}
        if missing:
            raise SystemExit(f'Unknown model names in --models: {missing}')
    else:
        models = cfg['models']

    for model_cfg in models:
        name = model_cfg['name']
        path = model_cfg['path']
        torch_dtype = _resolve(model_cfg, cfg, 'torch_dtype', 'bfloat16')
        device_map = _resolve(model_cfg, cfg, 'device_map', 'auto')
        trust = _resolve(model_cfg, cfg, 'trust_remote_code', True)
        split = args.split or _resolve(model_cfg, cfg, 'split', 'validation')

        print(f'\n=== Loading {name} from {path} ===')
        print(f'    dtype={torch_dtype} device_map={device_map} trust_remote_code={trust}')
        model, tok = load_hf_model_and_tokenizer(
            path, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=trust,
        )

        sample_seed = args.sample_seed if args.sample_seed != 0 else None
        for ds_name in datasets:
            try:
                examples = load_eval_dataset(
                    ds_name, split=split, max_samples=args.max_samples,
                    sample_seed=sample_seed, stratify=args.stratify,
                )
            except Exception as e:
                print(f'  [{ds_name}] dataset load failed: {e}')
                append_jsonl(args.output, {'model': name, 'dataset': ds_name, 'error': repr(e)})
                continue
            try:
                res = evaluate_examples(
                    model, tok, examples,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    input_max_tokens=args.input_max_tokens,
                )
            except Exception as e:
                print(f'  [{ds_name}] eval failed: {e}')
                traceback.print_exc()
                append_jsonl(args.output, {'model': name, 'dataset': ds_name, 'error': repr(e)})
                continue
            row = {'model': name, **res}
            print('  ', row)
            append_jsonl(args.output, row)

        if args.free_after:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
