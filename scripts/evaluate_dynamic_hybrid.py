#!/usr/bin/env python
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from chimera.dynamic_hybrid_model import DynamicHybridConfig, DynamicHybridForCausalLM, load_tokenizer_for_hybrid
from chimera.data import load_eval_dataset
from chimera.eval import evaluate_examples
from chimera.utils import append_jsonl, ensure_dir, read_yaml, set_seed


def load_dynamic(path_or_config):
    if os.path.isdir(path_or_config):
        with open(os.path.join(path_or_config, 'config.json')) as f:
            cfg = DynamicHybridConfig(**json.load(f))
        model = DynamicHybridForCausalLM.from_pretrained_branches(cfg, device_map=None)
        ckpt = torch.load(os.path.join(path_or_config, 'dynamic_hybrid.pt'), map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        tok = load_tokenizer_for_hybrid(cfg.attn_model_name_or_path, cfg.trust_remote_code)
        return model, tok
    cfg = DynamicHybridConfig(**read_yaml(path_or_config))
    model = DynamicHybridForCausalLM.from_pretrained_branches(cfg, device_map=None)
    tok = load_tokenizer_for_hybrid(cfg.attn_model_name_or_path, cfg.trust_remote_code)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_or_config', required=True)
    ap.add_argument('--datasets', nargs='+', default=['arc-e','arc-c','piqa','truthfulqa','cnn/dm','govreport'])
    ap.add_argument('--output', default='results/dynamic_hybrid.jsonl')
    ap.add_argument('--max_samples', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_new_tokens', type=int, default=None,
                    help='Override generation budget; defaults to a per-dataset value when omitted.')
    ap.add_argument('--sample_seed', type=int, default=42,
                    help='Seed for representative sampling. Set to 0 to take head-N rows.')
    ap.add_argument('--stratify', default='length', choices=['length', 'random', 'none'])
    ap.add_argument('--input_max_tokens', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    model, tok = load_dynamic(args.model_or_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    ensure_dir(args.output.rsplit('/',1)[0] if '/' in args.output else '.')
    sample_seed = args.sample_seed if args.sample_seed != 0 else None
    for ds_name in args.datasets:
        split = 'validation'
        examples = load_eval_dataset(
            ds_name, split=split, max_samples=args.max_samples,
            sample_seed=sample_seed, stratify=args.stratify,
        )
        res = evaluate_examples(
            model, tok, examples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            is_dynamic=True,
            input_max_tokens=args.input_max_tokens,
        )
        row = {'model': 'Dynamic Hybrid', **res}
        print(row)
        append_jsonl(args.output, row)

if __name__ == '__main__':
    main()
