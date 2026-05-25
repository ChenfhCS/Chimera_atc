#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from chimera.dynamic_hybrid_model import DynamicHybridConfig, DynamicHybridForCausalLM, load_tokenizer_for_hybrid
from chimera.utils import ensure_dir, read_yaml, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    cfg_dict = read_yaml(args.config)
    cfg = DynamicHybridConfig(**cfg_dict)
    model = DynamicHybridForCausalLM.from_pretrained_branches(cfg, device_map=None)
    tok = load_tokenizer_for_hybrid(cfg.attn_model_name_or_path, cfg.trust_remote_code)
    ensure_dir(args.output_dir)
    torch.save({'config': cfg.__dict__, 'state_dict': model.state_dict()}, f'{args.output_dir}/dynamic_hybrid.pt')
    tok.save_pretrained(args.output_dir)
    with open(f'{args.output_dir}/config.json', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)
    print('Saved initialized Dynamic Hybrid to', args.output_dir)
    print(model.trainable_parameter_summary())

if __name__ == '__main__':
    main()
