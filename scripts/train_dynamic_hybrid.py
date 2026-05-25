#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from chimera.dynamic_hybrid_model import DynamicHybridConfig, DynamicHybridForCausalLM, load_tokenizer_for_hybrid
from chimera.data import build_lm_training_texts
from chimera.utils import ensure_dir, read_yaml, set_seed


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]
    def collate(self, batch):
        enc = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100
        enc['labels'] = labels
        return enc


def load_model(cfg, checkpoint=None):
    model = DynamicHybridForCausalLM.from_pretrained_branches(cfg, device_map=None)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--datasets', nargs='+', default=['arc-e','arc-c','piqa','truthfulqa'])
    ap.add_argument('--max_samples_per_dataset', type=int, default=2000)
    ap.add_argument('--max_length', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--warmup_steps', type=int, default=50)
    ap.add_argument('--routing_mode', default='soft', choices=['soft','hard','straight_through'])
    ap.add_argument('--freeze_branches', action='store_true', help='Train gates only. Default trains gates and branch modules.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    if args.routing_mode == 'hard':
        raise SystemExit(
            "routing_mode='hard' freezes the gate during training (argmax is non-differentiable). "
            "Use 'soft' or 'straight_through'."
        )
    cfg = DynamicHybridConfig(**read_yaml(args.config))
    cfg.routing_mode = args.routing_mode
    tok = load_tokenizer_for_hybrid(cfg.attn_model_name_or_path, cfg.trust_remote_code)
    model = load_model(cfg, args.checkpoint)
    if args.freeze_branches:
        for n, p in model.named_parameters():
            p.requires_grad = '.gate.' in n
    # Cast trainable parameters to fp32 for stable AdamW updates even when the
    # branches were loaded in bf16/fp16. Frozen branch weights stay in their
    # original low-precision dtype.
    for p in model.parameters():
        if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.float()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()

    texts = build_lm_training_texts(args.datasets, split='train', max_samples_per_dataset=args.max_samples_per_dataset)
    ds = TextDataset(texts, tok, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = max(1, len(dl) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, args.warmup_steps, total_steps)

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch, routing_mode=args.routing_mode)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 10 == 0:
                print(json.dumps({'step': step, 'epoch': epoch, 'loss': float(out.loss.detach().cpu())}))
            step += 1
    torch.save({'config': cfg.__dict__, 'state_dict': model.state_dict()}, f'{args.output_dir}/dynamic_hybrid.pt')
    tok.save_pretrained(args.output_dir)
    with open(f'{args.output_dir}/config.json','w') as f: json.dump(cfg.__dict__, f, indent=2)
    print('Saved fine-tuned Dynamic Hybrid to', args.output_dir)

if __name__ == '__main__':
    main()
