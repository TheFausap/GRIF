# -----------------------------
# Utilities
# -----------------------------

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import random
from configs.griffin import GriffinConfig
import torch
import torch.nn as nn

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device(cfg: GriffinConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_dtype(cfg: GriffinConfig, device: torch.device) -> torch.dtype:
    if cfg.dtype == "float16":
        return torch.float16
    if cfg.dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32

def maybe_autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    # MPS mixed precision support varies by op/version; default script trains in float32 for stability.
    class _NullCtx:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return _NullCtx()

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def save_checkpoint(path: Path, model, optimizer, scheduler, step, cfg: GriffinConfig, best_val_loss=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": asdict(cfg),
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)

def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt

def parse_args():
    parser = argparse.ArgumentParser(description="Runnable Griffin-style small LM trainer for Mac MPS / CPU / CUDA.")
    parser.add_argument("--mode", choices=["train", "generate"], default="train")

    # Data/tokenizer
    parser.add_argument("--dataset", type=str, default=GriffinConfig.dataset)
    parser.add_argument("--train_split", type=str, default=GriffinConfig.train_split)
    parser.add_argument("--valid_split", type=str, default=GriffinConfig.valid_split)
    parser.add_argument("--text_field", type=str, default=GriffinConfig.text_field)
    parser.add_argument("--streaming", action="store_true", default=GriffinConfig.streaming)
    parser.add_argument("--no_streaming", dest="streaming", action="store_false")
    parser.add_argument("--tokenizer_path", type=str, default=GriffinConfig.tokenizer_path)
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--tokenizer_train_docs", type=int, default=GriffinConfig.tokenizer_train_docs)
    parser.add_argument("--vocab_size", type=int, default=GriffinConfig.vocab_size)
    parser.add_argument("--min_frequency", type=int, default=GriffinConfig.min_frequency)

    # Model
    parser.add_argument("--d_model", type=int, default=GriffinConfig.d_model)
    parser.add_argument("--d_rnn", type=int, default=GriffinConfig.d_rnn)
    parser.add_argument("--n_layers", type=int, default=GriffinConfig.n_layers)
    parser.add_argument("--n_heads", type=int, default=GriffinConfig.n_heads)
    parser.add_argument("--n_kv_heads", type=int, default=GriffinConfig.n_kv_heads)
    parser.add_argument("--mlp_mult", type=int, default=GriffinConfig.mlp_mult)
    parser.add_argument("--seq_len", type=int, default=GriffinConfig.seq_len)
    parser.add_argument("--attention_window", type=int, default=GriffinConfig.attention_window)
    parser.add_argument("--conv_kernel", type=int, default=GriffinConfig.conv_kernel)
    parser.add_argument("--dropout", type=float, default=GriffinConfig.dropout)
    parser.add_argument("--layer_pattern", type=str, default=GriffinConfig.layer_pattern)

    # Training
    parser.add_argument("--batch_size", type=int, default=GriffinConfig.batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=GriffinConfig.grad_accum_steps)
    parser.add_argument("--max_steps", type=int, default=GriffinConfig.max_steps)
    parser.add_argument("--lr", type=float, default=GriffinConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=GriffinConfig.weight_decay)
    parser.add_argument("--warmup_steps", type=int, default=GriffinConfig.warmup_steps)
    parser.add_argument("--max_grad_norm", type=float, default=GriffinConfig.max_grad_norm)
    parser.add_argument("--seed", type=int, default=GriffinConfig.seed)
    parser.add_argument("--log_interval", type=int, default=GriffinConfig.log_interval)
    parser.add_argument("--eval_interval", type=int, default=GriffinConfig.eval_interval)
    parser.add_argument("--eval_batches", type=int, default=GriffinConfig.eval_batches)
    parser.add_argument("--save_interval", type=int, default=GriffinConfig.save_interval)
    parser.add_argument("--num_workers", type=int, default=GriffinConfig.num_workers)
    parser.add_argument("--lr_finder", action="store_true")
    parser.add_argument("--lr_finder_steps", type=int, default=GriffinConfig.lr_finder_steps)
    parser.add_argument("--lr_finder_start", type=float, default=GriffinConfig.lr_finder_start)
    parser.add_argument("--lr_finder_end", type=float, default=GriffinConfig.lr_finder_end)

    # Runtime
    parser.add_argument("--output_dir", type=str, default=GriffinConfig.output_dir)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=GriffinConfig.device)
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default=GriffinConfig.dtype)
    parser.add_argument("--compile", action="store_true")

    # Generation
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=GriffinConfig.prompt)
    parser.add_argument("--max_new_tokens", type=int, default=GriffinConfig.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=GriffinConfig.temperature)
    parser.add_argument("--top_k", type=int, default=GriffinConfig.top_k)

    args = parser.parse_args()
    return GriffinConfig(**vars(args))