# training/trainer.py
# -----------------------------
# Optimizer / schedule
# -----------------------------

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from configs.griffin import GriffinConfig
from data.dataset import StreamingTokenChunkDataset, iter_dataset_texts, resolve_valid_split
from inference.generation import generate_text
from models.griffin_lm import GriffinLM
from tokenization.tokeniz import SimpleTokenizer, load_tokenizer, train_bpe_tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.util_funcs import count_parameters, get_device, get_dtype, load_checkpoint, maybe_autocast, save_checkpoint, save_json, set_seed


def build_optimizer(model: nn.Module, cfg: GriffinConfig):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            name.endswith("bias")
            or "norm" in name.lower()
            or "embed" in name.lower()
            or "recurrent_param" in name
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95))

def build_lr_scheduler(optimizer, cfg: GriffinConfig):
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return float(step + 1) / float(max(1, cfg.warmup_steps))
        progress = (step - cfg.warmup_steps) / float(max(1, cfg.max_steps - cfg.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_lr_finder(model, optimizer, train_loader, device, cfg):
    import math

    print("🔍 Running LR range test...")

    model.train()

    start_lr = cfg.lr_finder_start
    end_lr = cfg.lr_finder_end
    num_steps = cfg.lr_finder_steps

    gamma = (end_lr / start_lr) ** (1 / num_steps)
    lr = start_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    lrs = []
    losses = []

    avg_loss = 0.0
    beta = 0.98  # smoothing

    step = 0

    for batch in train_loader:
        if step >= num_steps:
            break

        if isinstance(batch, dict):
                inputs = batch["input_ids"].to(device)
                targets = batch["labels"].to(device)
        elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
        else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

        optimizer.zero_grad()

        loss = model(inputs, labels=targets)[1]
        loss.backward()
        optimizer.step()

        # EMA smoothing
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (step + 1))

        lrs.append(lr)
        losses.append(smoothed_loss)

        # update LR exponentially
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # stop early if exploding
        if step > 50 and smoothed_loss > 4 * min(losses):
            print("⚠️ Loss diverged early, stopping LR test")
            break

        step += 1

    print("✅ LR finder done")
    return lrs, losses

def analyze_lr_curve(lrs, losses):
    import numpy as np

    losses = np.array(losses)
    lrs = np.array(lrs)

    min_loss_idx = losses.argmin()

    # find divergence point
    for i in range(len(losses)):
        if losses[i] > 4 * losses[min_loss_idx]:
            div_idx = i
            break
    else:
        div_idx = len(losses) - 1

    lr_min_loss = lrs[min_loss_idx]
    lr_diverge = lrs[div_idx]
    suggested_lr = lr_diverge / 10

    print(f"\n📊 LR FINDER RESULTS")
    print(f"Min loss LR: {lr_min_loss:.2e}")
    print(f"Divergence LR: {lr_diverge:.2e}")
    print(f"✅ Suggested LR: {suggested_lr:.2e}\n")

    return suggested_lr

def plot_lr_finder(lrs, losses, save_path="lr_finder.png"):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("LR Finder")
    plt.grid()

    plt.savefig(save_path)
    print(f"📈 Saved LR curve to {save_path}")

# -----------------------------
# Train / eval / generate
# -----------------------------

def evaluate(model, loader, tokenizer: SimpleTokenizer, device, dtype, max_batches: int = 40):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            with maybe_autocast(device, dtype):
                _, loss = model(x, y)
            losses.append(loss.item())
        sample = generate_text(model, tokenizer, "The future of AI ", device)
        print("\n[SAMPLE]\n", sample[:200], "\n")
    model.train()
    mean_loss = float(sum(losses) / max(1, len(losses)))
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, ppl

def make_dataloaders(cfg: GriffinConfig, tokenizer: SimpleTokenizer):
    train_ds = StreamingTokenChunkDataset(
        cfg.dataset, cfg.train_split, tokenizer, cfg.seq_len, cfg.text_field, cfg.streaming
    )
    valid_split = resolve_valid_split(cfg.dataset, cfg.valid_split)
    valid_ds = StreamingTokenChunkDataset(
        cfg.dataset, valid_split, tokenizer, cfg.seq_len, cfg.text_field, cfg.streaming, max_docs=2048
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return train_loader, valid_loader

def maybe_train_tokenizer(cfg: GriffinConfig) -> SimpleTokenizer:
    tok_path = Path(cfg.tokenizer_path)
    if tok_path.exists() and not cfg.train_tokenizer:
        print(f"[tokenizer] loading existing tokenizer from {tok_path}")
        return load_tokenizer(str(tok_path))

    print(f"[tokenizer] training BPE tokenizer -> {tok_path}")
    text_iter = iter_dataset_texts(
        cfg.dataset, cfg.train_split, cfg.text_field, cfg.streaming, limit=cfg.tokenizer_train_docs
    )
    tok = train_bpe_tokenizer(
        text_iter=text_iter,
        tokenizer_path=str(tok_path),
        cfg=cfg,
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
    )
    return tok

def train(cfg: GriffinConfig):
    set_seed(cfg.seed)
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "config.json", asdict(cfg))

    device = get_device(cfg)
    dtype = get_dtype(cfg, device)

    print(f"[device] using {device} dtype={dtype}")
    if device.type == "mps":
        print("[device] MPS backend detected. For first stable runs, keep dtype=float32 and num_workers=0.")

    tokenizer = maybe_train_tokenizer(cfg)
    cfg.vocab_size = len(tokenizer.vocab)  # align model vocab with trained tokenizer
    save_json(outdir / "resolved_config.json", asdict(cfg))

    train_loader, valid_loader = make_dataloaders(cfg, tokenizer)

    model = GriffinLM(cfg).to(device)
    if cfg.compile and hasattr(torch, "compile") and device.type != "mps":
        model = torch.compile(model)

    optimizer = build_optimizer(model, cfg)

    # ---- LR FINDER HOOK ----
    if getattr(cfg, "lr_finder", False):
        lrs, losses = run_lr_finder(model, optimizer, train_loader, device, cfg)

        suggested_lr = analyze_lr_curve(lrs, losses)
        plot_lr_finder(lrs, losses)

        print(f"💡 Updating LR from {cfg.lr} → {suggested_lr}")

        cfg.lr = suggested_lr

        # reinitialize optimizer + scheduler ⚠️ IMPORTANT
        optimizer = build_optimizer(model, cfg)
        scheduler = build_lr_scheduler(optimizer, cfg)
    else:
        scheduler = build_lr_scheduler(optimizer, cfg)

    start_step = 0
    best_val = float("inf")
    if cfg.resume:
        ckpt = load_checkpoint(cfg.resume, model, optimizer, scheduler, map_location=device)
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val_loss", float("inf")))
        print(f"[resume] loaded step={start_step} best_val={best_val:.4f}")

    print(f"[model] params={count_parameters(model):,}")

    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    train_iter = iter(train_loader)
    for step in range(start_step, cfg.max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        with maybe_autocast(device, dtype):
            _, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps

        loss.backward()
        running_loss += loss.item() * cfg.grad_accum_steps

        if (step + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if (step + 1) % cfg.log_interval == 0:
            dt = time.time() - t0
            toks = cfg.log_interval * cfg.batch_size * cfg.seq_len
            toks_per_s = toks / max(dt, 1e-6)
            cur_loss = running_loss / cfg.log_interval
            cur_ppl = math.exp(cur_loss) if cur_loss < 20 else float("inf")
            lr = scheduler.get_last_lr()[0]
            print(json.dumps({
                "step": step + 1,
                "loss": round(cur_loss, 4),
                "ppl": round(cur_ppl, 3) if math.isfinite(cur_ppl) else "inf",
                "lr": lr,
                "tok/s": round(toks_per_s, 1),
            }))
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % cfg.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, valid_loader, tokenizer, device, dtype, max_batches=cfg.eval_batches)
            print(json.dumps({
                "step": step + 1,
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 3) if math.isfinite(val_ppl) else "inf",
            }))
            save_checkpoint(outdir / "checkpoint_last.pt", model, optimizer, scheduler, step + 1, cfg, best_val)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(outdir / "checkpoint_best.pt", model, optimizer, scheduler, step + 1, cfg, best_val)
                print(f"[checkpoint] new best checkpoint_best.pt @ step={step+1} val_loss={val_loss:.4f}")
            if device.type == "mps":
                torch.mps.empty_cache()

        if (step + 1) % cfg.save_interval == 0:
            save_checkpoint(outdir / f"checkpoint_step{step+1}.pt", model, optimizer, scheduler, step + 1, cfg, best_val)

    save_checkpoint(outdir / "checkpoint_final.pt", model, optimizer, scheduler, cfg.max_steps, cfg, best_val)
    print(f"[done] final checkpoint -> {outdir / 'checkpoint_final.pt'}")
