from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


# -----------------------------
# Config
# -----------------------------

@dataclass
class GriffinConfig:
    # Data
    dataset: str = "roneneldan/TinyStories"  # or emozilla/pg19
    train_split: str = "train"
    valid_split: str = "validation"
    text_field: str = "text"
    streaming: bool = True

    # Tokenizer
    tokenizer_path: str = "tokenizer.json"
    train_tokenizer: bool = False
    tokenizer_train_docs: int = 20000
    vocab_size: int = 8192
    min_frequency: int = 2

    # Model
    d_model: int = 384
    d_rnn: int = 512
    n_layers: int = 12
    n_heads: int = 6
    n_kv_heads: int = 1
    mlp_mult: int = 3
    seq_len: int = 256
    attention_window: int = 128
    conv_kernel: int = 4
    dropout: float = 0.0
    layer_pattern: str = "recurrent,recurrent,attention"

    # Training
    mode: str = "train"
    batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 3000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    seed: int = 42
    log_interval: int = 20
    eval_interval: int = 200
    eval_batches: int = 40
    save_interval: int = 200
    num_workers: int = 0

    # Checkpoint / runtime
    output_dir: str = "runs/griffin_tinystories_small"
    resume: Optional[str] = None
    device: str = "auto"  # auto|cpu|mps|cuda
    dtype: str = "float32"  # float32|float16|bfloat16
    compile: bool = False

    # Generation
    do_generate: bool = False
    ckpt_path: Optional[str] = None
    prompt: str = "Once upon a time"
    max_new_tokens: int = 120
    temperature: float = 0.9
    top_k: int = 50


# -----------------------------
# Utilities
# -----------------------------

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


# -----------------------------
# Tokenizer
# -----------------------------

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def _import_tokenizers():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
    return Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers

class SimpleTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab["<unk>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True):
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        ids = [int(x) for x in ids if int(x) not in {self.pad_id, self.bos_id, self.eos_id}]
        return self.tokenizer.decode(ids)

def train_bpe_tokenizer(text_iter: Iterable[str], tokenizer_path: str, vocab_size: int, min_frequency: int = 2):
    Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers = _import_tokenizers()
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iter, trainer=trainer)
    tokenizer.save(tokenizer_path)
    return load_tokenizer(tokenizer_path)

def load_tokenizer(tokenizer_path: str) -> SimpleTokenizer:
    Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers = _import_tokenizers()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return SimpleTokenizer(tokenizer)


# -----------------------------
# Dataset
# -----------------------------

def _import_datasets():
    from datasets import load_dataset, get_dataset_split_names
    return load_dataset, get_dataset_split_names

def resolve_valid_split(dataset_name: str, requested: str = "validation") -> str:
    try:
        _, get_dataset_split_names = _import_datasets()
        splits = set(get_dataset_split_names(dataset_name))
        for cand in [requested, "validation", "valid", "test"]:
            if cand in splits:
                return cand
    except Exception:
        pass
    return requested

def iter_dataset_texts(dataset_name: str, split: str, text_field: str = "text", streaming: bool = True, limit: Optional[int] = None) -> Iterator[str]:
    load_dataset, _ = _import_datasets()
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    n = 0
    for ex in ds:
        if text_field not in ex:
            raise KeyError(f"Expected text field '{text_field}', but example keys are: {list(ex.keys())}")
        text = ex[text_field]
        if text and isinstance(text, str):
            yield text
            n += 1
            if limit is not None and n >= limit:
                break

class StreamingTokenChunkDataset(IterableDataset):
    def __init__(self, dataset_name: str, split: str, tokenizer: SimpleTokenizer, seq_len: int, text_field: str = "text",
                 streaming: bool = True, max_docs: Optional[int] = None):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.streaming = streaming
        self.max_docs = max_docs

    def __iter__(self):
        buffer = []
        for text in iter_dataset_texts(self.dataset_name, self.split, self.text_field, self.streaming, self.max_docs):
            ids = self.tokenizer.encode(text, add_bos=False, add_eos=True)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


# -----------------------------
# Model components
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_f * rms
        return (y.to(x.dtype)) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class DepthwiseCausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, groups=channels, bias=False
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)

class RGLRU(nn.Module):
    """
    Practical PyTorch RG-LRU approximation aligned with the Griffin/RecurrentGemma family:
      g_x = sigmoid(Wx x_t + bx)
      g_a = sigmoid(Wa x_t + ba)
      log a_t = -8 * g_a * softplus(lambda)
      h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (g_x * x_t)
    Gates depend only on current input; the recurrence is diagonal.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.in_gate = nn.Linear(dim, dim)
        self.rec_gate = nn.Linear(dim, dim)
        self.recurrent_param = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_gate.weight, std=0.02)
        nn.init.zeros_(self.in_gate.bias)
        nn.init.normal_(self.rec_gate.weight, std=0.02)
        nn.init.zeros_(self.rec_gate.bias)
        nn.init.uniform_(self.recurrent_param, a=-5.0, b=-3.0)

    def forward(self, x, h0=None):
        B, T, R = x.shape
        gx = torch.sigmoid(self.in_gate(x))
        ga = torch.sigmoid(self.rec_gate(x))
        log_a = -8.0 * ga * F.softplus(self.recurrent_param).view(1, 1, R)
        a = torch.exp(log_a)
        gamma_sq = torch.clamp(1.0 - torch.exp(2.0 * log_a), min=self.eps)
        gamma = torch.sqrt(gamma_sq)

        u = gamma * (gx * x)

        h = torch.zeros(B, R, device=x.device, dtype=torch.float32) if h0 is None else h0.float()
        ys = []
        for t in range(T):
            h = a[:, t].float() * h + u[:, t].float()
            ys.append(h.to(x.dtype))
        y = torch.stack(ys, dim=1)
        return y, h

class SlidingWindowMQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        q = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))

        if self.n_kv_heads == 1:
            k = k.expand(B, self.n_heads, T, self.head_dim)
            v = v.expand(B, self.n_heads, T, self.head_dim)
        else:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

        if self.window < T:
            idx = torch.arange(T, device=x.device)
            dist = idx[None, :] - idx[:, None]
            local_mask = dist < -(self.window - 1)
            scores = scores.masked_fill(local_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class GatedMLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 3, dropout: float = 0.0):
        super().__init__()
        d_hidden = d_model * mult
        self.up1 = nn.Linear(d_model, d_hidden)
        self.up2 = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.dropout(F.gelu(self.up1(x)) * self.up2(x)))

class RecurrentBlock(nn.Module):
    def __init__(self, d_model: int, d_rnn: int, conv_kernel: int, dropout: float = 0.0):
        super().__init__()
        self.lin_u = nn.Linear(d_model, d_rnn, bias=False)
        self.lin_y = nn.Linear(d_model, d_rnn, bias=False)
        self.conv = DepthwiseCausalConv1d(d_rnn, kernel_size=conv_kernel)
        self.rglru = RGLRU(d_rnn)
        self.out = nn.Linear(d_rnn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        u = self.lin_u(x)
        y = F.gelu(self.lin_y(x))
        u = self.conv(u)
        h, _ = self.rglru(u)
        return self.out(self.dropout(h * y))

class DecoderLayer(nn.Module):
    def __init__(self, cfg: GriffinConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        if layer_type == "recurrent":
            self.temporal = RecurrentBlock(cfg.d_model, cfg.d_rnn, cfg.conv_kernel, cfg.dropout)
        elif layer_type == "attention":
            self.temporal = SlidingWindowMQA(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.attention_window, cfg.dropout)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        self.mlp = GatedMLP(cfg.d_model, cfg.mlp_mult, cfg.dropout)

    def forward(self, x):
        x = x + self.temporal(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GriffinLM(nn.Module):
    def __init__(self, cfg: GriffinConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        pattern = [x.strip() for x in cfg.layer_pattern.split(",")]
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, pattern[i % len(pattern)]) for i in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied embeddings
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss


# -----------------------------
# Optimizer / schedule
# -----------------------------

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
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------
# Train / eval / generate
# -----------------------------

def evaluate(model, loader, device, dtype, max_batches: int = 40):
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
    model.train()
    mean_loss = float(sum(losses) / max(1, len(losses)))
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, ppl

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

@torch.no_grad()
def generate_text(model, tokenizer: SimpleTokenizer, prompt: str, device, max_new_tokens=120, temperature=0.9, top_k=50):
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        if x.size(1) > model.cfg.seq_len:
            x_cond = x[:, -model.cfg.seq_len :]
        else:
            x_cond = x
        logits, _ = model(x_cond, labels=None)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == tokenizer.eos_id:
            break
    return tokenizer.decode(x[0].tolist())

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
            val_loss, val_ppl = evaluate(model, valid_loader, device, dtype, max_batches=cfg.eval_batches)
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

def generate(cfg: GriffinConfig):
    device = get_device(cfg)
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    cfg.vocab_size = len(tokenizer.vocab)
    model = GriffinLM(cfg).to(device)
    ckpt_path = cfg.ckpt_path or str(Path(cfg.output_dir) / "checkpoint_best.pt")
    load_checkpoint(ckpt_path, model, map_location=device)
    text = generate_text(
        model, tokenizer, cfg.prompt, device=device,
        max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k
    )
    print(text)

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

def main():
    cfg = parse_args()
    if cfg.mode == "train":
        train(cfg)
    else:
        generate(cfg)

if __name__ == "__main__":
    main()
