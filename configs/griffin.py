from typing import Optional
from dataclasses import dataclass
import argparse

@dataclass
class GriffinConfig:
    # Data
    dataset: str = "roneneldan/TinyStories"  # or emozilla/pg19
    dataset_weights: Optional[str] = None  # comma-separated; dataset may also be comma-separated
    train_split: str = "train"
    valid_split: str = "validation"
    text_field: str = "text"
    streaming: bool = True
    shuffle_buffer: int = 10000

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
    max_tokens: Optional[int] = None
    lr: float = 3e-4
    min_lr_ratio: float = 0.0
    lr_finder: bool = True
    lr_finder_steps: int = 800
    lr_finder_start: float = 1e-6
    lr_finder_end: float = 1
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
    eval_on_resume: bool = False
    finetune_reset: bool = False
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
