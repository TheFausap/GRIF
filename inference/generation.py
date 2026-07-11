# inference/generation.py
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

from configs.griffin import GriffinConfig
from models.griffin_lm import GriffinLM
from tokenization.tokeniz import SimpleTokenizer, load_tokenizer
from utils.util_funcs import get_device, load_checkpoint

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
        logits = logits[:, -1, :]
        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat([x, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break
            continue
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == tokenizer.eos_id:
            break
    return tokenizer.decode(x[0].tolist())


def load_for_inference(checkpoint_path: str, tokenizer_path: str | None = None, device: str = "auto"):
    """Load a model using the architecture saved inside its training checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = get_device(GriffinConfig(device=device))
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint has no saved model configuration")

    valid_fields = {field.name for field in fields(GriffinConfig)}
    cfg = GriffinConfig(**{key: value for key, value in saved_config.items() if key in valid_fields})
    cfg.device = device

    requested_tokenizer = tokenizer_path or cfg.tokenizer_path
    candidates = [Path(requested_tokenizer)]
    if not tokenizer_path:
        candidates.extend([
            checkpoint_path.parent / requested_tokenizer,
            checkpoint_path.parent / "tokenizer.json",
        ])
    tokenizer_file = next((path for path in candidates if path.is_file()), None)
    if tokenizer_file is None:
        raise FileNotFoundError(
            f"Tokenizer not found: {requested_tokenizer}. Pass its path with --tokenizer."
        )

    tokenizer = load_tokenizer(str(tokenizer_file))
    cfg.vocab_size = len(tokenizer.vocab)
    model = GriffinLM(cfg).to(target_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, target_device


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
