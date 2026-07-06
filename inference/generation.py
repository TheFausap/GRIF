# inference/generation.py
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
