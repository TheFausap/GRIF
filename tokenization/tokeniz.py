from __future__ import annotations

import os
from pathlib import Path

from configs.griffin import GriffinConfig
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dataclasses import dataclass, asdict
from typing import Iterable

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
    
def train_bpe_tokenizer(text_iter: Iterable[str], cfg: GriffinConfig, tokenizer_path: str, vocab_size: int, min_frequency: int = 2):
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
    tokenizer.save(str(Path(cfg.output_dir) / "tokenizer.json"))
    return load_tokenizer(tokenizer_path)

def load_tokenizer(tokenizer_path: str) -> SimpleTokenizer:
    Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers = _import_tokenizers()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return SimpleTokenizer(tokenizer)