from __future__ import annotations

import os

from tokenization.tokeniz import SimpleTokenizer
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset

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

def iter_dataset_texts(
    dataset_name: str,
    split: str,
    text_field: str = "text",
    streaming: bool = True,
    limit: Optional[int] = None,
    shuffle_buffer: int = 0,
    seed: int = 42,
) -> Iterator[str]:
    load_dataset, _ = _import_datasets()
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    if shuffle_buffer > 0:
        if streaming:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        else:
            ds = ds.shuffle(seed=seed)
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
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: SimpleTokenizer,
        seq_len: int,
        text_field: str = "text",
        streaming: bool = True,
        max_docs: Optional[int] = None,
        skip_chunks: int = 0,
        shuffle_buffer: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.streaming = streaming
        self.max_docs = max_docs
        self.skip_chunks = max(0, skip_chunks)
        self.shuffle_buffer = max(0, shuffle_buffer)
        self.seed = seed

    def __iter__(self):
        buffer = []
        skipped = 0
        for text in iter_dataset_texts(
            self.dataset_name,
            self.split,
            self.text_field,
            self.streaming,
            self.max_docs,
            self.shuffle_buffer,
            self.seed,
        ):
            ids = self.tokenizer.encode(text, add_bos=False, add_eos=True)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                if skipped < self.skip_chunks:
                    skipped += 1
                    continue
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
