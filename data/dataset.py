from __future__ import annotations

import os

from tokenization.tokeniz import SimpleTokenizer
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
from typing import Iterator, Optional, Sequence

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


def parse_dataset_mix(dataset: str, dataset_weights: Optional[str] = None) -> tuple[list[str], list[float]]:
    names = [name.strip() for name in dataset.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one dataset name is required")
    if dataset_weights is None:
        weights = [1.0 / len(names)] * len(names)
    else:
        try:
            weights = [float(value.strip()) for value in dataset_weights.split(",")]
        except ValueError as exc:
            raise ValueError("dataset_weights must contain comma-separated numbers") from exc
        if len(weights) != len(names):
            raise ValueError("dataset_weights must have one value for each dataset")
        if any(weight <= 0 for weight in weights):
            raise ValueError("dataset_weights must contain only positive values")
        total = sum(weights)
        weights = [weight / total for weight in weights]
    return names, weights


def iter_mixed_dataset_texts(
    dataset_names: Sequence[str],
    weights: Sequence[float],
    split: str,
    text_field: str = "text",
    streaming: bool = True,
    shuffle_buffer: int = 0,
    seed: int = 42,
) -> Iterator[str]:
    """Deterministically interleave document streams according to sampling weights."""
    rng = random.Random(seed)
    iterators = [
        iter(iter_dataset_texts(
            name, split, text_field, streaming,
            shuffle_buffer=shuffle_buffer, seed=seed + index,
        ))
        for index, name in enumerate(dataset_names)
    ]
    active = list(range(len(iterators)))
    while active:
        active_weights = [weights[index] for index in active]
        selected = rng.choices(active, weights=active_weights, k=1)[0]
        try:
            yield next(iterators[selected])
        except StopIteration:
            active.remove(selected)

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
        dataset_weights: Optional[str] = None,
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
        self.dataset_names, self.weights = parse_dataset_mix(dataset_name, dataset_weights)

    def __iter__(self):
        buffer = []
        skipped = 0
        if len(self.dataset_names) == 1:
            texts = iter_dataset_texts(
                self.dataset_names[0], self.split, self.text_field, self.streaming,
                self.max_docs, self.shuffle_buffer, self.seed,
            )
        else:
            texts = iter_mixed_dataset_texts(
                self.dataset_names, self.weights, self.split, self.text_field,
                self.streaming, self.shuffle_buffer, self.seed,
            )
        for text in texts:
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
