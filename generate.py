
import argparse

import torch

from inference.generation import generate_text, load_for_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a Griffin checkpoint. Model settings are read from the checkpoint."
    )
    parser.add_argument("checkpoint", help="Path to a checkpoint_*.pt file")
    parser.add_argument("prompt", nargs="?", default="Once upon a time")
    parser.add_argument("--tokenizer", help="Override the tokenizer path saved in the checkpoint")
    parser.add_argument("--tokens", type=int, default=120, help="Maximum number of new tokens")
    parser.add_argument("--temperature", type=float, default=0.9, help="0 for greedy decoding")
    parser.add_argument("--top-k", type=int, default=50, help="0 disables top-k filtering")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error("--tokens must be non-negative")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer, device = load_for_inference(
        args.checkpoint, tokenizer_path=args.tokenizer, device=args.device
    )
    print(generate_text(
        model,
        tokenizer,
        args.prompt,
        device=device,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ))
