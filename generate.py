"""
Generate Hindi text from a trained character-level Transformer.
"""

import argparse
from pathlib import Path

import torch

from src import SimpleTokenizer, SimpleLLM


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    model = SimpleLLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_length=config["max_length"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer()
    tokenizer.vocab = ckpt["tokenizer_vocab"]
    tokenizer.idx_to_token = ckpt["tokenizer_idx_to_token"]
    tokenizer.vocab_size = len(tokenizer.vocab)

    return model, tokenizer


@torch.no_grad()
def generate(
    model: SimpleLLM,
    tokenizer: SimpleTokenizer,
    prompt: str = "",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 0,
    device: torch.device = None,
) -> str:
    """
    Autoregressive generation with temperature and optional top-k sampling.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    tokens = tokenizer.encode(prompt)
    # Remove the trailing <eos> that encode() adds, so we can continue
    if tokens and tokens[-1] == tokenizer.vocab["<eos>"]:
        tokens = tokens[:-1]

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= model.max_length:
            break

        logits = model(input_ids)
        next_logits = logits[0, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = -float("inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == tokenizer.vocab["<eos>"]:
            break

        tokens.append(next_token)
        new_token = torch.tensor([[next_token]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, new_token], dim=1)

    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description="Generate Hindi text")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--prompt", type=str, default="मैं", help="Starting prompt")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please train the model first: python train.py")
        return

    model, tokenizer = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print("-" * 50)

    for i in range(args.num_samples):
        text = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"[{i+1}] {text}")


if __name__ == "__main__":
    main()
