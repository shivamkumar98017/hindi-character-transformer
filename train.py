"""
Train the character-level Hindi Transformer.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import SimpleTokenizer, SimpleLLM, TextDataset, load_sample_texts


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train Hindi Character-Level Transformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--data", type=str, default="data/sample_hindi.txt")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Data
    texts = load_sample_texts(args.data)
    print(f"Loaded {len(texts)} sentences")

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts, min_freq=1)  # min_freq=1 for tiny dataset

    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_length=args.max_length,
    ).to(device)

    print(f"Total trainable parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["<pad>"])

    # Training loop
    losses = []
    best_loss = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_vocab": tokenizer.vocab,
                    "tokenizer_idx_to_token": tokenizer.idx_to_token,
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "d_ff": args.d_ff,
                        "max_length": args.max_length,
                    },
                },
                save_dir / "best_model.pt",
            )

    # Final save
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_vocab": tokenizer.vocab,
            "tokenizer_idx_to_token": tokenizer.idx_to_token,
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "d_ff": args.d_ff,
                "max_length": args.max_length,
            },
        },
        save_dir / "final_model.pt",
    )
    tokenizer.save(save_dir / "vocab.txt")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=150)
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints & plots saved to: {save_dir}/")


if __name__ == "__main__":
    main()
