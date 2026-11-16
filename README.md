<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" alt="PyTorch" width="100">
</p>

<h1 align="center">Hindi Character-Level Transformer</h1>

<p align="center">
  <b>Build a tiny Hindi Language Model from scratch</b><br>
  No Hugging Face • No pre-trained weights • Pure PyTorch + Devanagari
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Purpose-Educational-orange" alt="Educational">
</p>

---

A clean, educational implementation of a **decoder-only Transformer** trained to generate Hindi text one character at a time.

---

## Highlights

- **From-scratch Transformer** — Multi-Head Attention, causal masking, residual connections, LayerNorm, GELU FFN
- **Hindi-first** — Character-level tokenizer that understands Devanagari (`U+0900–U+097F`)
- **Minimal & readable** — ~300 lines of core code, heavily commented
- **Ready to run** — Train in minutes on CPU, generate instantly
- **Loss curves & checkpoints** — Automatic saving of best model + training plot

---

## Architecture

```
Input tokens
    ↓
Token Embedding + Positional Embedding
    ↓
┌─────────────────────────────┐
│  Transformer Block × N      │
│  ├─ Causal Multi-Head Attn  │
│  ├─ Residual + LayerNorm    │
│  ├─ Feed-Forward (GELU)     │
│  └─ Residual + LayerNorm    │
└─────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Linear Head → Vocab Logits
```

**Default config** (tiny & educational):

| Parameter       | Value |
|-----------------|-------|
| `d_model`       | 128   |
| `n_heads`       | 8     |
| `n_layers`      | 4     |
| `d_ff`          | 512   |
| `max_length`    | 64    |
| Parameters      | ~870K |

---

## Project Structure

```
hindi-character-transformer/
├── src/
│   ├── __init__.py
│   ├── tokenizer.py      # Character-level Hindi tokenizer
│   ├── model.py          # MultiHeadAttention, TransformerBlock, SimpleLLM
│   └── dataset.py        # TextDataset + sample data loader
├── data/
│   └── sample_hindi.txt  # 20 simple Hindi sentences
├── train.py              # Full training script
├── generate.py           # Text generation CLI
├── requirements.txt
├── notebooks/            # Original Colab-style notebook (optional)
└── README.md
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/shivamkumar98017/hindi-character-transformer.git
cd hindi-character-transformer
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py --epochs 100 --batch-size 4
```

Training finishes in ~1–2 minutes on CPU. You will see:

```
Hindi Vocabulary Size: 42
Total trainable parameters: 869,674
Epoch   1/100 | Loss: 3.2145
Epoch  10/100 | Loss: 0.7138
...
Epoch 100/100 | Loss: 0.1362
```

Checkpoints are saved to `checkpoints/`:
- `best_model.pt`
- `final_model.pt`
- `vocab.txt`
- `loss_curve.png`

### 3. Generate

```bash
python generate.py --prompt "मैं" --num-samples 5 --temperature 0.8
```

Example output:

```
[1] मैं स्कूल जा रहा हूँ।
[2] मैं संगीत सुनना पसंद करता हूँ।
[3] मैं पार्क में खेलते हैं।
```

More prompts to try:

```bash
python generate.py --prompt "आज"
python generate.py --prompt "बिल्ली"
python generate.py --prompt "खिड़की"
python generate.py --prompt "भारत"
```

---

## How the Tokenizer Works

```python
from src import SimpleTokenizer

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(["आज मौसम बहुत अच्छा है।", "मैं स्कूल जा रहा हूँ।"])

print(tokenizer.encode("मैं"))
# → [2, 27, 15, 3]   (<bos> म ैं <eos>)

print(tokenizer.decode([2, 27, 15, 3]))
# → मैं
```

- Only keeps Devanagari characters + basic punctuation
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- Extremely small vocab (~40–60 tokens) → perfect for learning

---

## Key Learning Points

| Concept                    | Where to look          |
|---------------------------|------------------------|
| Causal self-attention     | `src/model.py` → `MultiHeadAttention` |
| Residual + Pre-Norm       | `TransformerBlock`     |
| Character-level LM        | `src/tokenizer.py`     |
| Next-token prediction     | `TextDataset` + loss   |
| Temperature / top-k sampling | `generate.py`       |

This project is intentionally **tiny** so you can:
1. Read every line of the Transformer
2. Watch loss drop to near-zero on a toy dataset
3. Experiment with `d_model`, layers, temperature, etc.

---

## Expected Training Curve

With the default 20-sentence dataset the model quickly memorizes the patterns (loss ≈ 0.13–0.15). This is expected and useful for learning. For real generation quality you would need:

- Much larger Hindi corpus (e.g. IndicCorp, OSCAR Hindi)
- Word / subword tokenization (BPE / SentencePiece)
- Larger model (d_model ≥ 256–512, more layers)

---

## Customization

```bash
# Bigger model
python train.py --d-model 256 --n-layers 6 --n-heads 8 --epochs 50

# Faster training
python train.py --batch-size 8 --lr 3e-4

# Different sampling
python generate.py --temperature 1.2 --top-k 20 --prompt "कल"
```

---

## Inspired By

- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al.)
- Andrej Karpathy’s nanoGPT / minGPT
- Character-level language modeling classics

---

## License

MIT License — feel free to use this for learning, teaching, or building upon.

---

## Author

**Shivam Kumar**  
AI/ML Engineer • Cybersecurity Enthusiast  
Motihari, Bihar

[GitHub](https://github.com/shivamkumar98017) • [Instagram](https://www.instagram.com/shivamkumar98017/) • [Email](mailto:shivamk34442@gmail.com)

---

<p align="center">
  <sub>Made with ❤️ for the Hindi NLP community</sub>
</p>
