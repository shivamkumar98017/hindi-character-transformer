<div align="center">

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" alt="PyTorch" width="90">

# Hindi Character-Level Transformer

**Build a tiny Hindi Language Model completely from scratch**  
No Hugging Face · No pre-trained weights · Pure PyTorch + Devanagari

<br>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Educational](https://img.shields.io/badge/Purpose-Educational-F59E0B?style=for-the-badge)

</div>

---

A clean, educational implementation of a **decoder-only Transformer** trained to generate Hindi text one character at a time.

Perfect for understanding how modern language models work under the hood — without any black-box libraries.

---

## Highlights

| Feature | Description |
|--------|-------------|
| **From-scratch Transformer** | Multi-Head Attention, causal masking, residual connections, LayerNorm, GELU FFN |
| **Hindi-first** | Character-level tokenizer that understands Devanagari (`U+0900–U+097F`) |
| **Minimal & readable** | ~300 lines of core code, heavily commented |
| **Ready to run** | Train in minutes on CPU, generate instantly |
| **Loss curves & checkpoints** | Automatic saving of best model + training plot |

---

## Architecture

```text
Input tokens
     │
     ▼
Token Embedding + Positional Embedding
     │
     ▼
┌─────────────────────────────┐
│  Transformer Block × N      │
│  ├─ Causal Multi-Head Attn  │
│  ├─ Residual + LayerNorm    │
│  ├─ Feed-Forward (GELU)     │
│  └─ Residual + LayerNorm    │
└─────────────────────────────┘
     │
     ▼
Final LayerNorm
     │
     ▼
Linear Head → Vocab Logits
```

**Default config** (tiny & educational):

| Parameter       | Value   |
|-----------------|---------| 
| `d_model`       | 128     |
| `n_heads`       | 8       |
| `n_layers`      | 4       |
| `d_ff`          | 512     |
| `max_length`    | 64      |
| Parameters      | ~870K   |

---

## Project Structure

```text
hindi-character-transformer/
├── src/
│   ├── tokenizer.py      # Character-level Hindi tokenizer
│   ├── model.py          # MultiHeadAttention, TransformerBlock, SimpleLLM
│   └── dataset.py        # TextDataset + sample data loader
├── data/
│   └── sample_hindi.txt  # 20 simple Hindi sentences
├── train.py              # Full training script
├── generate.py           # Text generation CLI
├── requirements.txt
└── README.md
```

---

## Quick Start

**1. Install**

```bash
git clone https://github.com/shivamkumar98017/hindi-character-transformer.git
cd hindi-character-transformer
pip install -r requirements.txt
```

**2. Train**

```bash
python train.py --epochs 100 --batch-size 4
```

```text
Hindi Vocabulary Size: 42
Total trainable parameters: 869,674
Epoch   1/100 | Loss: 3.2145
Epoch  10/100 | Loss: 0.7138
...
Epoch 100/100 | Loss: 0.1362
```

**3. Generate**

```bash
python generate.py --prompt "मैं" --num-samples 5 --temperature 0.8
```

```text
[1] मैं स्कूल जा रहा हूँ।
[2] मैं संगीत सुनना पसंद करता हूँ।
[3] मैं पार्क में खेलते हैं।
```

Try more prompts:

```bash
python generate.py --prompt "आज"
python generate.py --prompt "बिल्ली"
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

- Keeps only Devanagari characters + basic punctuation
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- Extremely small vocab (~40–60 tokens) → perfect for learning

---

## Key Learning Points

| Concept                      | File / Location                  |
|-----------------------------|----------------------------------|
| Causal self-attention       | `src/model.py` → `MultiHeadAttention` |
| Residual + Pre-Norm         | `TransformerBlock`               |
| Character-level LM          | `src/tokenizer.py`               |
| Next-token prediction       | `TextDataset` + loss             |
| Temperature / top-k sampling| `generate.py`                    |

This project is intentionally **tiny** so you can:
1. Read every line of the Transformer
2. Watch loss drop to near-zero on a toy dataset
3. Experiment freely with `d_model`, layers, temperature, etc.

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

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- Andrej Karpathy’s nanoGPT / minGPT
- Classic character-level language modeling

---

<div align="center">

**Shivam Kumar**  
AI/ML Engineer • Cybersecurity Enthusiast  
Motihari, Bihar

[GitHub](https://github.com/shivamkumar98017) · [Instagram](https://www.instagram.com/shivamkumar98017/) · [LinkedIn](https://www.linkedin.com/in/shivam-kumar-b24677431/) · [X](https://x.com/shivamk344007) · [Email](mailto:shivamk34442@gmail.com)

<br>

<sub>Made with ❤️ for the Hindi NLP community</sub>

</div>
