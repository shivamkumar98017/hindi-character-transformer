"""
Dataset utilities for character-level language modeling.
"""

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import SimpleTokenizer


def load_sample_texts(path: str = "data/sample_hindi.txt") -> List[str]:
    """Load Hindi sentences from a text file (one sentence per line)."""
    path = Path(path)
    if not path.exists():
        # Fallback to embedded samples if file is missing
        return [
            "आज मौसम बहुत अच्छा है।",
            "मैं स्कूल जा रहा हूँ।",
            "तुम क्या कर रहे हो?",
            "मुझे चाय पसंद है।",
            "यह किताब बहुत रोचक है।",
            "हम पार्क में खेलते हैं।",
            "कल मेरी परीक्षा है।",
            "माँ खाना बना रही हैं।",
            "सड़क पर बहुत ट्रैफिक है।",
            "मैं संगीत सुनना पसंद करता हूँ।",
            "तुम्हारा नाम क्या है?",
            "भारत एक सुंदर देश है।",
            "हमने फिल्म देखी।",
            "बिल्ली कुर्सी पर सो रही है।",
            "खिड़की से हवा आ रही है।",
            "पढ़ाई समय पर करनी चाहिए।",
            "पापा दफ़्तर जा चुके हैं।",
            "छुट्टी के दिन मैं घूमने गया।",
            "बच्चे मैदान में दौड़ रहे हैं।",
            "आज रात को चाँदनी बहुत सुंदर है।",
        ]
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class TextDataset(Dataset):
    """
    Simple next-token prediction dataset.

    Each sample is (input_ids, target_ids) where target is input shifted by 1.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: SimpleTokenizer,
        max_length: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[List[int]] = []

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_length:
                # Pad to max_length
                tokens = tokens + [tokenizer.vocab["<pad>"]] * (max_length - len(tokens))
                self.data.append(tokens)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids
