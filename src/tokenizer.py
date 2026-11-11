"""
Character-level tokenizer specialized for Hindi (Devanagari script).
"""

import re
from collections import Counter
from typing import List, Dict


class SimpleTokenizer:
    """
    A simple character-level tokenizer for Hindi text.

    Keeps Devanagari characters (U+0900–U+097F), whitespace, and basic punctuation.
    Special tokens: <pad>, <unk>, <bos>, <eos>
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of raw Hindi strings.
            min_freq: Minimum frequency for a character to be included.
        """
        all_chars = []
        for text in texts:
            # Keep only Hindi characters, whitespace, and basic punctuation
            cleaned = re.sub(r"[^\u0900-\u097F\s.,!?]", "", text)
            all_chars.extend(list(cleaned))

        char_counts = Counter(all_chars)

        # Special tokens
        self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx_to_token = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}

        idx = 4
        for char, count in char_counts.most_common():
            if count >= min_freq:
                self.vocab[char] = idx
                self.idx_to_token[idx] = char
                idx += 1

        self.vocab_size = len(self.vocab)
        print(f"Hindi Vocabulary Size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs (with <bos> and <eos>)."""
        tokens = [self.vocab["<bos>"]]
        for char in text:
            tokens.append(self.vocab.get(char, self.vocab["<unk>"]))
        tokens.append(self.vocab["<eos>"])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into a string (skipping special tokens)."""
        text = ""
        for token_id in token_ids:
            if token_id in (
                self.vocab["<pad>"],
                self.vocab["<bos>"],
                self.vocab["<eos>"],
            ):
                continue
            elif token_id == self.vocab["<unk>"]:
                text += "\ufffd"
            else:
                text += self.idx_to_token.get(token_id, "")
        return text

    def save(self, path: str) -> None:
        """Save vocabulary to a simple text file."""
        with open(path, "w", encoding="utf-8") as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{token}\n")

    def load(self, path: str) -> None:
        """Load vocabulary from a text file."""
        self.vocab = {}
        self.idx_to_token = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                idx_str, token = line.rstrip("\n").split("\t", 1)
                idx = int(idx_str)
                self.vocab[token] = idx
                self.idx_to_token[idx] = token
        self.vocab_size = len(self.vocab)
