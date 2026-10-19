"""Character vocabulary shared by the CTC head and the attention decoder.

Why characters and not subwords? AV-HuBERT's English VSR uses 1000 SentencePiece
units, but it was fine-tuned on 433 h of LRS3. With 50-100 h of Hindi and a much
richer morphology, most subwords would be seen only a handful of times. Unicode
code points (~70 symbols) are all seen thousands of times, need no tokenizer
training, and keep the model open-vocabulary (any name, any new word).

The inventory is *fixed* (not scanned from data) so a checkpoint trained on one
version of the dataset still decodes the same way on the next.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from punarvaani.text.hindi import DEVANAGARI_LETTERS, normalize

BLANK, PAD, BOS_EOS, UNK, SPACE = "<blank>", "<pad>", "<s>", "<unk>", " "
SPECIALS = (BLANK, PAD, BOS_EOS, UNK, SPACE)


class CharVocab:
    """id 0 is the CTC blank; ``<s>`` doubles as start and end for the decoder."""

    def __init__(self, symbols: Sequence[str] | None = None):
        symbols = list(symbols) if symbols is not None else [*SPECIALS, *DEVANAGARI_LETTERS]
        if symbols[: len(SPECIALS)] != list(SPECIALS):
            raise ValueError("vocabulary must start with the special symbols")
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate symbols in vocabulary")
        self.symbols = symbols
        self.index = {s: i for i, s in enumerate(symbols)}

    blank_id = property(lambda self: self.index[BLANK])
    pad_id = property(lambda self: self.index[PAD])
    bos_id = property(lambda self: self.index[BOS_EOS])
    eos_id = property(lambda self: self.index[BOS_EOS])
    unk_id = property(lambda self: self.index[UNK])
    space_id = property(lambda self: self.index[SPACE])

    def __len__(self) -> int:
        return len(self.symbols)

    def encode(self, text: str, already_normalized: bool = False) -> list[int]:
        if not already_normalized:
            text = normalize(text)
        return [self.index.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Iterable[int], strip_specials: bool = True) -> str:
        out = []
        for i in ids:
            sym = self.symbols[int(i)]
            if strip_specials and sym in SPECIALS and sym != SPACE:
                continue
            out.append(sym)
        return " ".join("".join(out).split())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.symbols, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharVocab:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_symbols(cls, letters: str) -> CharVocab:
        """A vocabulary over an arbitrary alphabet (used by the synthetic toy task)."""
        return cls([*SPECIALS, *[c for c in letters if c != " "]])
