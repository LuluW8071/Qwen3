from __future__ import annotations

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

BASE_TOKENIZER = "HuggingFaceTB/SmolLM-135M"
EOS_TOKEN = "<|endoftext|>"


class TextTokenizer:
    """Thin wrapper over `tokenizers.Tokenizer` exposing the small API training uses."""

    def __init__(self, tokenizer: Tokenizer):
        self._tk = tokenizer
        self.vocab_size = tokenizer.get_vocab_size(with_added_tokens=False)
        self.eos_token_id = tokenizer.token_to_id(EOS_TOKEN)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tk.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        return [e.ids for e in self._tk.encode_batch(texts, add_special_tokens=add_special_tokens)]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def to_json(self) -> str:
        return self._tk.to_str()

    @classmethod
    def from_json(cls, data: str) -> TextTokenizer:
        return cls(Tokenizer.from_str(data))

    @classmethod
    def from_pretrained(cls, name: str = BASE_TOKENIZER) -> TextTokenizer:
        return cls(Tokenizer.from_pretrained(name))

    def train_new(self, texts, vocab_size: int) -> TextTokenizer:
        """Train a fresh byte-level BPE with the same pre-tokenizer/specials and `vocab_size`."""
        specials = [
            t.content for _, t in sorted(self._tk.get_added_tokens_decoder().items()) if t.special
        ]
        new = Tokenizer(models.BPE())
        new.normalizer = self._tk.normalizer
        new.pre_tokenizer = self._tk.pre_tokenizer
        new.decoder = self._tk.decoder
        new.post_processor = self._tk.post_processor
        new.train_from_iterator(
            texts,
            trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=specials,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ),
            length=len(texts),
        )
        return TextTokenizer(new)
