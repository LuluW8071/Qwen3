from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.data.tokenizer import EOS_TOKEN, TextTokenizer

TEXTS = ["hello world 123", "the quick brown fox", "pretraining needs no transformers"]


def make_tokenizer() -> TextTokenizer:
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(
        TEXTS,
        trainers.BpeTrainer(vocab_size=300, special_tokens=[EOS_TOKEN],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    return TextTokenizer(tk)


def test_batch_encode_matches_single_encode():
    tok = make_tokenizer()
    assert tok.encode_batch(TEXTS) == [tok.encode(t) for t in TEXTS]


def test_json_round_trip_and_decode_skip_special():
    tok = make_tokenizer()
    again = TextTokenizer.from_json(tok.to_json())
    assert again.encode(TEXTS[0]) == tok.encode(TEXTS[0])
    assert again.eos_token_id == 0 and again.vocab_size == tok.vocab_size
    ids = tok.encode(f"hi{EOS_TOKEN}there")
    assert tok.decode(ids) == "hithere"
    assert EOS_TOKEN in tok.decode(ids, skip_special_tokens=False)


def test_train_new_keeps_specials_and_vocab_size():
    small = make_tokenizer().train_new(TEXTS, 280)
    assert small.eos_token_id == 0
    assert small.vocab_size <= 280
