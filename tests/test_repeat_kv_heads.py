import pytest
import torch

from src.neuralnet.attention import repeat_kv_heads


@pytest.fixture
def gqa_tensors():
    batch_size = 2
    seq_len = 8
    num_query_heads = 12
    num_kv_heads = 4
    head_dim = 64

    keys = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    n_rep = num_query_heads // num_kv_heads

    return keys, values, n_rep, num_query_heads


def test_repeat_kv_heads_shape(gqa_tensors):
    keys, values, n_rep, num_query_heads = gqa_tensors

    repeated_keys = repeat_kv_heads(keys, n_rep)
    repeated_values = repeat_kv_heads(values, n_rep)

    assert repeated_keys.shape == (
        keys.shape[0],
        num_query_heads,
        keys.shape[2],
        keys.shape[3],
    )

    assert repeated_values.shape == (
        values.shape[0],
        num_query_heads,
        values.shape[2],
        values.shape[3],
    )


def test_repeat_kv_heads_repeats_each_head(gqa_tensors):
    keys, values, n_rep, _ = gqa_tensors

    repeated_keys = repeat_kv_heads(keys, n_rep)
    repeated_values = repeat_kv_heads(values, n_rep)

    for kv_head in range(keys.shape[1]):
        start = kv_head * n_rep
        end = start + n_rep

        expected_keys = keys[:, kv_head : kv_head + 1].expand(
            -1, n_rep, -1, -1
        )
        expected_values = values[:, kv_head : kv_head + 1].expand(
            -1, n_rep, -1, -1
        )

        assert torch.equal(repeated_keys[:, start:end], expected_keys)
        assert torch.equal(repeated_values[:, start:end], expected_values)


def test_repeat_kv_heads_n_rep_one():
    x = torch.randn(2, 4, 8, 64)

    result = repeat_kv_heads(x, 1)

    assert result.shape == x.shape
    assert torch.equal(result, x)


def test_repeat_kv_heads_preserves_input():
    x = torch.randn(2, 4, 8, 64)
    original = x.clone()

    repeat_kv_heads(x, 3)

    assert torch.equal(x, original)


def test_repeat_kv_heads_gqa_ratio():
    num_query_heads = 12
    num_kv_heads = 4

    assert num_query_heads % num_kv_heads == 0

    n_rep = num_query_heads // num_kv_heads

    x = torch.randn(2, num_kv_heads, 8, 64)
    result = repeat_kv_heads(x, n_rep)

    assert result.shape[1] == num_query_heads