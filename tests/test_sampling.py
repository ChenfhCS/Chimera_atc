"""Tests for the length-stratified subsampler in chimera/data.py."""
import random

from chimera.data import EvalExample, _stratified_sample_by_length, _subsample


def _rows(lengths):
    return [EvalExample(dataset="X", prompt="x" * L, target="A") for L in lengths]


def test_stratified_sample_balances_length_buckets():
    rng = random.Random(0)
    # 100 prompts with lengths 1..100; stratified pick of 8 should hit short,
    # medium-short, medium-long, long buckets roughly evenly.
    rows = _rows(list(range(1, 101)))
    out = _stratified_sample_by_length(rows, n=8, rng=rng, buckets=4)
    assert len(out) == 8
    lens = sorted(len(r.prompt) for r in out)
    # At least one element from each quartile of the input range.
    assert any(l <= 25 for l in lens)
    assert any(26 <= l <= 50 for l in lens)
    assert any(51 <= l <= 75 for l in lens)
    assert any(l >= 76 for l in lens)


def test_subsample_head_when_no_seed():
    rows = _rows([10, 20, 30, 40, 50])
    out = _subsample(rows, max_samples=3, sample_seed=None, stratify="length")
    assert [len(r.prompt) for r in out] == [10, 20, 30]


def test_subsample_random_is_reproducible():
    rows = _rows(list(range(100)))
    a = _subsample(rows, max_samples=10, sample_seed=7, stratify="random")
    b = _subsample(rows, max_samples=10, sample_seed=7, stratify="random")
    assert [len(r.prompt) for r in a] == [len(r.prompt) for r in b]
    # Same seed produces same multiset; deterministic.


def test_subsample_returns_all_when_smaller():
    rows = _rows([1, 2, 3])
    out = _subsample(rows, max_samples=100, sample_seed=1, stratify="length")
    assert len(out) == 3


def test_subsample_no_cap():
    rows = _rows([1, 2, 3])
    out = _subsample(rows, max_samples=None, sample_seed=1, stratify="length")
    assert out is rows
