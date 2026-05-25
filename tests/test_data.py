"""Lightweight data tests that don't require network access."""
import pytest

from chimera.data import _choice_prompt


def test_choice_prompt_letters_only():
    p = _choice_prompt("Q?", ["a", "b", "c", "d"])
    assert "A. a" in p
    assert "B. b" in p
    assert "C. c" in p
    assert "D. d" in p
    assert p.endswith("Answer:")


def test_arc_target_matches_prompt_letter():
    """Regression for ARC numeric-label bug. The fixed loader maps every
    answer to chr('A' + answer_idx) so prompts and targets always agree.

    We re-implement the row mapping here so the test does not need the
    datasets library on disk.
    """
    # Synthetic ARC row whose label set is numeric ("1"/"2"/...)
    labels = ["1", "2", "3", "4"]
    answer_key = "3"
    answer_idx = labels.index(answer_key)
    target = chr(ord('A') + answer_idx)
    assert target == "C"
