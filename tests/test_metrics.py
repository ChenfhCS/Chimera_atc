from chimera.metrics import (
    exact_choice_accuracy,
    extract_choice_letter,
    f1_score,
    mean_f1,
    mean_rouge_l,
    rouge_l,
)


def test_choice_accuracy_basic():
    # Model emits the right letter prefix -> 100%
    assert exact_choice_accuracy(["A. luster", "B"], ["A", "B"]) == 100.0


def test_choice_accuracy_wrong():
    assert exact_choice_accuracy(["C", "A"], ["A", "B"]) == 0.0


def test_extract_letter_bare():
    assert extract_choice_letter("A") == "A"
    assert extract_choice_letter("  B  ") == "B"
    assert extract_choice_letter("C.") == "C"
    assert extract_choice_letter("D)") == "D"


def test_extract_letter_instruct_preamble():
    # The output formats we see from Llama-3 / Nemotron-H / Jamba instruct.
    assert extract_choice_letter("The answer is A.") == "A"
    assert extract_choice_letter("The answer is **B)**") == "B"
    assert extract_choice_letter("Of course! The correct answer is C.") == "C"
    assert extract_choice_letter("Answer: D") == "D"
    assert extract_choice_letter("\n\nA) lava\n") == "A"
    assert extract_choice_letter("(B)") == "B"
    assert extract_choice_letter("Option B is correct because...") == "B"
    assert extract_choice_letter("I think the answer is **A)**.") == "A"


def test_extract_letter_no_match():
    assert extract_choice_letter("") == ""
    assert extract_choice_letter("\n\n123") == ""
    # No A-D letter present (5-choice questions cut off here)
    assert extract_choice_letter("ZZZ") == ""


def test_choice_accuracy_instruct_outputs():
    preds = [
        "The answer is A.",
        "**B)**",
        "I'd say (C).",
        "Of course! The answer is D.",
    ]
    labels = ["A", "B", "C", "D"]
    assert exact_choice_accuracy(preds, labels) == 100.0


def test_f1_score_identical():
    assert f1_score("a b c", "a b c") == 1.0


def test_f1_score_partial():
    s = f1_score("the quick brown fox", "a quick red fox")
    assert 0.0 < s < 1.0


def test_mean_f1():
    s = mean_f1(["a b c"], ["a b c"])
    assert s == 100.0


def test_rouge_l_identical():
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_mean_rouge_l_empty():
    assert mean_rouge_l(["a"], ["b"]) == 0.0
