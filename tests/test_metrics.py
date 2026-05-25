from chimera.metrics import exact_choice_accuracy, f1_score, mean_f1, rouge_l, mean_rouge_l


def test_choice_accuracy_basic():
    # Model emits the right letter prefix -> 100%
    assert exact_choice_accuracy(["A. luster", "B"], ["A", "B"]) == 100.0


def test_choice_accuracy_wrong():
    assert exact_choice_accuracy(["C", "A"], ["A", "B"]) == 0.0


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
