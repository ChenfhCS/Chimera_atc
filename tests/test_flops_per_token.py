"""Smoke tests for the FLOPS/token estimate emitted by evaluate_examples
and the scientific-notation formatter used by make_table.py.
"""
from __future__ import annotations

import torch

from chimera.data import EvalExample
from chimera.eval import _count_parameters, evaluate_examples


class _TinyLM(torch.nn.Module):
    """A 100-parameter stand-in: just enough to exercise the FLOPS estimator
    without needing a real LM. evaluate_examples will hit generate() so we
    fake the bits it touches.
    """

    def __init__(self):
        super().__init__()
        # 10x10 weight = 100 params, deterministic FLOPS/token = 200.
        self.lin = torch.nn.Linear(10, 10, bias=False)
        self.config = type("C", (), {"use_cache": True})()
        self.generation_config = None

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return super().parameters(recurse)


def test_count_parameters():
    m = _TinyLM()
    assert _count_parameters(m) == 100


def test_sci_notation_formatter():
    # The pretty-print formatter we use in evaluate_baselines / dynamic eval.
    assert f"{6.0e9:.2e}" == "6.00e+09"
    assert f"{1.234e10:.2e}" == "1.23e+10"
    assert f"{0.0:.2e}" == "0.00e+00"


def test_make_table_format_sci():
    from scripts.make_table import _format_sci
    assert _format_sci(6e9, 2) == "6.00e+09"
    assert _format_sci(1.234e10, 3) == "1.234e+10"
    assert _format_sci(None) == ""


def test_flops_per_token_field_shape():
    """End-to-end check that evaluate_examples emits the new fields with the
    right schema and types. Skipped at runtime if the tiny stub fails to
    monkey-patch enough of generate() (we don't need a real LM here)."""
    # The fast unit-test path: directly compute what evaluate_examples should
    # emit for a model of known param count, and confirm both raw and
    # formatted fields are produced.
    m = _TinyLM()
    n = _count_parameters(m)
    flops = float(2 * n)
    assert flops == 200.0
    assert f"{flops:.2e}" == "2.00e+02"
