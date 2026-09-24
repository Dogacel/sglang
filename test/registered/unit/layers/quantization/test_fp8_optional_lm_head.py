"""Online head quantization is opt-in and must not affect embeddings/checkpoints."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def bare(cls):
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    return obj


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_online_head_only(monkeypatch, enabled, serialized):
    monkeypatch.setenv("SGLANG_FP8_QUANT_LM_HEAD", str(int(enabled)))
    config = Fp8Config(is_checkpoint_fp8_serialized=serialized)
    with patch(
        "sglang.srt.layers.quantization.fp8.Fp8LinearMethod", return_value="fp8"
    ):
        assert config.get_quant_method(bare(ParallelLMHead), "lm_head") == (
            "fp8" if enabled and not serialized else None
        )
        assert (
            config.get_quant_method(bare(VocabParallelEmbedding), "embed_tokens")
            is None
        )


def test_ignored_head_stays_unquantized(monkeypatch):
    monkeypatch.setenv("SGLANG_FP8_QUANT_LM_HEAD", "1")
    config = Fp8Config(ignored_layers=["lm_head"])
    method = config.get_quant_method(bare(ParallelLMHead), "lm_head")
    assert type(method).__name__ == "UnquantizedLinearMethod"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
