"""Reject unsupported quantization layouts before fusing draft KV projections."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "unsupported",
    ["disabled", "marlin", "block", "static", "scale", "layout", "dtype", "none"],
)
def test_fused_kv_rejects_unsupported_fp8_layouts(monkeypatch, unsupported):
    from types import SimpleNamespace

    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.speculative.dflash_utils import can_dflash_fuse_fp8_qkv

    monkeypatch.setenv("SGLANG_DFLASH_FP8_FUSED_KV", "1")
    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.block_quant = False
    method.use_marlin = False
    method.cutlass_fp8_supported = True
    layer = SimpleNamespace(
        quant_method=method,
        weight=torch.empty(32, 16, dtype=torch.float8_e4m3fn).t(),
        weight_scale=torch.ones(1, 32),
        input_scale=None,
    )
    assert can_dflash_fuse_fp8_qkv(layer)
    if unsupported == "disabled":
        monkeypatch.setenv("SGLANG_DFLASH_FP8_FUSED_KV", "0")
    elif unsupported == "marlin":
        method.use_marlin = True
    elif unsupported == "block":
        method.block_quant = True
    elif unsupported == "static":
        layer.input_scale = torch.tensor(1.0)
    elif unsupported == "scale":
        layer.weight_scale = torch.ones(1)
    elif unsupported == "layout":
        layer.weight = layer.weight.contiguous()
    elif unsupported == "dtype":
        layer.weight = layer.weight.bfloat16()
    elif unsupported == "none":
        layer.quant_method = None
    assert not can_dflash_fuse_fp8_qkv(layer)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
