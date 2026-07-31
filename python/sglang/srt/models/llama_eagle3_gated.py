"""
Depth-gated LLaMA EAGLE-3 draft model.

Inference-only implementation of TorchSpec's ``LlamaForCausalLMEagle3Gated``
architecture (draft config ``model_type: llama_gated``). The single decoder
layer carries three MLPs: a shared MLP scaled by a per-token sigmoid gate
(``shared_expert_gate``), a ``depth0_mlp`` specialist used for the first
rollout step, and a ``depthn_mlp`` specialist used for deeper rollout steps.

Rollout depth at serving time is derived from the forward mode:
draft-extend (the first rollout step over target-accepted tokens) maps to
depth 0; tree/decode expansion steps map to depth > 0. The distinction is a
capture-time constant for both CUDA graph families.
"""

import json
import os
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.models.llama_eagle3 import (
    LlamaDecoderLayer,
    LlamaForCausalLMEagle3,
    LlamaModel,
)
from sglang.srt.utils import add_prefix


class _GatingProbe:
    """Optional per-token gating probe (env SGLANG_GATED_PROBE_PATH).

    Records, per draft forward call, the shared-expert sigmoid gate value,
    next-token entropy (over the draft vocab), and top-1 probability for
    every token. Intended for eager (non-CUDA-graph) single-batch probing:
    decode-mode records arrive in bursts of exactly num_steps-1 forwards,
    so the rollout step is the position within the burst (reset on each
    draft-extend record, which is rollout depth 0).
    """

    def __init__(self):
        self.path = os.environ.get("SGLANG_GATED_PROBE_PATH", "")
        # Fallback: sglang curates the scheduler-process env; a config file
        # works regardless of process spawn mechanics.
        cfg = os.environ.get(
            "SGLANG_GATED_PROBE_CONFIG", "/home/ubuntu/bench/gating_probe_config.json"
        )
        if not self.path and os.path.exists(cfg):
            try:
                with open(cfg) as f:
                    self.path = json.load(f).get("path", "")
            except Exception:
                pass
        self.enabled = bool(self.path)
        self._fh = None
        self._pending = 0
        self._decode_step = 0

    def record(self, forward_batch, mlp, logits):
        if not self.enabled:
            return
        gate = getattr(mlp, "_probe_gate", None)
        depth = getattr(mlp, "_probe_depth", None)
        if gate is None or depth is None or logits is None:
            return
        if logits.dim() != 2 or logits.shape[0] != gate.shape[0]:
            return
        t = logits.float()
        logp = t - t.logsumexp(dim=-1, keepdim=True)
        p = logp.exp()
        entropy = -(p * logp).sum(-1).float()
        top1 = p.amax(dim=-1).float()
        is_extend = forward_batch.forward_mode.is_draft_extend_v2()
        if is_extend:
            self._decode_step = 0
            step = 0
        else:
            self._decode_step += 1
            step = self._decode_step
        shared_norm = getattr(mlp, "_probe_shared_norm", None)
        spec_norm = getattr(mlp, "_probe_spec_norm", None)
        rec = {
            "mode": "extend" if is_extend else "decode",
            "step": step,
            "gate": gate.squeeze(-1).tolist(),
            "entropy": entropy.tolist(),
            "top1": top1.tolist(),
        }
        if shared_norm is not None and spec_norm is not None:
            rec["shared_norm"] = shared_norm.tolist()
            rec["spec_norm"] = spec_norm.tolist()
        if self._fh is None:
            self._fh = open(self.path, "w")
        self._fh.write(json.dumps(rec) + "\n")
        self._pending += 1
        if self._pending >= 16:
            self._fh.flush()
            self._pending = 0


_probe = _GatingProbe()


class DepthGatedMLP(nn.Module):
    """Depth-selected specialist MLP plus a token-gated shared MLP.

    Matches TorchSpec's DepthGatedMLP: a token-dependent scalar sigmoid gate
    weights the shared MLP, while rollout depth top-k selects the specialist.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.shared_mlp = LlamaMLP(
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config,
            add_prefix("shared_mlp", prefix),
        )
        self.depth0_mlp = LlamaMLP(
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config,
            add_prefix("depth0_mlp", prefix),
        )
        self.depthn_mlp = LlamaMLP(
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config,
            add_prefix("depthn_mlp", prefix),
        )
        self.shared_expert_gate = ReplicatedLinear(
            hidden_size,
            1,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("shared_expert_gate", prefix),
        )

    def forward(self, hidden_states: torch.Tensor, depth: int) -> torch.Tensor:
        shared = self.shared_mlp(hidden_states)
        specialist = (
            self.depth0_mlp(hidden_states)
            if depth == 0
            else self.depthn_mlp(hidden_states)
        )
        gate_logits, _ = self.shared_expert_gate(hidden_states)
        # Sigmoid in fp32 for numerical parity with training.
        shared_gate = torch.sigmoid(gate_logits.float()).to(hidden_states.dtype)
        if _probe.enabled:
            self._probe_gate = shared_gate.float()
            self._probe_depth = depth
            self._probe_shared_norm = shared.float().norm(dim=-1)
            self._probe_spec_norm = specialist.float().norm(dim=-1)
        return specialist + shared_gate * shared


class GatedLlamaDecoderLayer(LlamaDecoderLayer):
    """Llama EAGLE-3 layer whose MLP specializes by rollout depth."""

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config=quant_config, prefix=prefix)
        inter_size = (
            config.intermediate_size_mlp
            if config.model_type == "llama4_text"
            else config.intermediate_size
        )
        self.mlp = DepthGatedMLP(
            config.hidden_size,
            inter_size,
            config.hidden_act,
            quant_config,
            add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Draft-extend is the first rollout step (depth 0); tree expansion
        # steps are deeper (depth >= 1). Both are graph-capture-time constants.
        depth = 0 if forward_batch.forward_mode.is_draft_extend_v2() else 1

        if self.is_input_layer:
            # Input layer consumes target hidden states; no carried residual to fuse.
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
            embeds = self.input_layernorm(embeds)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            # Fuse the previous layer's MLP residual add into hidden_norm.
            hidden_states, residual = self.hidden_norm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        hidden_states = self.mlp(hidden_states, depth)

        return hidden_states, residual


class GatedLlamaModel(LlamaModel):
    """Llama EAGLE-3 draft model with depth-gated decoder layers."""

    decoder_layer_class = GatedLlamaDecoderLayer


class LlamaForCausalLMEagle3Gated(LlamaForCausalLMEagle3):
    """Llama EAGLE-3 drafter with shared/depth-0/depth-N gated MLPs.

    Inherits weight loading: the checkpoint's ``midlayer.*`` prefix maps to
    ``layers.0.*``; ``gate_proj``/``up_proj`` fold into ``gate_up_proj``;
    ``d2t``/``t2d`` drive the draft vocabulary mapping. The checkpoint's
    ``fc_norm: true`` and ``norm_output: true`` flags are honored by the base
    ``LlamaModel`` (per-aux RMSNorm before ``fc`` and post-norm aux hidden
    states). ``fc_norm``/``norm_output`` default to enabled for this
    architecture if the config omits them.
    """

    model_class = GatedLlamaModel

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Architecture defaults (TorchSpec LlamaGatedConfig): enabled unless
        # explicitly disabled in the draft config.
        if getattr(config, "fc_norm", None) is None:
            config.fc_norm = True
        if getattr(config, "norm_output", None) is None:
            config.norm_output = True
        super().__init__(config, quant_config=quant_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        out = super().forward(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if _probe.enabled and not forward_batch.forward_mode.is_idle():
            _probe.record(
                forward_batch,
                self.model.layers[0].mlp,
                getattr(out, "next_token_logits", None),
            )
        return out


EntryClass = [LlamaForCausalLMEagle3Gated]
