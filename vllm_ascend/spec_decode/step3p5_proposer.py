# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import lmhead_tp_enable


class AscendStep3p5MTPProposer(Step3p5MTPProposer, AscendSpecDecodeBaseProposer):
    """Step3.5 MTP proposer adapted for Ascend NPU.

    Uses AscendSpecDecodeBaseProposer's ``_propose`` for NPU attention
    metadata management, and overrides ``_run_merged_draft`` to inject
    ``spec_step_idx`` and use ``_sample_draft_tokens_for_step`` for
    per-step MTP draft token sampling.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(
            self, vllm_config, device, pass_hidden_states_to_model=True, runner=runner
        )
        # Step3.5 MTP per-group data structures (from upstream
        # Step3p5MTPProposer).  These are created here explicitly
        # because AscendSpecDecodeBaseProposer.__init__ bypasses
        # Step3p5MTPProposer.__init__.
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._per_group_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        if not hasattr(self, "_enable_probabilistic_draft_probs"):
            self._enable_probabilistic_draft_probs = False

    @property
    def _is_step3p5(self) -> bool:
        return (
            self.method == "mtp"
            and self.speculative_config.draft_model_config is not None
            and self.speculative_config.draft_model_config.hf_config.model_type
            == "step3p5_mtp"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model: nn.Module) -> None:
        if self._is_step3p5:
            hf_config = self.vllm_config.model_config.hf_config
            layer_types = getattr(hf_config, "layer_types", None)
            if layer_types:
                num_mtp = getattr(hf_config, "num_nextn_predict_layers", 1)
                needed = hf_config.num_hidden_layers + num_mtp
                while len(layer_types) < needed:
                    layer_types.append('sliding_attention')
        super().load_model(model)

    def _get_model(self) -> nn.Module:
        """Load the draft model with the correct VllmConfig.

        Using ``self.vllm_config`` directly would cause
        ``Step3p5MTP.__init__`` to read
        ``vllm_config.model_config.hf_config`` — the **target** model's
        HF config — instead of the MTP draft model's HF config.  That
        misses ``num_nextn_predict_layers``, ``model_type``, and other
        MTP-specific fields.

        We use :func:`vllm.config.replace` to swap in the draft
        ``ModelConfig`` while preserving the target's
        ``QuantizationConfig`` (they share the same quantization
        method).  This avoids calling ``get_draft_quant_config()``
        which requires ``hf_overrides`` to be a dict.
        """
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.speculative_config.draft_model_config,
            )
        return model

    def _maybe_share_lm_head(self, model: nn.Module) -> None:
        if self._is_step3p5:
            if (
                self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
                and self.use_cuda_graph
            ):
                self.update_stream = torch.npu.Stream()
                self._runnable = ACLGraphWrapper(
                    self._run_merged_draft,
                    self.vllm_config,
                    runtime_mode=CUDAGraphMode.FULL,
                    use_eagle=self.use_eagle,
                    enable_enpu=self.enable_enpu,
                )
            return
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, model)

    # ------------------------------------------------------------------
    # _propose — inject ``sampling_metadata`` into ``_run_merged_draft``
    # ------------------------------------------------------------------

    def _propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        target_model_batch_desc,
        sampling_metadata,
        mm_embed_inputs=None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output=None,
        num_scheduled_tokens=0,
        num_rejected_tokens_gpu=None,
    ):
        original_runnable = self._runnable

        def _runnable_with_sampling(**model_inputs):
            model_inputs["sampling_metadata"] = sampling_metadata
            return self._run_merged_draft(**model_inputs)

        self._runnable = _runnable_with_sampling
        try:
            return AscendSpecDecodeBaseProposer._propose(
                self,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                target_model_batch_desc=target_model_batch_desc,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                req_scheduled_tokens=req_scheduled_tokens,
                long_seq_metadata=long_seq_metadata,
                num_prefill_reqs=num_prefill_reqs,
                num_decode_reqs=num_decode_reqs,
                scheduler_output=scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        finally:
            self._runnable = original_runnable

    # ------------------------------------------------------------------
    # _run_merged_draft — spec_step_idx + _sample_draft_tokens_for_step
    # ------------------------------------------------------------------

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
        sampling_metadata=None,
    ) -> torch.Tensor:
        # ---- Step 0: first forward ----
        model_input_ids = self.input_ids[:num_input_tokens]
        model_positions = self._get_positions(num_input_tokens)

        model_kwargs = {
            "input_ids": model_input_ids,
            "positions": model_positions,
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            mhs = self.hidden_states[:num_input_tokens]
            mhs, model_positions = self.maybe_pad_and_reduce(mhs, model_positions)
            model_kwargs["hidden_states"] = mhs
            if self.method == "mtp":
                model_kwargs["positions"] = model_positions

        model_kwargs["spec_step_idx"] = 0

        ret = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hs, hs = ret, ret
        else:
            last_hs, hs = ret

        if self.method != "dflash":
            last_hs, model_positions, hs = self.maybe_all_gather_and_unpad(
                last_hs, model_positions, hs
            )

        # ---- Step 0: sample ----
        n_indices = token_indices_to_sample.shape[0]
        if self.pcp_size > 1:
            hs = hs[:num_input_tokens]
            hs = get_pcp_group().all_gather(hs, 0)
            idx = self.runner.pcp_manager.pcp_allgather_restore_idx.gpu
            hs = torch.index_select(hs, 0, idx[: num_input_tokens * self.pcp_size])
            if self.method == "mtp":
                last_hs = hs
            else:
                last_hs = last_hs[:num_input_tokens]
                last_hs = get_pcp_group().all_gather(last_hs, 0)
                last_hs = torch.index_select(
                    last_hs, 0, idx[: num_input_tokens * self.pcp_size]
                )

        if lmhead_tp_enable():
            pad_lm = (
                self.vllm_config.scheduler_config.max_num_seqs
                * self.runner.uniform_decode_query_len
            )
            token_indices_to_sample = nn.functional.pad(
                token_indices_to_sample, (0, pad_lm - n_indices)
            )

        sample_hs = last_hs[token_indices_to_sample]
        draft_ids, draft_probs = self._sample_draft_tokens_for_step(
            sample_hs, sampling_metadata, spec_step_idx=0
        )
        if lmhead_tp_enable() and n_indices < draft_ids.shape[0]:
            draft_ids = draft_ids[:n_indices]
            token_indices_to_sample = token_indices_to_sample[:n_indices]

        if draft_probs is not None:
            self._last_draft_probs = draft_probs.view(
                -1, self.num_speculative_tokens, draft_probs.shape[-1]
            ).contiguous()

        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            return draft_ids.view(-1, self.num_speculative_tokens)

        if self.pcp_size * self.dcp_size > 1 and is_prefill:
            return torch.stack([draft_ids] * self.num_speculative_tokens, dim=1)

        if lmhead_tp_enable() and self.method == "mtp":
            batch_size = draft_ids.shape[0]

        # ---- Prepare for loop ----
        draft_tensor = torch.zeros(
            (self.num_speculative_tokens, *draft_ids.shape),
            dtype=draft_ids.dtype,
            device=self.device,
        )
        draft_tensor[0] = draft_ids
        positions = (
            self.mrope_positions[:, token_indices_to_sample]
            if self.uses_mrope
            else self.positions[token_indices_to_sample]
        )
        hs = hs[token_indices_to_sample]
        token_indices_to_sample = self.arange[:batch_size]

        input_bs = (
            num_input_tokens
            if (self.method == "mtp" or self.use_cuda_graph)
            else batch_size
        )

        fctx = get_forward_context()
        _EXTRA_CTX.num_tokens = input_bs
        _EXTRA_CTX.num_accept_tokens = batch_size

        probs_list = [draft_probs] if draft_probs is not None else None

        # ---- Loop ----
        for draft_step in range(self.num_speculative_tokens - 1):
            spec_step_idx = draft_step + 1

            fctx = get_forward_context()
            if fctx is not None:
                fctx.moe_layer_index = 0

            in_ids = draft_tensor[draft_step]
            positions = positions + 1

            if self.uses_mrope:
                exc = positions[0] >= self.vllm_config.model_config.max_model_len
                clamped_pos = torch.where(
                    exc.unsqueeze(0), torch.zeros_like(positions), positions
                )
            else:
                exc = positions >= self.vllm_config.model_config.max_model_len
                clamped_pos = torch.where(exc, 0, positions)

            self.input_ids[:batch_size] = in_ids
            self._set_positions(batch_size, clamped_pos)
            self.hidden_states[:batch_size] = hs.view(batch_size, -1)
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(in_ids)

            embeds_b = self.inputs_embeds[:input_bs] if self.supports_mm_inputs else None

            m_in_ids = self.input_ids[:input_bs]
            m_pos = self._get_positions(input_bs)
            m_hs = self.hidden_states[:input_bs]
            m_hs, m_pos = self.maybe_pad_and_reduce(m_hs, m_pos)

            fctx.attn_metadata = (
                multi_steps_attn_metadata[draft_step + 1]
                if multi_steps_attn_metadata
                else None
            )

            m_kwargs = {
                "input_ids": m_in_ids,
                "positions": m_pos,
                "inputs_embeds": embeds_b,
            }
            if self.pass_hidden_states_to_model:
                m_kwargs["hidden_states"] = m_hs
            m_kwargs["spec_step_idx"] = spec_step_idx

            ret = self.model(**m_kwargs)
            if not self.model_returns_tuple():
                last_hs, hs = ret, ret
            else:
                last_hs, hs = ret

            last_hs, m_pos, hs = self.maybe_all_gather_and_unpad(last_hs, m_pos, hs)

            n_indices = token_indices_to_sample.shape[0]
            if lmhead_tp_enable():
                pad_lm = (
                    self.vllm_config.scheduler_config.max_num_seqs
                    * self.runner.uniform_decode_query_len
                )
                token_indices_to_sample = nn.functional.pad(
                    token_indices_to_sample, (0, pad_lm - n_indices)
                )

            sample_hs = last_hs[token_indices_to_sample]
            draft_ids, draft_probs = self._sample_draft_tokens_for_step(
                sample_hs, sampling_metadata, spec_step_idx=spec_step_idx
            )
            if lmhead_tp_enable() and n_indices < draft_ids.shape[0]:
                draft_ids = draft_ids[:n_indices]
                token_indices_to_sample = token_indices_to_sample[:n_indices]

            if draft_probs is not None:
                assert probs_list is not None
                probs_list.append(draft_probs)

            hs = hs[:batch_size]
            draft_tensor[draft_step + 1] = draft_ids

        draft_ids = draft_tensor.swapaxes(0, 1)
        if probs_list is not None:
            self._last_draft_probs = torch.stack(probs_list, dim=1).contiguous()
        return draft_ids