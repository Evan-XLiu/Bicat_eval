"""Molmo2-ER backbone for Praxis's autoregressive BTT policy.

The local Praxis checkout has the BAR decoder and Molmo backbone helpers, but
does not register their combination used by the exported ``bar_btt_molmo``
checkpoint. Keep this adapter alongside the HTTP bridge so the training repo
does not need to be edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from torch import Tensor

from praxis.policies.backbones.molmo.prefix import MolmoPrefixAdapter
from praxis.policies.backbones.molmo.vlm import (
    MOLMO2_ER_MODEL_NAME,
    configure_molmo_vision_checkpointing,
    ensure_molmo_vision_eval,
    freeze_molmo_vision,
    load_molmo_vlm,
    run_molmo_processor,
)
from praxis.policies.bar.core.vlm.base import VlmActionTokenPolicyBase
from praxis.policies.bar.core.vlm.config import VlmActionTokenPolicyConfig
from praxis.policies.core.action_tokens import action_tokenizer_config_factory
from praxis.policies.core.observations import stack_observation_images
from praxis.policies.core.vlm.dtypes import parse_vlm_dtype
from praxis.policies.core.vlm.hf_chatprompt import render_chat_texts
from praxis.policies.core.vlm.prefix import VlmInputBatch, VlmPrefixLogitsAdapter


@dataclass
class BTTMolmoPolicyConfig(VlmActionTokenPolicyConfig):
    name: str = "bar_btt_molmo"
    vlm_model_name: str = MOLMO2_ER_MODEL_NAME
    gradient_checkpointing: bool = True
    lora_target_modules: tuple[str, ...] = (
        "att_proj", "attn_out", "ff_proj", "ff_out"
    )
    lora_modules_to_save: tuple[str, ...] = ("lm_head", "wte")
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("btt")
    )


class BTTMolmoPolicy(VlmActionTokenPolicyBase):
    CONFIG_CLASS = BTTMolmoPolicyConfig
    ACTION_TOKEN_OWNER_NAME: ClassVar[str] = "Molmo"
    _PREFIX_ATTENTION: ClassVar[Literal["causal"]] = "causal"
    _PREFILL_EXTRA_KEYS: ClassVar[tuple[str, ...]] = (
        "pixel_values", "image_token_pooling", "image_grids",
        "image_num_crops", "token_type_ids",
    )
    vlm: Any
    processor: Any

    def __init__(self, config: BTTMolmoPolicyConfig | dict[str, Any], **kwargs) -> None:
        super().__init__(config, **kwargs)
        configure_molmo_vision_checkpointing(
            self.vlm, bool(self.config.gradient_checkpointing)
        )

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        self.vlm, self.processor = load_molmo_vlm(model_name, dtype_str)

    def _freeze_vision_encoder_modules(self) -> None:
        freeze_molmo_vision(self.vlm)

    def _ensure_frozen_vision_eval(self) -> None:
        ensure_molmo_vision_eval(self.vlm)

    def _build_vlm_prefix_adapter(self) -> VlmPrefixLogitsAdapter:
        return MolmoPrefixAdapter(self.vlm)

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        action_token_strings: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        if self.io is None:
            raise RuntimeError("Policy IO must be set before building Molmo inputs.")
        image_batch = stack_observation_images(
            obs, image_features=self.io.image_features,
            as_uint8=True, channels_last=True,
        )
        bsz = int(image_batch.shape[0])
        num_images = int(image_batch.shape[1])
        if instructions is None:
            instructions = [self.config.default_instruction] * bsz
        elif len(instructions) != bsz:
            raise ValueError("Instructions batch length mismatch")
        state_prefix = self._prepare_state_prefix(obs, batch_size=bsz)
        texts = render_chat_texts(
            processor=self.processor,
            num_images=num_images,
            state_texts=state_prefix.texts,
            instructions=instructions,
            action_token_strings=action_token_strings,
            bsz=bsz,
            include_system=False,
        )
        batch_input = run_molmo_processor(
            processor=self.processor,
            text=texts,
            image_batch=image_batch,
            channels_last=True,
            padding=True,
            return_tensors="pt",
        )
        batch_input = batch_input.to(self.vlm.device)
        dtype = parse_vlm_dtype(self.config.vlm_dtype)
        vlm_inputs = {
            key: value.to(dtype=dtype)
            if isinstance(value, Tensor) and value.is_floating_point()
            else value
            for key, value in dict(batch_input).items()
        }
        return self._add_continuous_state_prefix_embeddings(vlm_inputs, state_prefix)
