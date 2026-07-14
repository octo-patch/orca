import os
from typing import Optional

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_5 import Qwen3_5

PREFIX = "qwen_vl_interface.model."

# Extra EOS id used by the Orca checkpoint generation recipe.
EXTRA_EOS_TOKEN_ID = 248044


@register_model("orca")
class Orca(Qwen3_5):

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_load_method: str = "auto",
        checkpoint_key: Optional[str] = None,
        strict_load: bool = False,
        skip_ckpt: bool = False,
        enable_thinking: bool = False,
        answer_format_prompt: str = "\nOnly give the best option. Do not provide any explanation.",
        **kwargs,
    ):
        self.answer_format_prompt = answer_format_prompt.replace("\\n", "\n") if answer_format_prompt else ""
        super().__init__(enable_thinking=enable_thinking, **kwargs)
        if isinstance(skip_ckpt, str):
            skip_ckpt = skip_ckpt.lower() in ("1", "true", "yes", "y")
        if skip_ckpt:
            return

        if checkpoint_load_method not in ("auto", "pt", "direct"):
            raise ValueError(
                f"checkpoint_load_method must be 'auto', 'pt', or 'direct', got: {checkpoint_load_method}"
            )
        if checkpoint_load_method == "auto":
            checkpoint_load_method = "direct" if checkpoint_path and os.path.isdir(checkpoint_path) else "pt"
        if checkpoint_load_method == "direct":
            self._load_direct_policy_checkpoint(checkpoint_path)
        else:
            self._load_checkpoint_if_provided(
                checkpoint_path=checkpoint_path,
                checkpoint_key=checkpoint_key,
                strict_load=strict_load,
            )

    def _load_direct_policy_checkpoint(self, checkpoint_path: Optional[str]) -> None:
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required when checkpoint_load_method='direct'")

        print(f"[orca] loading direct VLM weights from checkpoint directory: {checkpoint_path}")
        self._load_direct_vlm_weights(checkpoint_path)
        vlm_config_dir = os.path.join(checkpoint_path, "vlm_config")
        if os.path.isdir(vlm_config_dir):
            self.processor = AutoProcessor.from_pretrained(vlm_config_dir)
            self._tokenizer = self.processor.tokenizer
        self._config = self.model.config

    def _load_direct_vlm_weights(self, checkpoint_path: str) -> None:
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.isfile(safetensors_path):
            raise FileNotFoundError(f"Direct checkpoint is missing model.safetensors: {safetensors_path}")

        raw_state_dict = load_file(safetensors_path, device="cpu")
        prefix = "vlm.model."
        has_direct_prefix = any(key.startswith(prefix) for key in raw_state_dict.keys())
        state_dict = {}
        for key, value in raw_state_dict.items():
            if has_direct_prefix and not key.startswith(prefix):
                continue
            if has_direct_prefix:
                key = key[len(prefix) :]
            state_dict[key[len(PREFIX) :] if key.startswith(PREFIX) else key] = value

        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing = len(incompatible.missing_keys)
        unexpected = len(incompatible.unexpected_keys)
        if missing or unexpected:
            print(
                f"[orca] direct VLM weights loaded with "
                f"{missing} missing keys and {unexpected} unexpected keys."
            )

    def _preprocess_chunk(self, chunk):
        if self.answer_format_prompt:
            chunk = [
                (
                    (ctx.rstrip() + self.answer_format_prompt if isinstance(ctx, str) else ctx),
                    gk,
                    dtv,
                    did,
                    t,
                    s,
                )
                for (ctx, gk, dtv, did, t, s) in chunk
            ]
        return super()._preprocess_chunk(chunk)

    def _build_generate_kwargs(self, gen_kwargs):
        generate_kwargs = super()._build_generate_kwargs(gen_kwargs)

        _tok = self.processor.tokenizer
        _eos_cfg = getattr(self.model.generation_config, "eos_token_id", None)

        eos_set: set = set()
        if _tok.eos_token_id is not None:
            eos_set.add(int(_tok.eos_token_id))
        if isinstance(_eos_cfg, (list, tuple)):
            for x in _eos_cfg:
                if x is not None:
                    eos_set.add(int(x))
        elif _eos_cfg is not None:
            eos_set.add(int(_eos_cfg))
        eos_set.add(EXTRA_EOS_TOKEN_ID)

        eos_ids = sorted(eos_set)
        generate_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        generate_kwargs["pad_token_id"] = _tok.pad_token_id
        return generate_kwargs

    def _load_checkpoint_if_provided(
        self,
        checkpoint_path: Optional[str],
        checkpoint_key: Optional[str],
        strict_load: bool,
    ) -> None:
        if not checkpoint_path:
            return

        if checkpoint_key:
            try:
                blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(checkpoint_path, map_location="cpu")
            if not isinstance(blob, dict) or checkpoint_key not in blob:
                available = ", ".join(sorted(blob.keys())) if isinstance(blob, dict) else type(blob).__name__
                raise KeyError(f"checkpoint_key='{checkpoint_key}' not found in checkpoint. Available: {available}")
            state_dict = blob[checkpoint_key]
        else:
            state_dict = self._load_starvlm_qwen_state_dict(checkpoint_path)

        if isinstance(state_dict, dict):
            state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        else:
            raise TypeError(f"Loaded checkpoint payload is not a state_dict mapping: {type(state_dict)}")

        incompatible = self.model.load_state_dict(state_dict, strict=strict_load)
        if not strict_load:
            missing = len(incompatible.missing_keys)
            unexpected = len(incompatible.unexpected_keys)
            if missing or unexpected:
                print(f"[orca] checkpoint loaded with " f"{missing} missing keys and {unexpected} unexpected keys " f"(strict_load={strict_load}).")

    def _load_starvlm_qwen_state_dict(self, ckpt_path: str) -> dict:
        try:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(ckpt_path, map_location="cpu")
        raw = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        if not isinstance(raw, dict):
            raise TypeError(f"Loaded checkpoint payload is not a state_dict mapping: {type(raw)}")
        out = {}
        for k, v in raw.items():
            out[k[len(PREFIX) :] if k.startswith(PREFIX) else k] = v
        return out
