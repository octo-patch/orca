from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v")


def _as_bool(value: Union[bool, str]) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _as_dtype(value: Optional[Union[str, torch.dtype]]) -> Optional[Union[str, torch.dtype]]:
    if value is None or isinstance(value, torch.dtype):
        return value
    lowered = value.strip().lower()
    if lowered in {"", "none"}:
        return None
    if lowered == "auto":
        return "auto"
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if lowered not in aliases:
        raise ValueError(f"Unsupported dtype {value!r}; use auto, bfloat16, float16, or float32")
    return aliases[lowered]


def _extract_video_path(visuals: Any) -> Optional[str]:
    if visuals is None:
        return None
    if isinstance(visuals, (str, os.PathLike)):
        return os.fspath(visuals)
    if isinstance(visuals, dict):
        for key in ("video", "path", "video_path"):
            value = visuals.get(key)
            if isinstance(value, (str, os.PathLike)):
                return os.fspath(value)
    if isinstance(visuals, (list, tuple)):
        for item in visuals:
            path = _extract_video_path(item)
            if path:
                return path
    return None


def _uniform_indices(total: int, max_frames: int) -> list[int]:
    if total <= 0:
        return []
    sample_count = min(max_frames, total)
    if sample_count <= 0:
        return []
    return np.linspace(0, total - 1, sample_count, dtype=int).tolist()


@register_model("smolvlm2")
class SmolVLM2(lmms):
    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 32,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str = "your_model_path",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        dtype: Optional[str] = "bfloat16",
        attn_implementation: Optional[str] = None,
        max_num_frames: int = 32,
        do_image_splitting: Union[bool, str] = False,
        image_longest_edge: Optional[int] = 384,
        system_prompt: Optional[str] = None,
        use_cache: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        if int(batch_size) != 1:
            raise ValueError("SmolVLM2 wrapper currently supports batch_size=1 for video generation.")

        valid_attn_implementations = {None, "flash_attention_2", "sdpa", "eager"}
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {sorted(x for x in valid_attn_implementations if x) + [None]}, got {attn_implementation}")

        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            self.device_map = f"cuda:{self.accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        model_kwargs = {"dtype": _as_dtype(dtype)}
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if attn_implementation:
            model_kwargs["_attn_implementation"] = attn_implementation
        model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        if not self.device_map:
            self._model.to(self._device)

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer
        self._config = self.model.config
        self._max_length = getattr(self._config, "max_position_embeddings", 16384)
        self.batch_size_per_gpu = 1
        self.max_num_frames = int(max_num_frames)
        self.do_image_splitting = _as_bool(do_image_splitting)
        self.image_longest_edge = int(image_longest_edge) if image_longest_edge is not None else None
        self.system_prompt = self._resolve_system_prompt(system_prompt) if system_prompt else None
        self.use_cache = _as_bool(use_cache)

        if self.accelerator.num_processes > 1:
            assert self.accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only FSDP and MULTI_GPU are supported."
            if self.accelerator.distributed_type == DistributedType.FSDP:
                self._model = self.accelerator.prepare(self.model)
            else:
                self._model = self.accelerator.prepare_model(self.model, evaluation_mode=True)
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(
            f"Loaded SmolVLM2 from {pretrained}; max_num_frames={self.max_num_frames}, "
            f"do_image_splitting={self.do_image_splitting}, image_longest_edge={self.image_longest_edge}"
        )

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for SmolVLM2")

    def _load_video_frames_pyav(self, video_path: str) -> list[Image.Image]:
        import av

        decoded = []
        with av.open(video_path) as container:
            for frame in container.decode(video=0):
                decoded.append(frame.to_image().convert("RGB"))
        indices = _uniform_indices(len(decoded), self.max_num_frames)
        return [decoded[idx] for idx in indices]

    def _load_video_frames_decord(self, video_path: str) -> list[Image.Image]:
        from decord import VideoReader, cpu

        reader = VideoReader(video_path, ctx=cpu(0))
        indices = _uniform_indices(len(reader), self.max_num_frames)
        return [Image.fromarray(reader[idx].asnumpy()).convert("RGB") for idx in indices]

    def _load_video_frames(self, video_path: str) -> list[Image.Image]:
        try:
            return self._load_video_frames_pyav(video_path)
        except Exception as pyav_exc:
            eval_logger.warning(f"PyAV failed to decode {video_path}: {pyav_exc}; falling back to decord")
            try:
                return self._load_video_frames_decord(video_path)
            except Exception as decord_exc:
                raise RuntimeError(f"Failed to decode video {video_path} with PyAV and decord") from decord_exc

    def _build_messages(self, context: str, frames: list[Image.Image]) -> list[dict[str, Any]]:
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": context})
        messages = [{"role": "user", "content": content}]
        if self.system_prompt:
            messages = self._apply_system_prompt(messages, self.system_prompt)
        return messages

    def _processor_kwargs(self) -> dict[str, Any]:
        images_kwargs: dict[str, Any] = {
            "do_image_splitting": self.do_image_splitting,
        }
        if self.image_longest_edge is not None:
            images_kwargs["size"] = {"longest_edge": self.image_longest_edge}
        return {"images_kwargs": images_kwargs}

    def _prepare_inputs(self, context: str, video_path: str):
        frames = self._load_video_frames(video_path)
        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")

        messages = self._build_messages(context, frames)
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=self._processor_kwargs(),
        )

    def _build_generate_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        if current.get("temperature", 0) and current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)

        generate_kwargs = {
            "max_new_tokens": current["max_new_tokens"],
            "do_sample": current["do_sample"],
            "use_cache": self.use_cache,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            value = current.get(key)
            if value is not None:
                generate_kwargs[key] = value
        return generate_kwargs

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        results: list[GenerationResult] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            video_path = _extract_video_path(visuals)
            if not video_path:
                raise ValueError(f"SmolVLM2 expects a video path visual for task={task}, doc_id={doc_id}; got {type(visuals)}")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if not video_path.lower().endswith(VIDEO_EXTENSIONS):
                eval_logger.warning(f"Visual path does not look like a video extension: {video_path}")

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be str or list, got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            inputs = self._prepare_inputs(context, video_path)
            input_tokens = int(inputs["input_ids"].shape[-1])
            if input_tokens > self.max_length:
                raise RuntimeError(
                    f"SmolVLM2 input has {input_tokens} tokens, exceeding max_length={self.max_length}. "
                    "Reduce max_num_frames or disable image splitting."
                )

            target_device = next(self.model.parameters()).device if self.device_map else self.device
            inputs = inputs.to(target_device)
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **self._build_generate_kwargs(gen_kwargs))

            generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
            answer = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            for term in until:
                if term:
                    answer = answer.split(term)[0]
            answer = answer.strip()

            output_tokens = int(generated_ids_trimmed.shape[-1])
            result = GenerationResult(text=answer, token_counts=TokenCounts(input_tokens=input_tokens, output_tokens=output_tokens))
            results.append(result)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
            pbar.update(1)

        pbar.close()
        return results

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for SmolVLM2")
