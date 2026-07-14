import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Union

import numpy as np
from dotenv import load_dotenv
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    decide_next_concurrency,
    extract_text_prefix_from_chat_messages,
    is_rate_limit_error,
    make_prefix_hash,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64
from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
)
from lmms_eval.models.simple.openai import OpenAICompatible as OpenAICompatibleSimple
from lmms_eval.protocol import ChatMessages

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

load_dotenv(verbose=True)


@register_model("openai")
class OpenAICompatible(OpenAICompatibleSimple):
    is_simple = False
    LOCAL_VIDEO_IMAGE_MAX_SIDE = 448
    LOCAL_VIDEO_JPEG_QUALITY = 85

    def _use_local_video_sampling(self) -> bool:
        return os.getenv("LMMS_OPENAI_LOCAL_VIDEO_SAMPLING", "").strip().lower() in {"1", "true", "yes", "on"}

    def _local_video_preset(self) -> str:
        return os.getenv("LMMS_OPENAI_LOCAL_VIDEO_PRESET", "").strip().lower()

    def _use_local_video_linspace(self) -> bool:
        return self._local_video_preset() == "gpt4v" or os.getenv("LMMS_OPENAI_LOCAL_VIDEO_STRATEGY", "").strip().lower() in {"linspace", "uniform"}

    def _resize_local_video_frame(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        longest = max(width, height)
        if longest <= self.LOCAL_VIDEO_IMAGE_MAX_SIDE:
            return image
        scale = float(self.LOCAL_VIDEO_IMAGE_MAX_SIDE) / float(longest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _sample_video_frames_local(self, video_source, max_frames: int) -> List[Image.Image]:
        if VideoReader is None or cpu is None:
            raise ImportError("decord is required for local video sampling")

        if isinstance(video_source, dict):
            video_path = video_source.get("video") or video_source.get("url") or video_source.get("path")
            video_start = video_source.get("video_start")
            video_end = video_source.get("video_end")
        else:
            video_path = video_source
            video_start = None
            video_end = None

        if not isinstance(video_path, str):
            raise ValueError(f"Unsupported video source for local sampling: {video_source}")

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        if total_frames <= 0:
            return []

        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
        if fps <= 0:
            fps = 1.0

        if self._local_video_preset() == "gpt4v":
            target_frames = max(1, int(max_frames))
            frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int).tolist()
            if total_frames - 1 not in frame_indices:
                frame_indices.append(total_frames - 1)
        else:
            start_index = 0
            end_index = total_frames - 1
            if video_start is not None:
                start_index = max(0, min(int(np.ceil(float(video_start) * fps)), total_frames - 1))
            if video_end is not None:
                end_index = max(0, min(int(np.floor(float(video_end) * fps)), total_frames - 1))
            if end_index < start_index:
                end_index = start_index

            visible_length = max(1, end_index - start_index + 1)
            target_frames = max(1, int(max_frames))
            if self._use_local_video_linspace():
                sample_count = min(target_frames, visible_length)
                frame_indices = np.linspace(start_index, end_index, num=sample_count, dtype=int).tolist()
                if len(frame_indices) < target_frames:
                    frame_indices += [frame_indices[-1]] * (target_frames - len(frame_indices))
            else:
                acc_samples = min(target_frames, visible_length)
                intervals = np.linspace(0, visible_length, num=acc_samples + 1).astype(int)
                ranges = [(start, max(start, end - 1)) for start, end in zip(intervals[:-1], intervals[1:])]
                frame_indices = [(start + end) // 2 for start, end in ranges]
                if len(frame_indices) < target_frames:
                    frame_indices += [frame_indices[-1]] * (target_frames - len(frame_indices))
                frame_indices = np.clip(np.asarray(frame_indices, dtype=np.int64) + start_index, 0, total_frames - 1).tolist()

        frames = []
        for idx in frame_indices:
            frame = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
            if self._local_video_preset() != "gpt4v":
                frame = self._resize_local_video_frame(frame)
            frames.append(frame)
        return frames

    def _build_openai_messages_local_video(self, chat_messages: ChatMessages, video_kwargs: dict) -> list[dict]:
        openai_messages = []
        for message in chat_messages.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    if self._local_video_preset() == "gpt4v":
                        encoded = self.encode_image(content.url)
                        data_url = f"data:image/png;base64,{encoded}"
                    else:
                        encoded = encode_image_to_base64(
                            content.url,
                            image_format="JPEG",
                            convert_rgb=True,
                            quality=self.LOCAL_VIDEO_JPEG_QUALITY,
                            copy_if_pil=False,
                        )
                        data_url = f"data:image/jpeg;base64,{encoded}"
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                elif content.type == "video":
                    frames = self._sample_video_frames_local(content.url, video_kwargs.get("max_frames", self.max_frames_num))
                    for frame in frames:
                        if self._local_video_preset() == "gpt4v":
                            encoded = encode_image_to_base64(
                                frame,
                                image_format="PNG",
                                convert_rgb=False,
                                quality=None,
                                copy_if_pil=False,
                            )
                            data_url = f"data:image/png;base64,{encoded}"
                        else:
                            encoded = encode_image_to_base64(
                                frame,
                                image_format="JPEG",
                                convert_rgb=True,
                                quality=self.LOCAL_VIDEO_JPEG_QUALITY,
                                copy_if_pil=False,
                            )
                            data_url = f"data:image/jpeg;base64,{encoded}"
                        openai_message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            }
                        )
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        reordered_requests = list(requests)
        pbar = tqdm(
            total=len(reordered_requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        responses: List[Union[GenerationResult, None]] = [None] * len(reordered_requests)
        total_latency = 0.0
        total_tokens = 0
        current_concurrency = min(
            self.num_concurrent,
            self.adaptive_config.max_concurrency,
        )
        dispatch_order = list(range(len(reordered_requests)))
        if self.prefix_aware_queue:
            prefix_hashes = {}
            for idx in dispatch_order:
                req = reordered_requests[idx]
                prefix_text = req.args[0] if isinstance(req.args[0], str) else ""
                if not prefix_text:
                    _, doc_to_messages, _, doc_id, task, split = req.args
                    chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                    prefix_text = extract_text_prefix_from_chat_messages(chat_messages_raw, self.prefix_hash_chars)
                prefix_hashes[idx] = make_prefix_hash(prefix_text, self.prefix_hash_chars)
            dispatch_order.sort(key=lambda idx: (prefix_hashes[idx], idx))
        cursor = 0
        failed_requests = 0
        rate_limited_requests = 0
        latencies: List[float] = []
        completed_since_adapt = 0
        in_flight = {}
        max_workers = max(
            1,
            self.adaptive_config.max_concurrency if self.adaptive_concurrency else current_concurrency,
        )

        def process_single_request(local_index: int, payload: dict | None):
            if payload is None:
                return "", local_index, False, False, 0.0, 0, 0, 0
            started_at = time.time()
            rate_limited = False
            last_error_msg = "unknown error"
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(**payload)
                    elapsed = time.time() - started_at
                    response_text = response.choices[0].message.content
                    input_tokens = 0
                    output_tokens = 0
                    reasoning_tokens = 0
                    if hasattr(response, "usage") and response.usage:
                        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                        if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                            reasoning_tokens = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                        completion_tokens = output_tokens
                    else:
                        completion_tokens = len(response_text.split())
                        output_tokens = completion_tokens
                    log_usage(
                        model_name=self.model_version,
                        task_name=None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        source="model",
                    )
                    return (
                        response_text,
                        local_index,
                        True,
                        rate_limited,
                        elapsed,
                        completion_tokens,
                        input_tokens,
                        reasoning_tokens,
                    )
                except Exception as exc:
                    error_msg = str(exc)
                    last_error_msg = error_msg
                    rate_limited = rate_limited or is_rate_limit_error(error_msg)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                    else:
                        time.sleep(self.retry_backoff_s)

            elapsed = time.time() - started_at
            error_preview = last_error_msg.replace("\n", " ")[:200]
            failure_content = f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}"
            return failure_content, local_index, False, rate_limited, elapsed, 0, 0, 0

        def maybe_update_concurrency(force: bool = False) -> None:
            nonlocal current_concurrency
            nonlocal failed_requests
            nonlocal rate_limited_requests
            nonlocal latencies
            nonlocal completed_since_adapt

            if not self.adaptive_concurrency:
                return

            sample_threshold = max(4, current_concurrency)
            if not force and completed_since_adapt < sample_threshold:
                return
            if completed_since_adapt <= 0:
                return

            decision = decide_next_concurrency(
                current_concurrency=current_concurrency,
                total_requests=completed_since_adapt,
                failed_requests=failed_requests,
                rate_limited_requests=rate_limited_requests,
                latencies=latencies,
                config=self.adaptive_config,
            )
            if decision.next_concurrency != decision.current_concurrency:
                eval_logger.info(
                    "Adaptive concurrency update: "
                    f"{decision.current_concurrency} -> "
                    f"{decision.next_concurrency} "
                    f"(fail_rate={decision.failure_rate:.3f}, "
                    f"rate_limit_rate={decision.rate_limit_rate:.3f}, "
                    f"p95_latency={decision.p95_latency_s:.3f}s)"
                )
            current_concurrency = decision.next_concurrency
            failed_requests = 0
            rate_limited_requests = 0
            latencies = []
            completed_since_adapt = 0

        def build_payload_for_index(global_index: int) -> dict:
            req = reordered_requests[global_index]
            _, doc_to_messages, gen_kwargs, doc_id, task, split = req.args

            chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
            chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages_raw})
            request_gen_kwargs = dict(gen_kwargs)
            max_new_tokens = min(request_gen_kwargs.get("max_new_tokens", 1024), 4096)
            temperature = request_gen_kwargs.get("temperature", 0)
            top_p = request_gen_kwargs.get("top_p", 1.0)

            if self.video_fps is not None and self.video_fps > 0:
                video_kwargs = {"fps": self.video_fps, "max_frames": self.max_frames_num}
            else:
                # Let qwen_vl_utils choose the final nframes based on the
                # backend that actually decodes the video. This avoids
                # short-video errors where a fixed nframes can exceed the
                # readable frame count.
                video_kwargs = {"max_frames": self.max_frames_num}

            if self._use_local_video_sampling():
                openai_messages = self._build_openai_messages_local_video(chat_messages, video_kwargs)
            else:
                openai_messages = chat_messages.to_openai_messages(video_kwargs=video_kwargs)

            payload = {
                "messages": openai_messages,
                "model": self.model_version,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            if "o1" in self.model_version or "o3" in self.model_version or "o4" in self.model_version:
                payload.pop("temperature")
                payload.pop("top_p")
                payload.pop("max_tokens")
                payload["response_format"] = {"type": "text"}
                payload["max_completion_tokens"] = 5000
            elif "gpt-5" in self.model_version:
                payload.pop("max_tokens")
                payload["max_completion_tokens"] = max_new_tokens

            return payload

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    request_index = dispatch_order[cursor]
                    payload = build_payload_for_index(request_index)
                    if payload is None:
                        responses[request_index] = GenerationResult(text="", token_counts=TokenCounts())
                        pbar.update(1)
                        cursor += 1
                        continue

                    if is_budget_exceeded():
                        responses[request_index] = GenerationResult(text="[LMMS_EVAL_BUDGET_EXCEEDED]", token_counts=TokenCounts())
                        pbar.update(1)
                        cursor += 1
                        continue

                    assert payload is not None
                    future = executor.submit(process_single_request, request_index, payload)
                    in_flight[future] = request_index
                    cursor += 1

                if not in_flight:
                    break

                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    (
                        response_text,
                        local_index,
                        success,
                        rate_limited,
                        elapsed,
                        completion_tokens,
                        input_tokens,
                        reasoning_tokens,
                    ) = future.result()
                    in_flight.pop(future, None)
                    responses[local_index] = GenerationResult(
                        text=response_text,
                        token_counts=TokenCounts(
                            input_tokens=input_tokens,
                            output_tokens=completion_tokens,
                            reasoning_tokens=reasoning_tokens,
                        ),
                    )
                    total_latency += elapsed
                    total_tokens += completion_tokens
                    latencies.append(elapsed)
                    if not success:
                        failed_requests += 1
                    if rate_limited:
                        rate_limited_requests += 1
                    completed_since_adapt += 1
                    totals = get_running_totals()
                    pbar.set_postfix({"tokens": f"{totals['total_tokens']:,}"}, refresh=False)
                    pbar.update(1)
                    maybe_update_concurrency(force=False)

        maybe_update_concurrency(force=True)

        avg_speed = total_tokens / total_latency if total_latency > 0 else 0
        log_metrics(
            total_elapsed_time=total_latency,
            total_gen_tokens=total_tokens,
            avg_speed=avg_speed,
        )

        pbar.close()
        return [response if response is not None else GenerationResult(text="", token_counts=TokenCounts()) for response in responses]
