import time
import os
from typing import List

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

    @staticmethod
    def _normalize_messages_for_qwen_vl_utils(batched_messages):
        """
        Normalize video payloads before qwen_vl_utils.process_vision_info().

        qwen_vl_utils.fetch_video() requires ele["video"] to be either:
        - a single path string (str), or
        - a list/tuple of frame paths

        However lmms-eval can produce video elements where:
        - ele["video"] is a PathLike
        - ele["video"] is a dict carrying {url|path|video|file, video_start, video_end, ...}
        - the element accidentally also has image/image_url keys that route it to fetch_image
        - ele["video"] is a single-item list of an mp4 path (which would be wrongly
          decoded as a frame sequence by Image.open(mp4))

        This method rewrites the element in place into a form qwen_vl_utils expects.
        """
        VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv")

        def _extract_path(value):
            if isinstance(value, os.PathLike):
                return os.fspath(value)
            if isinstance(value, str):
                return value
            return None

        for msg_list in batched_messages:
            if not isinstance(msg_list, list):
                continue
            for msg in msg_list:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for ele in content:
                    if not isinstance(ele, dict):
                        continue
                    if ele.get("type") != "video" and "video" not in ele:
                        continue

                    v = ele.get("video")
                    if v is None:
                        continue

                    # If video element accidentally also carries image keys, qwen_vl_utils
                    # will route it to fetch_image() first and treat mp4 as an image.
                    ele.pop("image", None)
                    ele.pop("image_url", None)

                    path = _extract_path(v)
                    if path is not None:
                        ele["video"] = path
                        continue

                    if isinstance(v, dict):
                        nested_path = (
                            _extract_path(v.get("video"))
                            or _extract_path(v.get("url"))
                            or _extract_path(v.get("path"))
                            or _extract_path(v.get("file"))
                        )
                        if nested_path is None:
                            continue
                        # Lift metadata (video_start/video_end/fps/...) to ele itself so
                        # qwen_vl_utils backends can read them as kwargs.
                        for k, val in v.items():
                            if k in ("type", "video", "url", "path", "file"):
                                continue
                            ele.setdefault(k, val)
                        ele["video"] = nested_path
                        continue

                    if isinstance(v, (list, tuple)):
                        if len(v) == 1:
                            only_path = _extract_path(v[0])
                            if only_path is not None and only_path.lower().endswith(VIDEO_EXTS):
                                ele["video"] = only_path
                                continue
                        # Otherwise keep as a real frame sequence (list/tuple of paths).
                        continue

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)

            chat_messages: List[ChatMessages] = []
            visuals = []
            videos = []
            for idx, (ids, task_name, split_name) in enumerate(zip(doc_id, task, split)):
                messages = doc_to_messages[idx](self.task_dict[task_name][split_name][ids])
                messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
                chat_message = ChatMessages(**{"messages": messages})
                visual, video, _ = chat_message.extract_media()
                visuals.append(visual)
                videos.append(video)
                chat_messages.append(chat_message)

            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = self._build_video_kwargs()
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            self._normalize_messages_for_qwen_vl_utils(batched_messages)

            texts = self._apply_chat_template(batched_messages)

            try:
                image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                    batched_messages,
                    return_video_kwargs=True,
                    image_patch_size=16,
                    return_video_metadata=True,
                )
                video_kwargs = {**video_kwargs, **video_kwargs_qwen}

                video_metadatas = None
                if video_inputs is not None:
                    video_inputs, video_metadatas = zip(*video_inputs)
                    video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)

                if self.batch_size > 1:
                    inputs = self.processor(
                        text=texts,
                        images=image_inputs,
                        videos=video_inputs,
                        video_metadata=video_metadatas,
                        **video_kwargs,
                        do_resize=False,
                        padding=True,
                        padding_side="left",
                        return_tensors="pt",
                    )
                else:
                    inputs = self.processor(
                        text=texts,
                        images=image_inputs,
                        videos=video_inputs,
                        video_metadata=video_metadatas,
                        **video_kwargs,
                        do_resize=False,
                        return_tensors="pt",
                    )

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)

                start_time = time.time()
                cont = self.model.generate(**inputs, **generate_kwargs)
                end_time = time.time()

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                total_elapsed_time += end_time - start_time
                total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

                for i, (ans, context) in enumerate(zip(answers, texts)):
                    ans = self._strip_thinking(ans)
                    res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)

                    eval_logger.debug(f"Question: {context}")
                    eval_logger.debug(f"Model Response: {ans}")
            except Exception as exc:
                bad_video_eles = []
                for msg_list in batched_messages:
                    if not isinstance(msg_list, list):
                        continue
                    for msg in msg_list:
                        if not isinstance(msg, dict):
                            continue
                        msg_content = msg.get("content")
                        if not isinstance(msg_content, list):
                            continue
                        for ele in msg_content:
                            if isinstance(ele, dict) and ele.get("type") == "video":
                                bad_video_eles.append(ele)
                eval_logger.error(
                    f"qwen3_vl_chat: video/generation failure, aborting. "
                    f"task={task} | doc_id={doc_id} | error={exc!r}"
                )
                for ele in bad_video_eles:
                    summary = {
                        "video": ele.get("video"),
                        "video_start": ele.get("video_start"),
                        "video_end": ele.get("video_end"),
                        "nframes": ele.get("nframes"),
                        "fps": ele.get("fps"),
                        "min_frames": ele.get("min_frames"),
                        "max_frames": ele.get("max_frames"),
                        "min_pixels": ele.get("min_pixels"),
                        "max_pixels": ele.get("max_pixels"),
                        "total_pixels": ele.get("total_pixels"),
                    }
                    eval_logger.error(f"qwen3_vl_chat bad video element: {summary}")
                    eval_logger.error(f"qwen3_vl_chat bad video path: {ele.get('video')}")
                raise
            pbar.update(1)

        res = re_ords.get_original(res)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res
