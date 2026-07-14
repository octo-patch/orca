"""
Multi-GPU Orca inference on SWITCH-Basic style datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

DEFAULT_DATA_PATH = "your_data_path"
DEFAULT_OUTPUT_PATH = "results/Orca"
DEFAULT_MODEL_PATH = "your_qwen_path"
DEFAULT_CKPT = "your_checkpoint_path"
DEFAULT_DIRECT_CKPT = (
    "your_checkpoint_path"
)
ANSWER_SUFFIX = "Please directly give the best option."
PREFIX = "qwen_vl_interface.model."

TaskUnit = Tuple[str, str, str, str, List[Dict[str, Any]]]


def load_starvlm_qwen_state_dict(ckpt_path: str) -> dict:
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    raw = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    out = {}
    for k, v in raw.items():
        out[k[len(PREFIX):] if k.startswith(PREFIX) else k] = v
    return out


def load_starvlm_qwen_state_dict_from_dir(ckpt_dir: str) -> dict:
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.isfile(safetensors_path):
        raise FileNotFoundError(f"Checkpoint directory is missing model.safetensors: {safetensors_path}")
    raw = load_file(safetensors_path, device="cpu")
    out = {}
    direct_prefix = "vlm.model."
    has_direct_prefix = any(key.startswith(direct_prefix) for key in raw.keys())
    for key, value in raw.items():
        if has_direct_prefix and not key.startswith(direct_prefix):
            continue
        if has_direct_prefix:
            key = key[len(direct_prefix):]
        out[key[len(PREFIX):] if key.startswith(PREFIX) else key] = value
    return out


def resolve_processor_source(model_path: str, ckpt_path: str, skip_ckpt: bool) -> str:
    if skip_ckpt or not ckpt_path or not os.path.isdir(ckpt_path):
        return model_path
    vlm_config = os.path.join(ckpt_path, "vlm_config")
    if os.path.isdir(vlm_config):
        return vlm_config
    if (
        os.path.isfile(os.path.join(ckpt_path, "processor_config.json"))
        or os.path.isfile(os.path.join(ckpt_path, "tokenizer_config.json"))
    ):
        return ckpt_path
    return model_path


def load_model_and_processor(
    model_path: str,
    ckpt_path: str,
    load_method: str,
    skip_ckpt: bool,
    device: torch.device,
) -> Tuple[Any, AutoProcessor]:
    print(f"Loading base model: {model_path}")
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},
    }
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    if skip_ckpt:
        print("SKIP_CKPT enabled; using base model weights.")
    else:
        if load_method == "auto":
            use_dir = os.path.isdir(ckpt_path)
        elif load_method == "direct":
            use_dir = True
        else:
            use_dir = False

        if use_dir:
            print(f"Loading VLM state dict from directory: {ckpt_path}")
            state_dict = load_starvlm_qwen_state_dict_from_dir(ckpt_path)
        else:
            print(f"Loading pt state dict: {ckpt_path}")
            state_dict = load_starvlm_qwen_state_dict(ckpt_path)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[load_state_dict] strict=False: {msg}", flush=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(resolve_processor_source(model_path, ckpt_path, skip_ckpt))
    return model, processor


def discover_tasks(data_root: str) -> List[Tuple[str, str, str]]:
    found: List[Tuple[str, str, str]] = []
    for dirpath, _, filenames in os.walk(data_root):
        if "vqa.json" not in filenames:
            continue
        rel = os.path.relpath(dirpath, data_root)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            continue
        dim, mod = parts[0], parts[1]
        found.append((dim, mod, os.path.join(dirpath, "vqa.json")))
    return sorted(found, key=lambda x: (x[0], x[1]))


def query_image_path(item: Dict[str, Any]) -> Optional[str]:
    return item.get("query_img_path") or item.get("img_path") or item.get("image_path")


def option_image_paths(item: Dict[str, Any]) -> List[str]:
    return item.get("option_imgs_path") or item.get("option_img_paths") or []


def option_video_paths(item: Dict[str, Any]) -> List[str]:
    return item.get("option_videos_path") or item.get("option_video_paths") or []


def _image_part(image_path: str, max_pixels: int) -> Dict[str, Any]:
    return {"type": "image", "image": image_path, "max_pixels": max_pixels}


def _video_part(video_path: str, max_pixels: int) -> Dict[str, Any]:
    return {"type": "video", "video": video_path, "max_pixels": max_pixels}


def build_messages_for_item(
    task_dir: str,
    item: Dict[str, Any],
    media_max_pixels: int,
) -> List[Dict[str, Any]]:
    prompt = item["query"].rstrip()
    parts: List[Dict[str, Any]] = []

    query_video = item.get("query_video_path") or item.get("query_video") or item.get("video_path")
    query_image = query_image_path(item)
    opt_imgs = option_image_paths(item)
    opt_vids = option_video_paths(item)

    if query_video:
        video_abs = os.path.join(task_dir, query_video)
        if not os.path.isfile(video_abs):
            raise FileNotFoundError(f"Missing query video: {video_abs}")
        parts.append(
            {
                "type": "text",
                "text": (
                    "The following video is the query video only. "
                    "It is not one of the A/B/C/D answer choices."
                ),
            }
        )
        parts.append(_video_part(video_abs, media_max_pixels))
    elif query_image:
        image_abs = os.path.join(task_dir, query_image)
        if not os.path.isfile(image_abs):
            raise FileNotFoundError(f"Missing query image: {image_abs}")
        parts.append(
            {
                "type": "text",
                "text": (
                    "The following single image is the query/current-state image only. "
                    "It is not option A, B, C, or D."
                ),
            }
        )
        parts.append(_image_part(image_abs, media_max_pixels))
    else:
        raise ValueError(f"Sample {item.get('id')} has neither query_video_path nor image path")

    if opt_imgs or opt_vids:
        parts.append(
            {
                "type": "text",
                "text": (
                    "Everything below is exactly four answer choices in order: "
                    "A, then B, then C, then D."
                ),
            }
        )

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_index = 0
    for image_rel in opt_imgs:
        label = labels[choice_index] if choice_index < len(labels) else str(choice_index + 1)
        image_abs = os.path.join(task_dir, image_rel)
        if not os.path.isfile(image_abs):
            raise FileNotFoundError(f"Missing option image: {image_abs}")
        parts.append({"type": "text", "text": f"Option {label} (image)"})
        parts.append(_image_part(image_abs, media_max_pixels))
        choice_index += 1

    for video_rel in opt_vids:
        label = labels[choice_index] if choice_index < len(labels) else str(choice_index + 1)
        video_abs = os.path.join(task_dir, video_rel)
        if not os.path.isfile(video_abs):
            raise FileNotFoundError(f"Missing option video: {video_abs}")
        parts.append({"type": "text", "text": f"Option {label} (video)"})
        parts.append(_video_part(video_abs, media_max_pixels))
        choice_index += 1

    parts.append({"type": "text", "text": f"{prompt}\n{ANSWER_SUFFIX}"})
    return [{"role": "user", "content": parts}]


def prepare_inputs(
    messages: List[Dict[str, Any]],
    processor: AutoProcessor,
    enable_thinking: bool,
) -> Any:
    from qwen_vl_utils import process_vision_info

    template_kwargs: Dict[str, Any] = {}
    if enable_thinking:
        assert 0
    if not enable_thinking:
        template_kwargs["enable_thinking"] = False

    try:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    except TypeError:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def get_response(
    messages: List[Dict[str, Any]],
    model: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    enable_thinking: bool,
    max_new_tokens: int,
) -> str:
    inputs = prepare_inputs(messages, processor, enable_thinking)
    inputs = inputs.to(device)

    
    _tok = processor.tokenizer
    _eos_cfg = getattr(model.generation_config, "eos_token_id", None)
    _eos_ids = {_tok.eos_token_id, _eos_cfg, 248044}
    _eos_ids = sorted(x for x in _eos_ids if x is not None)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": _eos_ids if len(_eos_ids) > 1 else _eos_ids[0],
        "pad_token_id": _tok.pad_token_id,
    }
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def inference_with_retry(
    messages: List[Dict[str, Any]],
    model: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    enable_thinking: bool,
    max_new_tokens: int,
    maxtry: int = 3,
) -> str:
    tries = maxtry
    while True:
        try:
            return get_response(messages, model, processor, device, enable_thinking, max_new_tokens)
        except Exception as exc:
            if tries <= 0:
                print(f"Request failed permanently: {exc}")
                traceback.print_exc()
                return ""
            tries -= 1
            print(f"Request failed ({exc}); {tries} retries left...")
            traceback.print_exc()
            time.sleep(5)


def should_run_task(
    dim: str,
    mod: str,
    only_dims: Optional[List[str]],
    only_mods: Optional[List[str]],
) -> bool:
    if only_dims and dim not in only_dims:
        return False
    if only_mods and mod not in only_mods:
        return False
    return True


def gpu_worker(
    gpu_id: int,
    task_list: List[TaskUnit],
    model_path: str,
    ckpt_path: str,
    load_method: str,
    skip_ckpt: bool,
    media_max_pixels: int,
    enable_thinking: bool,
    max_new_tokens: int,
    overwrite: bool,
) -> None:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Loading model with method={load_method} ...")

    model, processor = load_model_and_processor(
        model_path=model_path,
        ckpt_path=ckpt_path,
        load_method=load_method,
        skip_ckpt=skip_ckpt,
        device=device,
    )
    print(f"[GPU {gpu_id}] Model ready. ({len(task_list)} task(s))")

    for dim, mod, task_dir, out_dir, data in task_list:
        tag = f"[GPU {gpu_id}] {dim}/{mod}"
        print(f"{tag}  ({len(data)} samples)")

        for item in tqdm(data, desc=tag, position=gpu_id, leave=True):
            sample_id = str(item["id"])
            pred_path = os.path.join(out_dir, f"{sample_id}.json")
            if not overwrite and os.path.isfile(pred_path):
                continue

            try:
                messages = build_messages_for_item(task_dir, item, media_max_pixels)
                text = inference_with_retry(
                    messages, model, processor, device,
                    enable_thinking, max_new_tokens,
                )
            except Exception as exc:
                print(f"{tag} [skip id={sample_id}] {exc}")
                traceback.print_exc()
                text = ""

            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(text, f, ensure_ascii=False, indent=2)

    print(f"[GPU {gpu_id}] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-GPU Qwen3.5 inference for SWITCH-Basic"
    )
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--skip_ckpt", action="store_true",
                        help="Skip loading --ckpt and use base model weights")
    parser.add_argument(
        "--load_method",
        choices=["auto", "pt", "direct"],
        default="auto",
        help=(
            "auto: infer from --ckpt (directory -> model.safetensors, file -> .pt); "
            "pt: load --model_path then --ckpt .pt state dict; "
            "direct: load --ckpt as a directory containing model.safetensors"
        ),
    )
    parser.add_argument(
        "--direct_ckpt",
        type=str,
        default=DEFAULT_DIRECT_CKPT,
        help="Convenience default for --load_method direct when --ckpt is not overridden.",
    )
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--media_max_pixels", type=int, default=720 * 720)
    parser.add_argument("--dimensions", type=str, default="",
                        help="Comma-separated task folders to run")
    parser.add_argument("--modalities", type=str, default="",
                        help="Comma-separated modality subfolders")
    parser.add_argument("--list_only", action="store_true",
                        help="Print discovered tasks and exit")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing prediction JSON files")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    only_dims = [x.strip() for x in args.dimensions.split(",") if x.strip()]
    only_mods = [x.strip() for x in args.modalities.split(",") if x.strip()]
    ckpt_path = args.direct_ckpt if args.load_method == "direct" and args.ckpt == DEFAULT_CKPT else args.ckpt
    skip_ckpt = args.skip_ckpt or (
        not ckpt_path or ckpt_path.strip().lower() in ("none", "skip", "no", "false")
    )
    if not skip_ckpt:
        if args.load_method == "auto":
            if not (os.path.isdir(ckpt_path) or os.path.isfile(ckpt_path)):
                raise FileNotFoundError(f"Missing checkpoint file or directory: {ckpt_path}")
        elif args.load_method == "direct":
            if not os.path.isdir(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint directory: {ckpt_path}")
            if not os.path.isfile(os.path.join(ckpt_path, "model.safetensors")):
                raise FileNotFoundError(f"Missing model.safetensors under checkpoint directory: {ckpt_path}")
        elif not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")

    tasks = discover_tasks(args.data_path)
    if args.list_only:
        for dim, mod, vqa_path in tasks:
            print(f"{dim}/{mod}\t{vqa_path}")
        return

    all_task_units: List[TaskUnit] = []
    for dim, mod, vqa_path in tasks:
        if not should_run_task(dim, mod, only_dims or None, only_mods or None):
            continue
        task_dir = os.path.dirname(vqa_path)
        out_dir = os.path.join(args.output_path, dim, mod)
        os.makedirs(out_dir, exist_ok=True)
        with open(vqa_path, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]
        all_task_units.append((dim, mod, task_dir, out_dir, data))

    total_tasks = len(all_task_units)
    if total_tasks == 0:
        print("No tasks to run.")
        return
    num_gpus = min(args.num_gpus, total_tasks, torch.cuda.device_count())
    if num_gpus <= 0:
        raise RuntimeError("No CUDA devices available for multi-GPU inference.")
    print(f"Discovered {total_tasks} task(s), launching {num_gpus} GPU worker(s).")

    worker_tasks: List[List[TaskUnit]] = [[] for _ in range(num_gpus)]
    for i, tu in enumerate(all_task_units):
        worker_tasks[i % num_gpus].append(tu)

    procs: List[mp.Process] = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                worker_tasks[gpu_id],
                args.model_path,
                ckpt_path,
                args.load_method,
                skip_ckpt,
                args.media_max_pixels,
                args.enable_thinking,
                args.max_new_tokens,
                args.overwrite,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All done!")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
