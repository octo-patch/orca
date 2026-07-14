import argparse
import json
import os
import re
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


DEFAULT_DATA_FOLDER = "your_data_path"
DEFAULT_DATA_JSON = "temporalbench_short_qa.json"
DEFAULT_MODEL_PATH = "your_qwen_path"
DEFAULT_CKPT_PATH = "your_checkpoint_path"
DEFAULT_DIRECT_CKPT = "your_checkpoint_path"
DEFAULT_OUTPUT_FOLDER = "./outputs"
PREFIX = "qwen_vl_interface.model."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU Qwen3.5-4B inference on TemporalBench short-video QA."
    )
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--data_json", type=str, default=DEFAULT_DATA_JSON)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT_PATH)
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
    parser.add_argument("--model_name", type=str, default="Orca")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--nframes", type=int, default=8)
    parser.add_argument("--media_max_pixels", type=int, default=720 * 720)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip_ckpt",
        action="store_true",
        default=False,
        help="Skip loading --ckpt and use base weights from --model_path only.",
    )
    parser.add_argument("--skip_missing", action="store_true", default=True)
    parser.add_argument("--list_missing", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument(
        "--score_only",
        action="store_true",
        help="Skip inference and compute BA/MBA from the existing prediction JSONL.",
    )
    return parser.parse_args()


def load_starvlm_qwen_state_dict(ckpt_path: str) -> dict:
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    raw = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    out = {}
    for key, value in raw.items():
        out[key[len(PREFIX):] if key.startswith(PREFIX) else key] = value
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


def selected_questions(
    questions: List[Dict[str, Any]],
    start: int,
    limit: int,
) -> List[Dict[str, Any]]:
    end: Optional[int] = None if limit <= 0 else start + limit
    return questions[start:end]


def validate_short_qa(args: argparse.Namespace) -> None:
    if args.data_json != DEFAULT_DATA_JSON:
        raise ValueError(
            "This script is configured for short-video QA. "
            f"Use --data_json {DEFAULT_DATA_JSON} or edit the script intentionally."
        )


def build_messages(
    video_path: str,
    question: Dict[str, Any],
    nframes: int,
    media_max_pixels: int,
) -> List[Dict[str, Any]]:
    video_part: Dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "max_pixels": media_max_pixels,
    }
    if nframes > 0:
        video_part["nframes"] = nframes

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "These are frames from the video."},
                video_part,
                {"type": "text", "text": question["question"].strip()},
            ],
        }
    ]


def prepare_inputs(
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
    enable_thinking: bool,
) -> Dict[str, Any]:
    template_kwargs: Dict[str, Any] = {}
    if not enable_thinking:
        template_kwargs["enable_thinking"] = False

    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )


def get_response(
    model: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    messages: List[Dict[str, Any]],
    enable_thinking: bool,
    max_new_tokens: int,
) -> str:
    inputs = prepare_inputs(processor, messages, enable_thinking).to(device)
    tokenizer = processor.tokenizer
    eos_cfg = getattr(model.generation_config, "eos_token_id", None)
    eos_candidates = [tokenizer.eos_token_id, eos_cfg, 248044]
    eos_ids = sorted(
        {
            item
            for candidate in eos_candidates
            for item in (candidate if isinstance(candidate, (list, tuple, set)) else [candidate])
            if item is not None
        }
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_ids if len(eos_ids) > 1 else eos_ids[0],
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def inference_with_retry(
    model: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    messages: List[Dict[str, Any]],
    enable_thinking: bool,
    max_new_tokens: int,
    maxtry: int = 3,
) -> str:
    tries = maxtry
    while True:
        try:
            return get_response(
                model,
                processor,
                device,
                messages,
                enable_thinking,
                max_new_tokens,
            )
        except Exception as exc:
            if tries <= 0:
                print(f"Request failed permanently: {exc}")
                return ""
            tries -= 1
            print(f"Request failed ({exc}); {tries} retries left...")
            time.sleep(5)


def load_model(
    model_path: str,
    ckpt_path: str,
    load_method: str,
    gpu_id: int,
    skip_ckpt: bool,
) -> tuple[Any, AutoProcessor, torch.device]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},
    }
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    if skip_ckpt:
        print(f"[GPU {gpu_id}] SKIP_CKPT enabled; using base weights from {model_path}")
    else:
        if load_method == "auto":
            use_dir = os.path.isdir(ckpt_path)
        elif load_method == "direct":
            use_dir = True
        else:
            use_dir = False

        if use_dir:
            state_dict = load_starvlm_qwen_state_dict_from_dir(ckpt_path)
            print(f"[GPU {gpu_id}] Loaded VLM checkpoint directory: {ckpt_path}")
        else:
            state_dict = load_starvlm_qwen_state_dict(ckpt_path)
            print(f"[GPU {gpu_id}] Loaded pt checkpoint: {ckpt_path}")
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[GPU {gpu_id}] {msg}")
    model.eval()
    processor = AutoProcessor.from_pretrained(resolve_processor_source(model_path, ckpt_path, skip_ckpt))
    processor.tokenizer.padding_side = "left"
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    eot_id = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    model.generation_config.eos_token_id = [im_end_id, eot_id]
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    print(f"[GPU {gpu_id}] eos im_end={im_end_id} endoftext={eot_id}")
    return model, processor, device


def extract_answer(response: str) -> str:
    text = response.strip()
    if not text:
        return ""

    patterns = [
        r"###\s*Answer\s*[:：]?\s*([A-D])\b",
        r"(?:answer|option|choice)\s*(?:is|:|：)?\s*([A-D])\b",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return text[0].upper()


def accuracy_summary(values: Iterable[bool]) -> Dict[str, Any]:
    values = list(values)
    correct = sum(1 for value in values if value)
    total = len(values)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def compute_metrics(
    preds: List[Dict[str, Any]],
    id2question: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    binary_correct: List[bool] = []
    mba_by_video: Dict[str, bool] = {}
    binary_by_dataset: Dict[str, List[bool]] = defaultdict(list)
    mba_by_dataset: Dict[str, Dict[str, bool]] = defaultdict(dict)
    binary_by_category: Dict[str, List[bool]] = defaultdict(list)
    mba_by_category: Dict[str, Dict[str, bool]] = defaultdict(dict)

    for pred in preds:
        idx = pred["idx"]
        if idx not in id2question:
            print(f"Skip unknown prediction id: {idx}")
            continue

        question = id2question[idx]
        gt = question["GT"].strip().upper()
        pred_answer = pred.get("pred") or extract_answer(pred.get("response", ""))
        correct = pred_answer == gt
        pred["pred"] = pred_answer
        pred["GT"] = gt
        pred["correct"] = correct

        video_name = question["video_name"]
        dataset = question.get("dataset", "unknown")
        category = question.get("category", "unknown")

        binary_correct.append(correct)
        mba_by_video[video_name] = mba_by_video.get(video_name, True) and correct

        binary_by_dataset[dataset].append(correct)
        mba_by_dataset[dataset][video_name] = (
            mba_by_dataset[dataset].get(video_name, True) and correct
        )

        binary_by_category[category].append(correct)
        mba_by_category[category][video_name] = (
            mba_by_category[category].get(video_name, True) and correct
        )

    return {
        "binary_accuracy": accuracy_summary(binary_correct),
        "multiple_binary_accuracy": accuracy_summary(mba_by_video.values()),
        "per_dataset": {
            dataset: {
                "binary_accuracy": accuracy_summary(binary_values),
                "multiple_binary_accuracy": accuracy_summary(
                    mba_by_dataset[dataset].values()
                ),
            }
            for dataset, binary_values in sorted(binary_by_dataset.items())
        },
        "per_category": {
            category: {
                "binary_accuracy": accuracy_summary(binary_values),
                "multiple_binary_accuracy": accuracy_summary(
                    mba_by_category[category].values()
                ),
            }
            for category, binary_values in sorted(binary_by_category.items())
        },
    }


def print_accuracy(label: str, summary: Dict[str, Any]) -> None:
    print(
        f"{label:<36} "
        f"{summary['correct']}/{summary['total']} "
        f"{summary['accuracy'] * 100:.2f}%"
    )


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("=" * 90)
    print_accuracy("Binary Accuracy (BA)", metrics["binary_accuracy"])
    print_accuracy(
        "Multiple Binary Accuracy (MBA)",
        metrics["multiple_binary_accuracy"],
    )

    print("-" * 90)
    print("Per dataset")
    for dataset, dataset_metrics in metrics["per_dataset"].items():
        print(f"[{dataset}]")
        print_accuracy("  BA", dataset_metrics["binary_accuracy"])
        print_accuracy("  MBA", dataset_metrics["multiple_binary_accuracy"])

    print("-" * 90)
    print("Per category")
    for category, category_metrics in metrics["per_category"].items():
        print(f"[{category}]")
        print_accuracy("  BA", category_metrics["binary_accuracy"])
        print_accuracy("  MBA", category_metrics["multiple_binary_accuracy"])


def read_predictions(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    preds: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_questions(
    questions: List[Dict[str, Any]],
    num_workers: int,
) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_workers)]
    for index, question in enumerate(questions):
        shards[index % num_workers].append(question)
    return shards


def gpu_worker(
    gpu_id: int,
    questions: List[Dict[str, Any]],
    completed_ids: List[str],
    data_folder: str,
    model_path: str,
    ckpt_path: str,
    load_method: str,
    shard_path: str,
    nframes: int,
    media_max_pixels: int,
    skip_missing: bool,
    enable_thinking: bool,
    max_new_tokens: int,
    overwrite: bool,
    skip_ckpt: bool,
) -> None:
    print(f"[GPU {gpu_id}] Loading model with method={load_method} ...")
    model, processor, device = load_model(model_path, ckpt_path, load_method, gpu_id, skip_ckpt)
    print(f"[GPU {gpu_id}] Model ready. ({len(questions)} sample(s))")

    done = set(completed_ids)
    mode = "w" if overwrite else "a"
    with open(shard_path, mode, encoding="utf-8") as out_file:
        for question in tqdm(questions, desc=f"GPU {gpu_id}", position=gpu_id, leave=True):
            idx = question["idx"]
            if not overwrite and idx in done:
                continue

            video_path = os.path.join(data_folder, question["video_name"])
            if not os.path.isfile(video_path):
                message = f"[GPU {gpu_id}] Missing video, skip: {video_path}"
                if not skip_missing:
                    raise FileNotFoundError(message)
                print(message)
                continue

            try:
                messages = build_messages(
                    video_path,
                    question,
                    nframes,
                    media_max_pixels,
                )
                response = inference_with_retry(
                    model,
                    processor,
                    device,
                    messages,
                    enable_thinking,
                    max_new_tokens,
                )
                pred_answer = extract_answer(response)
                correct = pred_answer == question["GT"].strip().upper()
                record = {
                    "idx": idx,
                    "response": response,
                    "pred": pred_answer,
                    "GT": question["GT"],
                    "correct": correct,
                    "gpu": gpu_id,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                done.add(idx)
            except Exception as exc:
                print(f"[GPU {gpu_id}] Error running {video_path}: {exc}")
                traceback.print_exc()
                continue

    print(f"[GPU {gpu_id}] Done.")


def merge_predictions(
    output_path: str,
    shard_paths: List[str],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    if not overwrite:
        for pred in read_predictions(output_path):
            merged[pred["idx"]] = pred

    for shard_path in shard_paths:
        for pred in read_predictions(shard_path):
            merged[pred["idx"]] = pred

    preds = list(merged.values())
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    return preds


def main() -> None:
    args = parse_args()
    validate_short_qa(args)

    skip_ckpt = args.skip_ckpt or (
        not args.ckpt or args.ckpt.strip().lower() in ("none", "skip", "no", "false")
    )
    ckpt_path = args.direct_ckpt if args.load_method == "direct" and args.ckpt == DEFAULT_CKPT_PATH else args.ckpt
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

    data_json_path = os.path.join(args.data_folder, args.data_json)
    output_dir = os.path.join(args.output_folder, args.data_json.split(".")[0], args.model_name)
    output_path = os.path.join(output_dir, f"{args.model_name}-frame{args.nframes}.jsonl")
    metrics_path = os.path.join(output_dir, f"{args.model_name}-frame{args.nframes}-metrics.json")

    with open(data_json_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    id2question = {q["idx"]: q for q in all_questions}
    questions = selected_questions(all_questions, args.start, args.limit)

    missing = [
        q for q in questions
        if not os.path.isfile(os.path.join(args.data_folder, q["video_name"]))
    ]
    if args.list_missing:
        for q in missing:
            print(os.path.join(args.data_folder, q["video_name"]))
        print(f"missing {len(missing)} / selected {len(questions)} videos")
        return

    os.makedirs(output_dir, exist_ok=True)
    if args.score_only:
        preds = read_predictions(output_path)
        metrics = compute_metrics(preds, id2question)
        print_metrics(metrics)
        write_json(metrics_path, metrics)
        print(f"Saved metrics to {metrics_path}")
        return

    num_gpus = min(args.num_gpus, torch.cuda.device_count(), len(questions))
    if num_gpus <= 0:
        raise RuntimeError("No CUDA devices available for multi-GPU inference.")
    print(f"Launching {num_gpus} GPU worker(s) for {len(questions)} sample(s).")

    completed_ids: List[str] = []
    if not args.overwrite:
        completed_ids = [pred["idx"] for pred in read_predictions(output_path)]

    question_shards = split_questions(questions, num_gpus)
    shard_paths = [
        os.path.join(output_dir, f"{args.model_name}-frame{args.nframes}.gpu{gpu_id}.jsonl")
        for gpu_id in range(num_gpus)
    ]

    procs: List[mp.Process] = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                question_shards[gpu_id],
                completed_ids,
                args.data_folder,
                args.model_path,
                ckpt_path,
                args.load_method,
                shard_paths[gpu_id],
                args.nframes,
                args.media_max_pixels,
                args.skip_missing,
                args.enable_thinking,
                args.max_new_tokens,
                args.overwrite,
                skip_ckpt,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True
            print(f"Worker exited with code {p.exitcode}")

    if failed:
        raise RuntimeError("At least one GPU worker failed.")

    preds = merge_predictions(output_path, shard_paths, args.overwrite)
    metrics = compute_metrics(preds, id2question)
    print_metrics(metrics)
    write_json(metrics_path, metrics)
    print(f"Saved predictions to {output_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
