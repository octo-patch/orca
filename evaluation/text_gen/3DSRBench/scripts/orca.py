#!/usr/bin/env python3
"""Evaluate 3DSRBench with Orca inference.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

# Shared multi-GPU evaluation helpers.
REPO_ROOT = Path(__file__).resolve().parent.parent
TZH_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(TZH_ROOT))
from common.qwen_eval_multigpu import (  # noqa: E402
    SINGLE_GPU_DEVICE_MAP,
    WORKER_CONFIG_FLAG,
    load_worker_config,
    resolve_physical_gpus,
    spawn_subprocess_workers,
    split_round_robin,
)

PREFIX = "qwen_vl_interface.model."
MODEL_SPECS: dict[str, str] = {
    "4B": "your_qwen_path",
    "0.8B": "your_qwen_path",
}
DEFAULT_MODEL_SIZE = "4B"
SYSTEM_MESSAGE = "You are a helpful assistant."
DEFAULT_DATASET_ROOT = "your_data_path"
DEFAULT_TSV = "${PROJECT_ROOT}/3dsrbench_v1_vlmevalkit_circular.tsv"
FALLBACK_TSV_NAME = "3dsrbench_v1_vlmevalkit_circular.tsv"
DEFAULT_OUTPUT_ROOT = "${PROJECT_ROOT}/local/outputs"


def resolve_project_root_path(raw_path: str, project_root: Path) -> Path:
    return Path(raw_path.replace("${PROJECT_ROOT}", str(project_root))).expanduser()

# Circular evaluation categories.
LABELS = ["A", "B", "C", "D"]
TYPE_MAPPING = {
    "location": ["location_above", "location_closer_to_camera", "location_next_to"],
    "height": ["height_higher"],
    "orientation": ["orientation_in_front_of", "orientation_on_the_left", "orientation_viewpoint"],
    "multi_object": [
        "multi_object_closer_to",
        "multi_object_facing",
        "multi_object_viewpoint_towards_object",
        "multi_object_parallel",
        "multi_object_same_direction",
    ],
}
QUESTION_TYPES = ["height", "location", "orientation", "multi_object"]
SUBTYPES = sum([TYPE_MAPPING[k] for k in QUESTION_TYPES], [])


def run_id_from_weight_location(model_dir: str, ckpt_path: str | None) -> str:
    """Last three path segments, underscores; file basename without extension if ckpt is set."""
    if ckpt_path:
        p = Path(ckpt_path).expanduser().resolve()
        is_file = True
    else:
        p = Path(model_dir).expanduser().resolve()
        is_file = False
    parts = list(p.parts)
    tail = parts[-3:] if len(parts) >= 3 else parts
    segments = list(tail)
    if is_file:
        segments[-1] = Path(segments[-1]).stem
    return "_".join(segments)


def git_head_hash_40(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        h = proc.stdout.strip()
        return h[:40] if h else None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def load_starvlm_qwen_state_dict(ckpt_path: str) -> dict:
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    raw = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    out = {}
    for k, v in raw.items():
        out[k[len(PREFIX) :] if k.startswith(PREFIX) else k] = v
    return out


def load_starvlm_qwen_state_dict_from_dir(ckpt_dir: Path) -> dict:
    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint directory is missing model.safetensors: {safetensors_path}"
        )
    raw = load_file(str(safetensors_path), device="cpu")
    out = {}
    direct_prefix = "vlm.model."
    has_direct_prefix = any(k.startswith(direct_prefix) for k in raw.keys())
    for k, v in raw.items():
        if has_direct_prefix and not k.startswith(direct_prefix):
            continue
        key = k
        if has_direct_prefix and key.startswith(direct_prefix):
            key = key[len(direct_prefix) :]
        out[key[len(PREFIX) :] if key.startswith(PREFIX) else key] = v
    return out


def resolve_processor_source(model_dir: str, ckpt_path: str | None) -> str:
    if not ckpt_path:
        return model_dir
    ckpt = Path(ckpt_path).expanduser().resolve()
    if ckpt.is_dir():
        vlm_cfg = ckpt / "vlm_config"
        if vlm_cfg.is_dir():
            return str(vlm_cfg)
        if (ckpt / "processor_config.json").is_file() or (ckpt / "tokenizer_config.json").is_file():
            return str(ckpt)
    return model_dir


def apply_eos_from_processor(model: Qwen3_5ForConditionalGeneration, processor: Any) -> None:
    tok = processor.tokenizer
    im_end_id = tok.convert_tokens_to_ids(tok.eos_token)
    eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
    model.generation_config.eos_token_id = [im_end_id, eot_id]
    model.generation_config.pad_token_id = tok.pad_token_id


def load_model(
    model_dir: str,
    ckpt_path: str | None,
    load_method: str = "auto",
    *,
    device_map: str | dict[str, int] = SINGLE_GPU_DEVICE_MAP,
) -> tuple[Qwen3_5ForConditionalGeneration, Any]:
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device_map,
            trust_remote_code=True,
        )
    if ckpt_path is not None:
        ckpt = Path(ckpt_path).expanduser().resolve()
        if load_method == "auto":
            use_dir = ckpt.is_dir()
        else:
            use_dir = load_method == "direct"
        if use_dir:
            state_dict = load_starvlm_qwen_state_dict_from_dir(ckpt)
        else:
            state_dict = load_starvlm_qwen_state_dict(str(ckpt))
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[load_state_dict] strict=False: {msg}", flush=True)
    model.eval()

    processor_source = resolve_processor_source(model_dir, ckpt_path)
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    apply_eos_from_processor(model, processor)
    return model, processor


def decode_image_b64(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def build_mcq_user_text(question: str, choices: dict[str, str]) -> str:
    lines = [question.strip(), ""]
    for lab in LABELS:
        c = choices.get(lab, "").strip()
        if c:
            lines.append(f"{lab}. {c}")
    lines.append("")
    lines.append("Answer with a single capital letter (A, B, C, or D) only.")
    return "\n".join(lines)


def normalize_circular_qid(qid: str) -> str:
    if len(qid) >= 2 and qid[-2] == "-":
        return qid[:-2]
    return qid


def extract_choice_letter(raw: str, allowed: set[str]) -> str | None:
    s = raw.strip().upper()
    m = re.search(r"\b([ABCD])\b", s)
    if m and m.group(1) in allowed:
        return m.group(1)
    m = re.search(r"[ABCD]", s)
    if m and m.group(0) in allowed:
        return m.group(0)
    return None


@torch.inference_mode()
def generate_reply(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    *,
    image: Image.Image,
    user_text: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    device = model.device
    input_ids = inputs["input_ids"]
    gen_in = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == "pixel_values":
            gen_in[k] = v.to(device=device, dtype=torch.bfloat16)
        else:
            gen_in[k] = v.to(device)

    generated_ids = model.generate(**gen_in, max_new_tokens=max_new_tokens, do_sample=False)
    gen_trimmed = generated_ids[:, input_ids.shape[1] :]
    return processor.batch_decode(
        gen_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]


def aggregate_circular(per_row: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate circular evaluation scores by question group."""
    grouped: dict[str, dict[str, Any]] = {}
    for r in per_row:
        gqid = r["group_qid"]
        cat = r["category"]
        assert cat in SUBTYPES, cat
        hit = int(r["correct"])
        if gqid in grouped:
            grouped[gqid]["product"] *= hit
        else:
            grouped[gqid] = {"product": hit, "category": cat}

    products = [grouped[k]["product"] for k in grouped]
    overall = float(np.mean(products)) if products else 0.0

    by_type: dict[str, float] = {}
    for t in QUESTION_TYPES:
        sub = TYPE_MAPPING[t]
        vals = [grouped[k]["product"] for k in grouped if grouped[k]["category"] in sub]
        by_type[t] = float(np.mean(vals)) if vals else 0.0

    by_subtype: dict[str, float] = {}
    for st in SUBTYPES:
        vals = [grouped[k]["product"] for k in grouped if grouped[k]["category"] == st]
        by_subtype[st] = float(np.mean(vals)) if vals else 0.0

    return {
        "num_groups": len(grouped),
        "overall": round(overall * 100, 1),
        "by_type": {k: round(v * 100, 1) for k, v in by_type.items()},
        "by_subtype": {k: round(v * 100, 1) for k, v in by_subtype.items()},
        "group_detail": {k: {"product": v["product"], "category": v["category"]} for k, v in grouped.items()},
    }


def _eval_shard_3dsrbench(
    physical_gpu: int,
    indices: list[int],
    run_cfg: dict[str, Any],
    shard_path: Path,
) -> None:
    model, processor = load_model(
        run_cfg["model_dir"],
        run_cfg.get("ckpt"),
        run_cfg.get("load_method", "auto"),
        device_map=SINGLE_GPU_DEVICE_MAP,
    )

    df = pd.read_csv(run_cfg["tsv_path"], sep="\t")
    limit = run_cfg.get("limit")
    if limit is not None:
        df = df.iloc[: int(limit)].copy()
    df = df.reset_index(drop=True)
    max_new_tokens = int(run_cfg["max_new_tokens"])
    quiet = bool(run_cfg["quiet"])

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(shard_path, "w", encoding="utf-8") as out_f:
        for pos_i in tqdm(indices, desc=f"gpu={physical_gpu}", disable=quiet):
            row = df.iloc[pos_i]
            qid = str(row["qid"])
            category = str(row["category"])
            gold = str(row["answer"]).strip().upper()
            choice_labels = {lab: str(row[lab]) for lab in LABELS if lab in row}
            non_empty = {k: v for k, v in choice_labels.items() if str(v).strip() != ""}
            allowed = set(non_empty.keys())

            image = decode_image_b64(str(row["image"]))
            user_text = build_mcq_user_text(str(row["question"]), choice_labels)

            raw = generate_reply(
                model,
                processor,
                image=image,
                user_text=user_text,
                max_new_tokens=max_new_tokens,
            )
            pred = extract_choice_letter(raw, allowed)
            is_correct = pred is not None and pred == gold

            df_index = df.index[pos_i]
            record = {
                "index": pos_i,
                "df_index": int(df_index) if isinstance(df_index, (int, np.integer)) else df_index,
                "qid": qid,
                "group_qid": normalize_circular_qid(qid),
                "category": category,
                "gold": gold,
                "parsed": pred,
                "raw_response": raw,
                "correct": bool(is_correct),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if not quiet and pos_i < 3:
                print(f"[gpu={physical_gpu}] --- {qid} {category}", flush=True)
                print(user_text[:500], flush=True)
                print(f"gold={gold} pred={pred} raw={raw!r}", flush=True)


def _load_shard_records(shard_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["index"]))
    return rows


def _run_worker_job(config_path: Path) -> None:
    job = load_worker_config(config_path)
    _eval_shard_3dsrbench(
        int(job["physical_gpu"]),
        list(job["indices"]),
        job["run_cfg"],
        Path(job["shard_path"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3DSRBench circular split with Orca.")
    parser.add_argument(WORKER_CONFIG_FLAG, type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_SPECS.keys()),
        default=DEFAULT_MODEL_SIZE,
        help="Qwen3.5 base model size (sets --model_dir unless overridden).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Override base HuggingFace model directory (default from --model-size).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Optional checkpoint file (.pt) or checkpoint export directory "
            "(expects model.safetensors). Omit to use --model_dir weights only."
        ),
    )
    parser.add_argument(
        "--load_method",
        choices=["auto", "pt", "direct"],
        default="auto",
        help=(
            "How to load --ckpt: auto infers from the path (directory -> model.safetensors, "
            "file -> .pt state dict); pt forces a .pt state dict; direct forces a checkpoint "
            "export directory."
        ),
    )
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--tsv", type=str, default=DEFAULT_TSV)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        metavar="ID",
        help="Logical GPU ids to use. Default: all visible CUDA devices.",
    )
    args = parser.parse_args()
    if args._worker_config is not None:
        _run_worker_job(Path(args._worker_config))
        return

    physical_gpus = resolve_physical_gpus(args.gpus)
    print(f"[INFO] Using physical GPU(s): {physical_gpus}", flush=True)

    model_dir = args.model_dir if args.model_dir is not None else MODEL_SPECS[args.model_size]
    print(f"[INFO] model-size={args.model_size}, model_dir={model_dir}", flush=True)

    ckpt = args.ckpt if args.ckpt else None
    repo_root = REPO_ROOT
    run_id = run_id_from_weight_location(model_dir, ckpt)
    git_commit = git_head_hash_40(repo_root)

    root = Path(args.dataset_root).expanduser()
    tsv_path = resolve_project_root_path(args.tsv, repo_root)
    if not tsv_path.is_file():
        if args.tsv == DEFAULT_TSV:
            tsv_path = root / FALLBACK_TSV_NAME
        else:
            tsv_arg = Path(args.tsv)
            if not tsv_arg.is_absolute():
                tsv_path = root / tsv_arg

    df = pd.read_csv(tsv_path, sep="\t")
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()
    df = df.reset_index(drop=True)

    n_samples = len(df)
    if n_samples == 0:
        raise SystemExit("TSV dataset is empty.")

    output_root = resolve_project_root_path(args.output_dir, repo_root)
    run_dir_name = f"{run_id}_{datetime.now().strftime('%m%d-%H%M%S')}"
    output_dir = output_root / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(n_samples))
    shards = split_round_robin(indices, len(physical_gpus))
    for gpu, shard in zip(physical_gpus, shards):
        print(f"[INFO] gpu={gpu}, shard_size={len(shard)}", flush=True)

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    run_cfg: dict[str, Any] = {
        "model_dir": model_dir,
        "ckpt": ckpt,
        "load_method": args.load_method,
        "tsv_path": str(tsv_path),
        "max_new_tokens": args.max_new_tokens,
        "quiet": args.quiet,
        "limit": args.limit,
    }

    jobs: list[dict[str, Any]] = []
    for gpu, shard in zip(physical_gpus, shards):
        if not shard:
            continue
        jobs.append(
            {
                "physical_gpu": gpu,
                "indices": shard,
                "run_cfg": run_cfg,
                "shard_path": str(shard_dir / f"shard_{gpu}.jsonl"),
                "config_path": str(shard_dir / f"worker_{gpu}.json"),
            }
        )
    spawn_subprocess_workers(script_path=Path(__file__), jobs=jobs)

    shard_paths = [shard_dir / f"shard_{gpu}.jsonl" for gpu in physical_gpus]
    per_question = _load_shard_records(shard_paths)
    if len(per_question) != n_samples:
        raise SystemExit(
            f"Expected {n_samples} predictions, got {len(per_question)} after merging shards."
        )

    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as pred_f:
        for record in per_question:
            pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = aggregate_circular(per_question)
    payload = {
        "config": {
            "backend": "transformers_orca",
            "run_id": run_id,
            "model_size": args.model_size,
            "model_dir": model_dir,
            "ckpt": ckpt,
            "load_method": args.load_method,
            "tsv": str(tsv_path),
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "system_message": SYSTEM_MESSAGE,
            "git_commit_hash": git_commit,
            "gpus": physical_gpus,
            "parallel_workers": len(physical_gpus),
        },
        "summary": summary,
        "per_question": per_question,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {results_path}", flush=True)
    print(f"Wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()
