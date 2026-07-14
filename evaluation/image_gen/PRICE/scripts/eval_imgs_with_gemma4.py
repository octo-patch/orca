#!/usr/bin/env python3
"""Evaluate PRICE predictions with a local Gemma 4 judge.

This script only runs correctly inside the isolated environment under
envs/gemma4/ (see envs/gemma4/README.md). It is not part of the root
pyproject.toml dependency set: torch/transformers stay out of the
lightweight API-judge environment on purpose.

Reads benchmark samples from the local BAAI/PRICE dataset downloaded from
Hugging Face and matches predictions at OUTPUT_ROOT/imgs/<sample_id>.png.

Multi-GPU behavior (default: visible_gpus // gpus_per_worker workers):
- When launched without --worker, the script detects visible CUDA devices and
  spawns data-parallel workers with disjoint sample shards.
- Each worker loads one model replica on --gpus-per-worker GPUs (device_map=auto).
- Workers can also be launched manually via --num-shards / --shard-index.

Writes:
  <output-root>/eval_results.<template>.<model-name>.json
  <output-root>/eval_results_summary.<template>.<model-name>.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_imgs as eval_imgs_api  # noqa: E402

DEFAULT_MODEL_NAME = "gemma4"
DEFAULT_GPUS_PER_WORKER = 2
MAX_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2


def _gemma4_env_root() -> Path:
    return Path(__file__).resolve().parent.parent / "envs" / "gemma4"


def load_gemma4_env_config(path: Path | None = None) -> dict[str, Any]:
    env_path = path or (_gemma4_env_root() / "env.toml")
    if not env_path.is_file():
        return {}
    import tomllib

    with env_path.open("rb") as f:
        config = tomllib.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"env config must be a TOML table: {env_path}")
    return config


def resolve_model_path_arg(model_path: str | None) -> str:
    if model_path:
        return model_path
    config = load_gemma4_env_config()
    gemma4_config = config.get("gemma4", {})
    configured_path = str(gemma4_config.get("model_path", "")).strip() if isinstance(
        gemma4_config, dict
    ) else ""
    if configured_path and configured_path != "/path/to/your/local/gemma-4-checkpoint":
        return configured_path
    raise ValueError(
        "No model path provided. Pass --model-path, or set [gemma4].model_path in "
        f"{_gemma4_env_root() / 'env.toml'} (copy it from env.example.toml first)."
    )


def get_visible_device_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [part.strip() for part in visible.split(",") if part.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(int(torch.cuda.device_count()))]
    return ["0"]


def detect_num_gpus() -> int:
    return len(get_visible_device_ids())


def worker_device_ids(device_ids: list[str], *, shard_index: int, gpus_per_worker: int) -> list[str]:
    start = shard_index * gpus_per_worker
    end = start + gpus_per_worker
    return device_ids[start:end]


def resolve_num_workers(args: argparse.Namespace) -> int:
    gpus_per_worker = max(int(args.gpus_per_worker), 1)
    total_gpus = detect_num_gpus()
    if args.num_workers is not None:
        num_workers = int(args.num_workers)
    elif args.num_shards is not None:
        num_workers = int(args.num_shards)
    else:
        num_workers = total_gpus // gpus_per_worker

    if num_workers <= 0:
        raise RuntimeError(
            f"no workers to launch: total_gpus={total_gpus}, gpus_per_worker={gpus_per_worker}"
        )
    if total_gpus < num_workers * gpus_per_worker:
        raise RuntimeError(
            f"not enough GPUs: need {num_workers * gpus_per_worker}, have {total_gpus} "
            f"(num_workers={num_workers}, gpus_per_worker={gpus_per_worker})"
        )
    return num_workers


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"model directory not found: {path}")
    config_file = path / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"config.json not found under {path}")
    return path


def resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype_name}, choose from {sorted(dtype_map)}")
    return dtype_map[dtype_name]


def model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _image_content(image_path: str | Path) -> dict[str, Any]:
    return {"type": "image", "url": str(image_path)}


def build_messages(sample: dict[str, Any], instruction_text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        _image_content(sample["input_image"]),
        _image_content(sample["output_image"]),
        {"type": "text", "text": instruction_text},
    ]
    return [{"role": "user", "content": content}]


def load_gemma4(model_path: Path, *, dtype_name: str) -> tuple[Any, Any]:
    dtype = resolve_dtype(dtype_name)
    num_gpus = detect_num_gpus()
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if num_gpus > 1:
        load_kwargs["device_map"] = "auto"
        print(
            f"Loading Gemma 4 from {model_path} "
            f"(dtype={dtype_name}, device_map=auto, num_gpus={num_gpus})"
        )
    else:
        print(f"Loading Gemma 4 from {model_path} (dtype={dtype_name}, device=cuda)")

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(str(model_path), **load_kwargs).eval()
    if num_gpus <= 1:
        model = model.cuda()
    return model, processor


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        config["temperature"] = args.temperature
    return config


def eval_sample_with_model(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    *,
    gen_config: dict[str, Any],
) -> dict[str, Any]:
    attempt_errors: list[str] = []
    instruction_text = eval_imgs_api.build_eval_prompt(sample["instruction"])
    messages = build_messages(sample, instruction_text)
    device = model_device(model)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = _move_to_device(inputs, device)
            input_len = inputs["input_ids"].shape[-1]
            outputs = model.generate(**inputs, **gen_config)
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            if not response or not str(response).strip():
                raise ValueError("empty model response")
            parsed = eval_imgs_api.parse_eval_response(str(response))
            result: dict[str, Any] = {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "input_image": sample["input_image"],
                "output_image": sample["output_image"],
                "status": "success",
                "score": parsed["score"],
                "reasoning": parsed["reasoning"],
            }
            if "instruction_score" in parsed:
                result["instruction_score"] = parsed["instruction_score"]
            if "physical_score" in parsed:
                result["physical_score"] = parsed["physical_score"]
            return result
        except Exception as exc:
            attempt_errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)

    return {
        "id": sample["id"],
        "instruction": sample["instruction"],
        "input_image": sample["input_image"],
        "output_image": sample["output_image"],
        "status": "failed",
        "error": f"Failed after {MAX_ATTEMPTS} attempts: {attempt_errors[-1]}",
        "attempt_errors": attempt_errors,
    }


def shard_samples(
    samples: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [sample for idx, sample in enumerate(samples) if idx % num_shards == shard_index]


def shard_results_path(output_root: Path, shard_index: int) -> Path:
    return output_root / f"eval_results_shard_{shard_index}.json"


def count_eval_shard_results(output_root: Path) -> int:
    total = 0
    for path in sorted(output_root.glob("eval_results_shard_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload.get("samples", [])
        if isinstance(samples, list):
            total += len(samples)
    return total


def wait_workers_with_progress(
    processes: list[tuple[int, subprocess.Popen[Any], Path]],
    *,
    output_root: Path,
    total_samples: int,
    poll_interval: float = 2.0,
) -> int:
    exit_code = 0
    pending = {shard_index: (proc, log_path) for shard_index, proc, log_path in processes}
    progress = tqdm(total=total_samples, desc="Evaluating", unit="img")
    last_done = 0

    try:
        while pending:
            done_count = count_eval_shard_results(output_root)
            if done_count > last_done:
                progress.update(done_count - last_done)
                last_done = done_count

            for shard_index in list(pending.keys()):
                proc, log_path = pending[shard_index]
                rc = proc.poll()
                if rc is None:
                    continue
                if rc != 0:
                    exit_code = rc
                    print(f"[launcher] shard {shard_index} failed with exit code {rc}, see {log_path}")
                else:
                    print(f"[launcher] shard {shard_index} done")
                pending.pop(shard_index)

            if pending:
                time.sleep(poll_interval)

        done_count = count_eval_shard_results(output_root)
        if done_count > last_done:
            progress.update(done_count - last_done)
    finally:
        progress.close()

    return exit_code


def build_worker_command(args: argparse.Namespace, *, num_shards: int, shard_index: int) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.output_root),
        "--model-path",
        str(args.model_path),
        "--model-name",
        str(args.model_name),
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--gpus-per-worker",
        str(args.gpus_per_worker),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--worker",
    ]
    if args.eval_prompt is not None:
        cmd.extend(["--eval-prompt", args.eval_prompt])
    if args.do_sample:
        cmd.append("--do-sample")
    if getattr(args, "ids", None):
        cmd.extend(["--ids", args.ids])
    if getattr(args, "ids_file", None):
        cmd.extend(["--ids-file", args.ids_file])
    cmd.extend(["--dataset-dir", args.dataset_dir])
    return cmd


def launch_parallel_workers(args: argparse.Namespace, num_workers: int) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    device_ids = get_visible_device_ids()
    gpus_per_worker = max(int(args.gpus_per_worker), 1)
    max_workers = len(device_ids) // gpus_per_worker
    num_workers = min(int(num_workers), max_workers)
    if num_workers <= 0:
        raise RuntimeError("no CUDA devices available for parallel evaluation")

    sample_ids = eval_imgs_api.resolve_sample_ids_from_args(args)
    dataset_root = eval_imgs_api.price_dataset.resolve_dataset_root(args.dataset_dir)
    all_samples = eval_imgs_api.load_samples(
        output_root, dataset_root=dataset_root, ids=sample_ids
    )
    total_samples = len(all_samples)

    processes: list[tuple[int, subprocess.Popen[Any], Path]] = []
    for shard_index in range(num_workers):
        cmd = build_worker_command(args, num_shards=num_workers, shard_index=shard_index)
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        shard_device_ids = worker_device_ids(
            device_ids,
            shard_index=shard_index,
            gpus_per_worker=gpus_per_worker,
        )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(shard_device_ids)
        log_path = output_root / f"eval_shard_{shard_index}.log"
        print(
            f"[launcher] shard {shard_index}/{num_workers} "
            f"gpus={env['CUDA_VISIBLE_DEVICES']} -> {log_path}"
        )
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((shard_index, proc, log_path))

    return wait_workers_with_progress(
        processes,
        output_root=output_root,
        total_samples=total_samples,
    )


def merge_shard_results(output_root: Path, *, num_shards: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        path = shard_results_path(output_root, shard_index)
        if not path.is_file():
            raise FileNotFoundError(f"missing shard results: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError(f"invalid samples payload in {path}")
        merged.extend(samples)
    merged.sort(key=lambda item: item["id"])
    return merged


def configure_eval_prompt(args: argparse.Namespace) -> tuple[Path, eval_imgs_api.EvalPromptConfig]:
    eval_prompt_path = eval_imgs_api.resolve_eval_prompt_path(args.eval_prompt)
    eval_prompt_config = eval_imgs_api.load_eval_prompt_config(eval_prompt_path)
    eval_imgs_api._apply_eval_prompt_config(eval_prompt_config)
    return eval_prompt_path, eval_prompt_config


def run_worker(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    model_path = resolve_model_path(args.model_path)
    configure_eval_prompt(args)
    sample_ids = eval_imgs_api.resolve_sample_ids_from_args(args)
    dataset_root = eval_imgs_api.price_dataset.resolve_dataset_root(args.dataset_dir)
    all_samples = eval_imgs_api.load_samples(
        output_root, dataset_root=dataset_root, ids=sample_ids
    )
    samples = shard_samples(
        all_samples,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(
        f"Shard {args.shard_index}/{args.num_shards}: {len(samples)} samples, "
        f"model={model_path}, gpus={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    model, processor = load_gemma4(model_path, dtype_name=args.dtype)
    gen_config = generation_config(args)
    results: list[dict[str, Any]] = []
    shard_path = shard_results_path(output_root, args.shard_index)

    def write_shard() -> None:
        shard_payload = {
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": str(model_path),
            "model_name": args.model_name,
            "output_root": str(output_root),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "samples": results,
        }
        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(shard_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp_path.replace(shard_path)

    progress = tqdm(samples, desc=f"Shard {args.shard_index}/{args.num_shards}", unit="img")
    for sample in progress:
        result = eval_sample_with_model(
            model,
            processor,
            sample,
            gen_config=gen_config,
        )
        results.append(result)
        ok_count = sum(1 for item in results if item.get("status") == "success")
        fail_count = len(results) - ok_count
        progress.set_postfix(ok=ok_count, fail=fail_count, refresh=False)
        write_shard()
    progress.close()

    write_shard()
    print(f"Wrote {shard_path}")
    failed = sum(1 for item in results if item.get("status") != "success")
    return 0 if failed == 0 else 1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PRICE predictions with a local Gemma 4 judge."
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Output directory containing imgs/<sample_id>.png files.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the local Gemma 4 checkpoint. If omitted, read from "
        "[gemma4].model_path in envs/gemma4/env.toml.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Short model name used in output filenames (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--gpus-per-worker",
        type=int,
        default=DEFAULT_GPUS_PER_WORKER,
        help=f"GPUs per model replica via device_map=auto (default: {DEFAULT_GPUS_PER_WORKER}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: visible_gpus // gpus_per_worker).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of worker shards (manual worker mode).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Worker shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens for judge generation (default: 1024).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling for judge generation (default: greedy decoding).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used when --do-sample is set (default: 1.0).",
    )
    parser.add_argument(
        "--eval-prompt",
        type=str,
        default=None,
        help="Path to eval prompt Python file. Defaults to eval_prompt/0601.py.",
    )
    eval_imgs_api.add_ids_arguments(parser)
    eval_imgs_api.price_dataset.add_dataset_argument(parser)
    return parser.parse_args(argv)


def finalize_results(args: argparse.Namespace, *, num_workers: int) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    eval_prompt_path, _ = configure_eval_prompt(args)
    results = merge_shard_results(output_root, num_shards=num_workers)
    summary = eval_imgs_api.compute_summary(
        results,
        output_root=output_root,
        concurrency=num_workers,
        eval_prompt_path=eval_prompt_path,
        dataset_root=eval_imgs_api.price_dataset.resolve_dataset_root(args.dataset_dir),
    )
    summary["model"] = str(Path(args.model_path).expanduser().resolve())
    summary["model_name"] = args.model_name
    summary["gpus_per_worker"] = args.gpus_per_worker
    summary["num_workers"] = num_workers

    evaluated_at = summary["evaluated_at"]
    eval_results = {
        "evaluated_at": evaluated_at,
        "model": str(Path(args.model_path).expanduser().resolve()),
        "model_name": args.model_name,
        "output_root": str(output_root),
        "num_workers": num_workers,
        "gpus_per_worker": args.gpus_per_worker,
        "samples": results,
    }

    template_name = eval_imgs_api.eval_prompt_template_name(eval_prompt_path)
    output_paths = [
        eval_imgs_api.eval_results_path(output_root, template=template_name, model=args.model_name),
        eval_imgs_api.eval_results_summary_path(
            output_root, template=template_name, model=args.model_name
        ),
    ]
    write_json(output_paths[0], eval_results)
    write_json(output_paths[1], summary)

    for path in output_paths:
        print(f"Wrote {path}")
    print(
        f"Done: {summary['successful_samples']}/{summary['total_samples']} succeeded, "
        f"average_score={summary['average_score']}"
    )
    return 0 if summary["failed_samples"] == 0 else 1


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.gpus_per_worker <= 0:
        raise ValueError("--gpus-per-worker must be >= 1")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"output directory not found: {output_root}")

    args.model_path = resolve_model_path_arg(args.model_path)
    resolve_model_path(args.model_path)

    if args.worker:
        if args.num_shards is None or args.shard_index is None:
            raise ValueError("--worker requires --num-shards and --shard-index")
        return run_worker(args)

    num_workers = resolve_num_workers(args)
    exit_code = launch_parallel_workers(args, num_workers)
    if exit_code != 0:
        return exit_code
    return finalize_results(args, num_workers=num_workers)


if __name__ == "__main__":
    raise SystemExit(main())
