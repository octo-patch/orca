"""Shared multi-GPU helpers for Qwen3.5-VL eval scripts (one model copy per GPU)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

WORKER_CONFIG_FLAG = "--_worker-config"
SINGLE_GPU_DEVICE_MAP: dict[str, int] = {"": 0}


def logical_to_physical_gpu_map() -> list[int]:
    """Physical GPU id for each logical ``cuda:i`` index visible to this process."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        return [int(part.strip()) for part in env.split(",") if part.strip()]
    import torch

    return list(range(torch.cuda.device_count()))


def resolve_physical_gpus(gpus: list[int] | None) -> list[int]:
    """Resolve ``--gpus`` logical ids to physical ids; default = all visible devices."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for evaluation.")
    phys_map = logical_to_physical_gpu_map()
    n = len(phys_map)
    if n == 0:
        raise SystemExit("No CUDA devices visible.")
    if gpus is None:
        return list(phys_map)
    if len(gpus) == 0:
        raise SystemExit("--gpus requires at least one GPU id.")
    out: list[int] = []
    seen: set[int] = set()
    for logical in gpus:
        if logical < 0 or logical >= n:
            raise SystemExit(f"GPU id {logical} out of range (logical 0..{n - 1}).")
        physical = phys_map[logical]
        if physical not in seen:
            out.append(physical)
            seen.add(physical)
    return out


def split_round_robin(indices: list[int], n_shards: int) -> list[list[int]]:
    shards: list[list[int]] = [[] for _ in range(n_shards)]
    for i, idx in enumerate(indices):
        shards[i % n_shards].append(idx)
    return shards


def load_worker_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def spawn_subprocess_workers(*, script_path: Path, jobs: list[dict[str, Any]]) -> None:
    """Launch one subprocess per GPU (in parallel) with ``CUDA_VISIBLE_DEVICES`` set before import."""
    script_path = script_path.resolve()
    active: list[tuple[int, subprocess.Popen]] = []

    for job in jobs:
        indices = job["indices"]
        if not indices:
            continue
        physical_gpu = int(job["physical_gpu"])
        cfg_path = Path(job["config_path"])
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        cmd = [sys.executable, str(script_path), WORKER_CONFIG_FLAG, str(cfg_path)]
        print(
            f"[INFO] spawn worker physical_gpu={physical_gpu} shard_size={len(indices)}",
            flush=True,
        )
        proc = subprocess.Popen(cmd, env=env)
        active.append((physical_gpu, proc))

    failures: list[tuple[int, int]] = []
    for physical_gpu, proc in active:
        code = proc.wait()
        if code != 0:
            failures.append((physical_gpu, code))
    if failures:
        msg = ", ".join(f"gpu{g}={code}" for g, code in failures)
        raise SystemExit(f"One or more workers failed: {msg}")
