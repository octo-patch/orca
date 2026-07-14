"""Shared PRICE dataset discovery and Hugging Face download helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_REPO = "BAAI/PRICE"
DEFAULT_DATASET_REVISION = "main"
DATASET_METADATA_PATH = Path("data/test/metadata.jsonl")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_dataset_dir() -> Path:
    return project_root().parents[1] / "data" / "PRICE"


def add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(default_dataset_dir()),
        help="Local BAAI/PRICE dataset directory created by "
        "scripts/download_dataset.py (default: evaluation/data/PRICE).",
    )


def resolve_dataset_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    metadata_path = root / DATASET_METADATA_PATH
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"PRICE dataset metadata not found: {metadata_path}. Download it first with "
            "`python scripts/download_dataset.py`, or pass --dataset-dir."
        )
    return root


def load_manifest(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    root = resolve_dataset_root(dataset_root)
    metadata_path = root / DATASET_METADATA_PATH
    metadata_dir = metadata_path.parent
    manifest: dict[str, dict[str, Any]] = {}

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sample_id = str(entry.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"missing sample id at {metadata_path}:{line_number}")
            if sample_id in manifest:
                raise ValueError(f"duplicate sample id in dataset metadata: {sample_id}")

            query_name = str(entry.get("query_file_name", "")).strip()
            output_name = str(entry.get("output_file_name", "")).strip()
            instruction = str(entry.get("lang", "")).strip()
            if not query_name or not output_name or not instruction:
                raise ValueError(
                    f"incomplete PRICE metadata entry for {sample_id}: expected "
                    "query_file_name, output_file_name, and lang"
                )

            query_path = (metadata_dir / query_name).resolve()
            output_path = (metadata_dir / output_name).resolve()
            if not query_path.is_file():
                raise FileNotFoundError(f"query image missing for {sample_id}: {query_path}")
            if not output_path.is_file():
                raise FileNotFoundError(f"reference image missing for {sample_id}: {output_path}")

            normalized = dict(entry)
            normalized["query_path"] = str(query_path)
            normalized["output_path"] = str(output_path)
            manifest[sample_id] = normalized

    if not manifest:
        raise RuntimeError(f"PRICE dataset metadata is empty: {metadata_path}")
    return manifest


def download_dataset(
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    revision: str = DEFAULT_DATASET_REVISION,
    local_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install the PRICE evaluator first with "
            "`pip install .` or `uv pip install .`."
        ) from exc

    destination = Path(local_dir or default_dataset_dir()).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        allow_patterns=["README.md", "data/test/**"],
        force_download=force_download,
    )
    return resolve_dataset_root(downloaded)
