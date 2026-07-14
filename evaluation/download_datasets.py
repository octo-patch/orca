#!/usr/bin/env python3
"""Download Orca evaluation datasets into evaluation/data/."""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = EVALUATION_ROOT / "data"


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    directory: str
    allow_patterns: tuple[str, ...]
    archives: tuple[str, ...] = ()


DATASETS = {
    "price": DatasetSpec(
        repo_id="BAAI/PRICE",
        directory="PRICE",
        allow_patterns=("README.md", "data/test/**"),
    ),
    "switch": DatasetSpec(
        repo_id="BAAI-Agents/SWITCH-Basic-v1-open",
        directory="SWITCH",
        allow_patterns=(
            "README.md",
            "action/**",
            "final_state/**",
            "verification_action/**",
            "verification_state/**",
            "vqa_state/**",
            "vqa_task/**",
            "ui_grounding/**",
        ),
    ),
    "mvbench": DatasetSpec(
        repo_id="OpenGVLab/MVBench",
        directory="MVBench",
        allow_patterns=("README.md", "json/**", "video/**"),
        archives=("video/*.zip",),
    ),
    "temporalbench": DatasetSpec(
        repo_id="microsoft/TemporalBench",
        directory="TemporalBench",
        allow_patterns=("README.md", "temporalbench_short_qa.json", "short_video.zip"),
        archives=("short_video.zip",),
    ),
    "3dsrbench": DatasetSpec(
        repo_id="ccvl/3DSRBench",
        directory="3DSRBench",
        allow_patterns=("README.md", "3dsrbench_v1_vlmevalkit_circular.tsv"),
    ),
}

ALIASES = {"3dsr": "3dsrbench", "temporal": "temporalbench", "mv": "mvbench"}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member in {archive}: {member.filename}")
        handle.extractall(destination)


def _extract_archives(
    dataset_root: Path,
    patterns: tuple[str, ...],
    *,
    force: bool,
    delete_archives: bool,
) -> None:
    marker_root = dataset_root / ".extracted"
    for pattern in patterns:
        for archive in sorted(dataset_root.glob(pattern)):
            marker_name = str(archive.relative_to(dataset_root)).replace("/", "__") + ".done"
            marker = marker_root / marker_name
            if marker.is_file() and not force:
                print(f"Already extracted: {archive}")
            else:
                print(f"Extracting {archive} -> {archive.parent}")
                _safe_extract(archive, archive.parent)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("ok\n", encoding="utf-8")
            if delete_archives and archive.exists():
                archive.unlink()


def _validate(name: str, root: Path) -> None:
    if name == "price":
        required = [root / "data/test/metadata.jsonl"]
    elif name == "switch":
        required = list(root.rglob("vqa.json"))
    elif name == "mvbench":
        required = [root / "json/action_sequence.json", root / "video"]
    elif name == "temporalbench":
        required = [root / "temporalbench_short_qa.json"]
    else:
        required = [root / "3dsrbench_v1_vlmevalkit_circular.tsv"]

    missing = [path for path in required if not path.exists()]
    if not required or missing:
        raise RuntimeError(f"Dataset validation failed for {name}; missing: {missing or 'vqa.json'}")


def download_one(
    name: str,
    *,
    data_root: Path,
    revision: str,
    max_workers: int,
    force_download: bool,
    force_extract: bool,
    delete_archives: bool,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install the downloader dependency first: "
            "`python -m pip install -r requirements-data.txt`."
        ) from exc

    spec = DATASETS[name]
    destination = (data_root / spec.directory).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {spec.repo_id}@{revision} -> {destination}")
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(spec.allow_patterns),
            max_workers=max_workers,
            force_download=force_download,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"Access to {spec.repo_id} is gated. Request access on its Hugging Face "
            "page, run `hf auth login`, then retry."
        ) from exc

    _extract_archives(
        destination,
        spec.archives,
        force=force_extract,
        delete_archives=delete_archives,
    )
    _validate(name, destination)

    if name == "mvbench":
        print(
            "MVBench note: 320 NTU RGB+D videos are not hosted on Hugging Face due to "
            "their license. Download the files listed in video/MVBench_videos_ntu.txt "
            "from the official NTU source and place them under video/nturgbd/."
        )
    print(f"Ready: {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Orca evaluation datasets into evaluation/data/."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets to download: price, switch, mvbench, temporalbench, 3dsrbench.",
    )
    parser.add_argument("--all", action="store_true", help="Download every dataset.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--revision", default="main", help="HF branch, tag, or commit.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete ZIP files after successful extraction to save disk space.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be >= 1")
    if args.all:
        selected = list(DATASETS)
    else:
        selected = [ALIASES.get(name.lower(), name.lower()) for name in args.datasets]
    if not selected:
        raise SystemExit("Choose one or more datasets, or pass --all.")
    unknown = [name for name in selected if name not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown)}")

    data_root = Path(args.data_root).expanduser().resolve()
    for name in selected:
        download_one(
            name,
            data_root=data_root,
            revision=args.revision,
            max_workers=args.max_workers,
            force_download=args.force_download,
            force_extract=args.force_extract,
            delete_archives=args.delete_archives,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
