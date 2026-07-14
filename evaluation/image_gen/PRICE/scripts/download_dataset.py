#!/usr/bin/env python3
"""Download the PRICE benchmark dataset from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

import price_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the BAAI/PRICE benchmark dataset from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=price_dataset.DEFAULT_DATASET_REPO,
        help=f"Hugging Face dataset repository (default: {price_dataset.DEFAULT_DATASET_REPO}).",
    )
    parser.add_argument(
        "--revision",
        default=price_dataset.DEFAULT_DATASET_REVISION,
        help=f"Dataset branch, tag, or commit (default: {price_dataset.DEFAULT_DATASET_REVISION}).",
    )
    parser.add_argument(
        "--local-dir",
        default=str(price_dataset.default_dataset_dir()),
        help="Download destination (default: evaluation/data/PRICE).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files even when they are already cached locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = price_dataset.download_dataset(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        force_download=args.force_download,
    )
    manifest = price_dataset.load_manifest(root)
    print(f"Downloaded {len(manifest)} PRICE samples to {Path(root)}")
    print(f"Dataset source: {args.repo_id}@{args.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
