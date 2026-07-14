# Local Gemma 4 Judge Environment

This directory is an isolated runtime environment for the optional local
Gemma 4 judge (`scripts/eval_imgs_with_gemma4.py`). It is kept
separate from the root `pyproject.toml` on purpose: the Gemma 4 judge needs
`torch` + a specific `transformers` release + CUDA, which are large,
hardware-sensitive dependencies that the default API-only workflow
(`scripts/eval_imgs.py`) does not need.

If you only plan to use the AiHubMix API judges, you can ignore this
directory entirely.

## Model weights

**You must bring your own Gemma 4 checkpoint.** This repository does not
distribute model weights, and we cannot confirm that a compatible checkpoint
is publicly downloadable. `scripts/eval_imgs_with_gemma4.py` only
provides the evaluation harness; it loads whatever local directory you point
it at via `AutoModelForMultimodalLM.from_pretrained(..., trust_remote_code=True)`
and `AutoProcessor.from_pretrained(...)`. The checkpoint directory must
contain a `config.json` and be loadable by those two calls.

## Install

This environment is managed with [`uv`](https://docs.astral.sh/uv/), not
plain `pip`/`venv` — `pyproject.toml` routes `torch`/`torchaudio`/`torchvision`
to a dedicated PyTorch CUDA 12.9 wheel index via `[tool.uv.sources]`, which
`pip` cannot resolve on its own.

```bash
cd envs/gemma4
uv sync
```

This creates `.venv/` in this directory and installs the pinned versions
(`torch==2.10.0+cu129`, `transformers==5.5.0`, `accelerate`, `pillow`,
`safetensors`, `tqdm`) from `uv.lock`. It requires Python >= 3.12 (uv will
download an interpreter if needed) and an NVIDIA GPU with a CUDA 12.9
compatible driver.

`.venv/` itself is gitignored; only `pyproject.toml` and `uv.lock` are
tracked, so `uv sync` reproduces the same resolved environment for anyone
who clones this repository.

### Optional: flash-attn

`flash-attn` is not on this project's dependency list because it has no
prebuilt wheel on a public package index for this exact combination
(CUDA 12.9, torch 2.10, Python 3.12, `cxx11abiTRUE`). The judge runs fine
without it (falls back to eager/sdpa attention), just slower. If you have a
matching wheel, install it manually into this venv:

```bash
uv pip install /path/to/flash_attn-<version>+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## Configure the model path

```bash
cp env.example.toml env.toml
```

Edit `env.toml`:

```toml
[gemma4]
model_path = "/path/to/your/local/gemma-4-checkpoint"
```

`env.toml` is gitignored and must not be committed. Alternatively, pass
`--model-path` on the command line each time instead of using `env.toml`.

## Run

From the `Orca/evaluation/image_gen/PRICE` directory, using this environment's interpreter:

```bash
envs/gemma4/.venv/bin/python scripts/eval_imgs_with_gemma4.py OUTPUT_ROOT
```

`OUTPUT_ROOT` uses the same layout as the API judge (see `evaluation/image_gen/PRICE/README.md`):
`OUTPUT_ROOT/imgs/<sample_id>.png`, matched by filename against the downloaded
`BAAI/PRICE` dataset metadata. Run `python scripts/download_dataset.py` first.

Useful options:

```bash
# Explicit model path instead of env.toml
envs/gemma4/.venv/bin/python scripts/eval_imgs_with_gemma4.py OUTPUT_ROOT \
  --model-path /path/to/local/gemma4

# Evaluate a subset of samples
envs/gemma4/.venv/bin/python scripts/eval_imgs_with_gemma4.py OUTPUT_ROOT \
  --ids agibot_world_1,agibot_world_2

# Multi-GPU: 2 GPUs per model replica (device_map=auto), auto-detects worker count
envs/gemma4/.venv/bin/python scripts/eval_imgs_with_gemma4.py OUTPUT_ROOT \
  --gpus-per-worker 2
```

By default this only supports evaluation without a ground-truth reference
image, matching the API judge and `eval_prompt/0601.py`.

## Outputs

Writes the same file pair as the API judge, using `--model-name` (default
`gemma4`) as the model component of the filename:

```text
OUTPUT_ROOT/eval_results.<prompt>.<model-name>.json
OUTPUT_ROOT/eval_results_summary.<prompt>.<model-name>.json
```
