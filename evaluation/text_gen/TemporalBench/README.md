# TemporalBench Evaluation

## Download

TemporalBench is gated. Request access at [microsoft/TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), authenticate with `hf auth login`, then run:

```bash
cd Orca/evaluation
python download_datasets.py temporalbench
cd text_gen/TemporalBench
```

The downloader fetches `temporalbench_short_qa.json` and extracts `short_video.zip` under `../../data/TemporalBench`.

Run all commands from the TemporalBench directory:

```bash
cd Orca/evaluation/text_gen/TemporalBench
conda activate orca_temporalbench
```

`--data_json` is resolved under `--data_folder`.

## Common Arguments

- `--data_folder ../../data/TemporalBench`: TemporalBench dataset root.
- `--data_json temporalbench_short_qa.json`: QA annotation file.
- `--output_folder /path/to/outputs`: output directory.
- `--num_gpus N`: number of GPU workers.
- `--overwrite`: overwrite existing predictions.
  
## Orca Inference

```bash
python scripts/orca.py \
  --data_folder ../../data/TemporalBench \
  --data_json temporalbench_short_qa.json \
  --model_path /path/to/Qwen3.5 \
  --ckpt /path/to/Orca/checkpoint \
  --load_method auto \
  --model_name Orca \
  --output_folder /path/to/outputs \
  --num_gpus 8 \
  --overwrite
```
