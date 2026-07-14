# 3DSRBench Evaluation

## Download

```bash
cd Orca/evaluation
python download_datasets.py 3dsrbench
cd text_gen/3DSRBench
```

Run all commands from the 3DSRBench directory:

```bash
cd Orca/evaluation/text_gen/3DSRBench
conda activate orca_3dsrbench
```

The script imports shared helpers from the sibling `../common/` directory automatically.

## Common Arguments

- `--dataset-root ../../data/3DSRBench`: 3DSRBench dataset root.
- `--tsv ../../data/3DSRBench/3dsrbench_v1_vlmevalkit_circular.tsv`: circular TSV file.
- `--output-dir /path/to/outputs`: directory for predictions and metrics.


## Orca Inference

```bash
python scripts/orca.py \
  --model_dir /path/to/Qwen3.5 \
  --ckpt /path/to/Orca/checkpoint \
  --dataset-root ../../data/3DSRBench \
  --tsv ../../data/3DSRBench/3dsrbench_v1_vlmevalkit_circular.tsv \
  --output-dir /path/to/outputs
```
