# SWITCH Evaluation

## Download

```bash
cd Orca/evaluation
python download_datasets.py switch
cd text_gen/SWITCH
```

The dataset is stored at `../../data/SWITCH` relative to this directory.

Run all commands from the SWITCH directory:

```bash
cd Orca/evaluation/text_gen/SWITCH
conda activate orca_switch
```

## Common arguments

- `--data_path ../../data/SWITCH`: SWITCH dataset root.
- `--output_path /path/to/outputs`: prediction output directory.
- `--num_gpus N`: number of GPU workers.
- `--overwrite`: overwrite existing predictions.

## Orca Inference

```bash
python scripts/orca.py \
  --data_path ../../data/SWITCH \
  --output_path /path/to/outputs \
  --model_path /path/to/Qwen3.5 \
  --ckpt /path/to/Orca/checkpoint \
  --load_method auto \
  --num_gpus 8 \
  --overwrite
```
After inference, score with:

```bash
python scripts/eval.py \
  --gen_baseline_name RUN_NAME \
  --results_root_dir /path/to/results \
  --gt_root_dir ../../data/SWITCH
```