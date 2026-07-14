# lmms-eval Evaluation

## Download MVBench

```bash
cd Orca/evaluation
python download_datasets.py mvbench
cd text_gen/lmms-eval
```

The downloader stores annotations and extracted archives under `../../data/MVBench`. Due to the NTU RGB+D license, 320 videos must still be downloaded manually using `../../data/MVBench/video/MVBench_videos_ntu.txt` and placed under `../../data/MVBench/video/nturgbd/`.

This local lmms-eval fork is used for Orca evaluation on MVBench. Run commands from this directory:

```bash
cd Orca/evaluation/text_gen/lmms-eval
conda activate orca_mvbench
```

Install it in editable mode as described in [`../env/README.md`](../env/README.md) before launching evaluation.

## Common Arguments

- `--tasks mvbench`: task name or comma-separated task list.
- `--model MODEL_NAME`: model adapter registered by `lmms-eval`.
- `--model_args key=value,...`: model paths and runtime options.
- `--batch_size 1`: recommended default for video/multi-image VLM evaluation.
- `--output_path /path/to/outputs`: output directory.

For MVBench-style local data, set the dataset root with environment variables.
`MVBENCH_DATA_ROOT` should contain the `json/` annotation directory. If
`MVBENCH_VIDEO_ROOT` is not set, the code uses `MVBENCH_DATA_ROOT` for videos.

```bash
export MVBENCH_DATA_ROOT=../../data/MVBench
export MVBENCH_VIDEO_ROOT=../../data/MVBench
```

## Orca Inference

```bash
accelerate launch \
  --num_processes 1 \
  --main_process_port 29539 \
  -m lmms_eval \
  --model orca \
  --model_args pretrained=/path/to/Qwen3.5,checkpoint_path=/path/to/Orca/checkpoint,checkpoint_load_method=auto,enable_thinking=False,max_num_frames=64 \
  --tasks mvbench \
  --gen_kwargs max_new_tokens=32,temperature=0 \
  --batch_size 1 \
  --log_samples \
  --output_path /path/to/outputs
```
