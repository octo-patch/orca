# Text Generation Evaluation

This directory contains Orca evaluation code for four text-generation VQA benchmarks: SWITCH, MVBench, TemporalBench, and 3DSRBench. Each benchmark keeps its own runtime requirements and launch instructions.

## Directory layout

```text
Orca/evaluation/text_gen/
├── README.md
├── env/                    # Environment setup guide
├── common/                 # Shared multi-GPU evaluation helpers
├── SWITCH/                 # SWITCH inference and scoring
├── TemporalBench/          # TemporalBench inference and scoring
├── 3DSRBench/              # 3DSRBench inference and scoring
└── lmms-eval/              # Local lmms-eval fork used for MVBench
```

Paths in this documentation are relative to the cloned Orca repository. Start from the repository root unless a command explicitly changes into a benchmark directory:

```bash
git clone https://github.com/orca-wm/Orca.git
cd Orca
```

## 1. Install benchmark environments

The benchmarks use separate conda environments because their dependency sets differ. Follow [env/README.md](env/README.md), which always changes into the relevant benchmark directory before installing requirements.

After activating an environment, use `python` or `accelerate` directly. You do not need to construct an environment-specific interpreter path.

## 2. Download datasets

All datasets use the shared `Orca/evaluation/data/` directory. From the Orca repository root:

```bash
cd evaluation
python -m pip install -r requirements-data.txt

python download_datasets.py switch
python download_datasets.py mvbench
python download_datasets.py temporalbench
python download_datasets.py 3dsrbench
```

The resulting paths are fixed:

```text
evaluation/data/
├── SWITCH/
├── MVBench/
├── TemporalBench/
└── 3DSRBench/
```

| Benchmark | Dataset | Working directory | Instructions |
| --- | --- | --- | --- |
| SWITCH | [BAAI-Agents/SWITCH-Basic-v1-open](https://huggingface.co/datasets/BAAI-Agents/SWITCH-Basic-v1-open) | `evaluation/text_gen/SWITCH` | [SWITCH/README.md](SWITCH/README.md) |
| MVBench | [OpenGVLab/MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) | `evaluation/text_gen/lmms-eval` | [lmms-eval/README.md](lmms-eval/README.md) |
| TemporalBench | [microsoft/TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) | `evaluation/text_gen/TemporalBench` | [TemporalBench/README.md](TemporalBench/README.md) |
| 3DSRBench | [ccvl/3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench) | `evaluation/text_gen/3DSRBench` | [3DSRBench/README.md](3DSRBench/README.md) |

For MVBench, `MVBENCH_DATA_ROOT` must point to the dataset root containing the `json/` annotation directory. `MVBENCH_VIDEO_ROOT` may point to a separate video root; if omitted, it defaults to `MVBENCH_DATA_ROOT`. Set these variables after changing into `Orca/evaluation/text_gen/lmms-eval`:

```bash
export MVBENCH_DATA_ROOT=../../data/MVBench
export MVBENCH_VIDEO_ROOT=../../data/MVBench
```

TemporalBench requires prior approval on its gated Hugging Face page. MVBench additionally requires the 320 NTU RGB+D videos listed in `data/MVBench/video/MVBench_videos_ntu.txt`.

## 3. Prepare model weights

Download the Orca fine-tuned checkpoint:

- [Orca-4B](https://huggingface.co/BAAI/Orca-4B)

Download the required Qwen base model, for example:

- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)

The benchmark commands distinguish two paths:

- `MODEL_PATH`: base Qwen model directory;
- `CHECKPOINT_PATH`: Orca fine-tuned checkpoint file or exported checkpoint directory, such as `BAAI/Orca-4B`.

## 4. Run a benchmark

Run each command from its benchmark working directory:

```bash
cd Orca/evaluation/text_gen/SWITCH
# follow SWITCH/README.md

cd ../TemporalBench
# follow TemporalBench/README.md

cd ../3DSRBench
# follow 3DSRBench/README.md

cd ../lmms-eval
# follow lmms-eval/README.md for MVBench
```

The `common/` directory is imported by benchmark scripts and is not a standalone entry point.
