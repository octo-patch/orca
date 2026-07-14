# Orca Evaluation

Orca evaluation code is organized by generation modality:

```text
evaluation/
├── download_datasets.py
├── requirements-data.txt
├── data/                       # Downloaded datasets; ignored by Git
│   ├── PRICE/
│   ├── SWITCH/
│   ├── MVBench/
│   ├── TemporalBench/
│   └── 3DSRBench/
├── image_gen/
│   └── PRICE/
└── text_gen/
    ├── SWITCH/
    ├── TemporalBench/
    ├── 3DSRBench/
    └── lmms-eval/              # MVBench
```

## Download datasets

From the Orca repository root:

```bash
cd evaluation
python -m pip install -r requirements-data.txt

python download_datasets.py price
python download_datasets.py switch
python download_datasets.py mvbench
python download_datasets.py temporalbench
python download_datasets.py 3dsrbench
```

Download several datasets in one command, or all of them:

```bash
python download_datasets.py switch temporalbench 3dsrbench
python download_datasets.py --all
```

ZIP archives for MVBench and TemporalBench are extracted automatically. Pass `--delete-archives` to remove ZIP files after extraction.

TemporalBench is gated: request access on [microsoft/TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench), then run `hf auth login` before downloading.

MVBench requires 320 NTU RGB+D videos that Hugging Face cannot redistribute under the source license. Follow the upstream instructions and place those videos under `data/MVBench/video/nturgbd/`.

## Run evaluations

- Image generation: [image_gen/PRICE/README.md](image_gen/PRICE/README.md)
- Text generation: [text_gen/README.md](text_gen/README.md)
