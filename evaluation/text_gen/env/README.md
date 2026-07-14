# Evaluation Environment Installation

This directory documents how to create conda environments for the Orca text-generation benchmarks. Run the commands from the Orca repository root; each block changes into the benchmark directory that owns its requirement files.

Each benchmark folder contains its own requirement files:

```text
SWITCH/requirements.txt
TemporalBench/requirements.txt
lmms-eval/requirements.txt
3DSRBench/requirements.txt
```

The source environments were used on Ubuntu 22.04 and CUDA 12.4.

## 1. SWITCH

```bash
cd Orca/evaluation/text_gen/SWITCH
conda create -n orca_switch python=3.10 -c conda-forge
conda activate orca_switch
python -m pip install -r requirements.txt
python -m pip install flash_attn==2.7.4.post1 --no-build-isolation
```

## 2. TemporalBench

```bash
cd Orca/evaluation/text_gen/TemporalBench
conda create -n orca_temporalbench python=3.10 -c conda-forge
conda activate orca_temporalbench
python -m pip install -r requirements.txt
python -m pip install flash_attn==2.7.4.post1 --no-build-isolation
```

## 3. MVBench

MVBench uses the local `lmms-eval` code.

```bash
cd Orca/evaluation/text_gen/lmms-eval
conda create -n orca_mvbench python=3.10 -c conda-forge
conda activate orca_mvbench
python -m pip install -r requirements.txt
```

## 4. 3DSRBench

```bash
cd Orca/evaluation/text_gen/3DSRBench
conda create -n orca_3dsrbench python=3.12 -c conda-forge
conda activate orca_3dsrbench
python -m pip install -r requirements.txt
```
