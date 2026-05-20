# PhysVLM Reproduction & S-P Map Ablation Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/01_colab_pro_reproduction.ipynb)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)

> Reproducible evaluation pipeline and diagnostic ablation study for **PhysVLM** robotic physical reachability reasoning, built on the EQA-phys simulator benchmark.

**PhysVLM** (CVPR 2025) is a vision-language model that takes an RGB image and a Spatial-Physical Map (S-P Map) as dual-channel input to reason about whether a robot arm can physically reach a target object. This project reproduces the evaluation, designs ablation experiments to probe how much the model actually relies on S-P Map information, and provides structured error analysis.

## Motivation

The official PhysVLM release is research-oriented and lacks a turnkey evaluation path: the server entrypoint (`start_physvlm_server.py`) is missing, and full-checkpoint loading can silently return `None`. This project addresses those gaps and goes further by asking: **does PhysVLM genuinely use the S-P Map, or does it rely primarily on RGB and language priors?**

## Pipeline

```mermaid
graph LR
    A[EQA-phys Simulator<br/>PyBullet scenes] --> B[RGB + S-P Map<br/>generation]
    B --> C[Standalone<br/>Inference]
    C --> D[Offline<br/>Evaluation]
    D --> E[Benchmark<br/>Analysis]
    D --> F[S-P Map<br/>Ablation]
    D --> G[Inference<br/>Profiling]
    E --> H[Per-robot accuracy<br/>Confusion matrix<br/>Object-level errors]
    F --> I[normal vs black<br/>vs mismatch]
    G --> J[Latency / Throughput<br/>CUDA memory]
```

## What Is Different From The Official Repo

- **Standalone inference**: bypasses the missing server entrypoint with a direct `PhysVLMPredictor` class.
- **Compatibility fixes**: handles full-checkpoint loading returning `None` and resolves the released SigLIP tower alias to the correct Hugging Face model id.
- **Offline evaluator**: batch evaluation over the full EQA-phys benchmark with structured JSON output.
- **S-P Map ablation protocol**: three depth modes (`normal` / `black` / `mismatch`) to isolate the contribution of spatial-physical information.
- **Error analysis**: per-robot accuracy, question-type breakdown, Yes/No confusion matrices, and object-level error rates.
- **Inference profiling**: end-to-end PyTorch latency, throughput, and CUDA peak-memory measurement.

## Benchmark Results

Full evaluation on 1,600 Yes/No reachability QA examples across four robot platforms (Colab A100, fp16):

| Robot | Correct | Total | Accuracy |
| :--- | ---: | ---: | ---: |
| **ALL** | **1,279** | **1,600** | **79.94%** |
| CR5 | 325 | 400 | 81.25% |
| FR5 | 335 | 400 | 83.75% |
| PANDA | 321 | 400 | 80.25% |
| UR5 | 298 | 400 | 74.50% |

By question template: `direct_pick` reached 83.50%, `reachable_space` reached 76.38%.

## S-P Map Ablation

The core diagnostic question: **does the model use S-P Map information, or is it just relying on RGB and language cues?**

| Mode | Description | Accuracy | Delta |
| :--- | :--- | ---: | ---: |
| `normal` | Aligned RGB + S-P Map | 79.94% | baseline |
| `black` | S-P Map replaced with black image | 77.44% | -2.50 pp |
| `mismatch` | S-P Map from a different scene | 79.37% | -0.57 pp |

**Key findings:**

1. **S-P Map provides useful signal** -- removing it (`black`) drops accuracy by 2.50 percentage points.
2. **RGB/language prior dominance** -- using a *wrong* S-P Map (`mismatch`) only drops 0.57 pp, indicating the model still relies heavily on visual and linguistic cues rather than fine-grained spatial reasoning.
3. **Persistent Yes-bias across all modes** -- the error direction table below shows false positives dominate regardless of S-P Map quality:

| Mode | False Positive (No->Yes) | False Negative (Yes->No) |
| :--- | ---: | ---: |
| `normal` | 319 | 2 |
| `black` | 360 | 1 |
| `mismatch` | 327 | 3 |

## Error Analysis

The dominant failure mode is a strong **Yes-bias**: out of 321 total errors in normal mode, 319 are false positives (label `No`, prediction `Yes`). The model almost never predicts `No` when the answer is `Yes`, but frequently overestimates reachability for unreachable objects.

| Label \ Prediction | Yes | No |
| :--- | ---: | ---: |
| Yes | 972 | 2 |
| No | 319 | 307 |

<p align="center">
  <img src="results/visualization/wrong_cases_contact_sheet.jpg" width="80%" alt="Wrong case examples"/>
  <br/>
  <em>Selected false-positive cases across four robot platforms</em>
</p>

<p align="center">
  <img src="results/visualization/correct_cases_contact_sheet.jpg" width="80%" alt="Correct case examples"/>
  <br/>
  <em>Selected correctly classified cases</em>
</p>

## Inference Profiling

PyTorch end-to-end profiling on Colab A100 (fp16, 50 examples):

| Metric | Value |
| :--- | ---: |
| Mean latency | 165.3 ms |
| Median latency | 155.2 ms |
| P90 latency | 161.8 ms |
| Throughput | 6.05 examples/s |
| Peak CUDA memory | 8.41 GB |

## Project Structure

```
PhysVLM-reproduce/
  scripts/
    standalone_inference.py   # PhysVLMPredictor: direct RGB + S-P Map inference
    eval_standalone.py        # Offline evaluator with normal/black/mismatch modes
    analyze_ablation.py       # Ablation CSV + Markdown report generator
    benchmark_inference.py    # Latency / throughput / memory profiler
  notebooks/
    01_colab_pro_reproduction.ipynb  # Self-contained Colab reproduction runbook
  results/
    analysis/                 # Benchmark summaries, confusion tables, error analysis
    ablation/                 # S-P Map ablation tables and report
    profiling/                # Inference profiling JSON and report
    visualization/            # RGB + S-P Map visual case studies
```

## Quick Start

**1. Clone and install dependencies:**

```bash
git clone https://github.com/Xxc33128/PhysVLM-reproduce.git
cd PhysVLM-reproduce
pip install -r requirements.txt
```

**2. Run evaluation (example with black ablation):**

```bash
python scripts/eval_standalone.py \
  --physvlm-root /path/to/PhysVLM/physvlm-main \
  --model-path /path/to/PhysVLM-Qwen2.5-3B \
  --qa-json /path/to/phys_bench_sim_qas.json \
  --data-root /path/to/EQA-phys-simulator \
  --depth-mode black \
  --device cuda --load-4bit \
  --output-json results/eval_black.json
```

**3. Generate ablation report:**

```bash
python scripts/analyze_ablation.py \
  --normal-json results/eval_normal.json \
  --black-json results/eval_black.json \
  --mismatch-json results/eval_mismatch.json \
  --output-dir results/ablation
```

**4. Profile inference:**

```bash
python scripts/benchmark_inference.py \
  --physvlm-root /path/to/PhysVLM/physvlm-main \
  --model-path /path/to/PhysVLM-Qwen2.5-3B \
  --qa-json /path/to/phys_bench_sim_qas.json \
  --data-root /path/to/EQA-phys-simulator \
  --limit 50 --device cuda --load-4bit \
  --output-json results/profile.json --output-md results/profile.md
```

For the full step-by-step walkthrough including checkpoint download and simulator data generation, see the [Colab notebook](notebooks/01_colab_pro_reproduction.ipynb).

## Limitations

- This project does **not** retrain PhysVLM or propose a new model. The contribution is reproducible engineering, ablation tooling, and diagnostic analysis.
- Inference profiling is PyTorch-only; ONNX Runtime and TensorRT exports are left for future work.
- v1 does not include a generic VLM baseline (e.g., Qwen2-VL or InternVL); this is planned as a future extension.

## Acknowledgements

This project builds on [PhysVLM](https://github.com/unira-zwj/PhysVLM) by Weijie Zhu et al. (CVPR 2025) and uses the [EQA-phys simulator](https://github.com/unira-zwj/PhysVLM/tree/main/EQA-phys-simulator) for benchmark data generation.
