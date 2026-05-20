# Claude Handoff Notes

This repository is a polished GitHub/portfolio version of a PhysVLM reproduction project.

## Current Repository

- GitHub: https://github.com/Xxc33128/PhysVLM-reproduce
- Latest release-prep commit before this handoff: `35f0e5a prepare PhysVLM reproduction release`
- Main branch: `master`
- Local root: `/Users/xerxes3/Documents/研一/xxc/PhysVLM-reproduce`

## Project Story

This is not a plain clone of the official PhysVLM repository. The project turns the released research code into a reproducible and diagnosable engineering pipeline for robotic physical reachability reasoning.

The project:

1. Reproduces PhysVLM inference for EQA-phys robot reachability QA.
2. Replaces the missing/fragile official server-style entrypoint with standalone inference.
3. Adds offline evaluation over the EQA-phys simulator benchmark.
4. Adds S-P Map ablations to test whether the model actually uses physical-space information.
5. Adds error analysis, visual case studies, and PyTorch inference profiling.

## Main Results

Full benchmark on four robot platforms, 1600 Yes/No QA examples:

- Normal aligned RGB + S-P Map: `1279 / 1600 = 79.94%`
- CR5: `81.25%`
- FR5: `83.75%`
- PANDA: `80.25%`
- UR5: `74.50%`

S-P Map ablation:

- `normal`: `79.94%`
- `black`: `77.44%`
- `mismatch`: `79.37%`

Main interpretation:

- The black-depth setting drops by 2.50 percentage points, so S-P Map provides useful signal.
- The mismatched S-P Map only drops by 0.57 percentage points, so the model still relies heavily on RGB/language priors.
- The dominant failure mode is Yes-bias: in normal mode, 319 of 321 errors are false positives (`No -> Yes`).

Profiling:

- Device: Colab A100 fp16
- 50 normal examples
- Mean latency: `165.274 ms`
- Median latency: `155.228 ms`
- P90 latency: `161.754 ms`
- Throughput: `6.0505 examples/sec`
- Peak CUDA memory: `8.41 GB`

## Important Files

- `README.md`: polished GitHub project page and primary narrative.
- `notebooks/01_colab_pro_reproduction.ipynb`: Colab reproduction notebook.
- `scripts/standalone_inference.py`: direct RGB + S-P/depth inference wrapper.
- `scripts/eval_standalone.py`: offline evaluator with `--depth-mode normal|black|mismatch`.
- `scripts/analyze_ablation.py`: generates ablation CSVs and Markdown report.
- `scripts/benchmark_inference.py`: PyTorch latency/throughput/memory profiler.
- `results/analysis/`: benchmark summaries, confusion tables, object-level errors.
- `results/ablation/`: S-P Map ablation tables and report.
- `results/profiling/`: profiling JSON and Markdown report.
- `results/visualization/`: selected wrong/correct visual cases and contact sheets.

## Large Files And Ignore Policy

Do not commit:

- model checkpoints or Hugging Face weights
- full simulator images
- raw Colab output JSONs
- zip packs
- `results/images/`
- `results/raw/`

These are intentionally ignored by `.gitignore`.

## Validation Already Run

Before publishing:

- `python3 -m py_compile scripts/standalone_inference.py scripts/eval_standalone.py scripts/analyze_ablation.py scripts/benchmark_inference.py`
- `git diff --cached --check`
- secret/path scan over README, result reports, profiling JSON, and visual gallery
- large-file check to avoid committing checkpoints or raw archives

## Recommended Next Steps

Good v2 directions:

1. Add a generic VLM baseline such as Qwen2-VL or InternVL for comparison.
2. Add ONNX Runtime or TensorRT benchmark only if export is stable.
3. Improve analysis by grouping errors by object location, object category, and robot type.
4. Add a short demo video or GIF for the README if the repo needs stronger visual appeal.
5. Create a resume-oriented Chinese project explanation and interview Q&A.

## Suggested Prompt For Claude

If you are Claude reading this repository, start with:

> Please read `CLAUDE.md`, `README.md`, and the scripts under `scripts/`. Then review whether this repository is ready as a GitHub portfolio project for an embodied intelligence / LLM Agent internship application. Focus on project narrative, technical credibility, missing risks, and resume/interview positioning.

