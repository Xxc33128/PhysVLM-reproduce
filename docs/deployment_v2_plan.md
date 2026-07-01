# PhysVLM V2 Roadmap: Deployment Optimization Loop

Last updated: 2026-05-20 (revised)

## Summary

V1 has completed the PhysVLM reproduction, S-P Map ablation, error analysis, and PyTorch inference profiling. V2 upgrades the project from "reproducible + diagnosable" to "reproducible + diagnosable + deployment-aware".

The target story:

> This project reproduces PhysVLM for robotic physical reachability reasoning, diagnoses its S-P Map dependency and Yes-bias, and further builds a deployment optimization loop on Colab A100 covering PyTorch fp16, `torch.compile`, ONNX Runtime CUDA, ONNX Runtime TensorRT EP feasibility, and FastAPI serving benchmark.

The deployment goal is deliberately high but technically honest. We try full-model optimization where possible, but success does not depend on forcing the entire VLM `generate()` path into TensorRT. A valuable V2 outcome can be either:

- successful acceleration of the full inference path, or
- successful ONNX/TensorRT acceleration of exportable submodules plus a clear, reproducible failure analysis for the non-exportable VLM generation path.

### Design Principle: Focused Deployment Evidence

V2 is scoped as a reproducible deployment-feasibility study, not a production serving platform. Every task must produce measurable evidence about latency, parity, memory, or conversion boundaries. The plan avoids over-engineering (enterprise-level status enums, redundant engine builds, high-concurrency stress tests) in favor of clean, focused experiments that complete in approximately 12 hours of Colab A100 time.

## Current V1 Baseline

- GitHub repo: <https://github.com/Xxc33128/PhysVLM-reproduce>
- V1 documentation polish commit: `ae73731 polish PhysVLM v1 documentation`
- Full benchmark: `1279 / 1600 = 79.94%`
- S-P Map ablation:
  - `normal`: `79.94%`
  - `black`: `77.44%`
  - `mismatch`: `79.37%`
- Main failure mode: Yes-bias, with `319 / 321` normal-mode errors being false positives.
- PyTorch profiling baseline on Colab A100 fp16, 50 normal examples:
  - mean latency: `165.274 ms`
  - median latency: `155.228 ms`
  - p90 latency: `161.754 ms`
  - throughput: `6.0505 examples/s`
  - peak CUDA memory: `8.41 GB`

## Preflight Before V2 Work

Run these before implementation:

```bash
git status -sb --ignored
git pull --ff-only
python3 -m py_compile scripts/standalone_inference.py scripts/eval_standalone.py scripts/analyze_ablation.py scripts/benchmark_inference.py
```

Expected repo policy:

- Keep `results/images/` and `results/raw/` ignored.
- Do not commit model checkpoints, full simulator images, raw Colab archives, full raw prediction JSON files, ONNX models, or TensorRT engines.
- Commit only scripts, docs, lightweight CSV/JSON summaries, Markdown reports, and selected small visual evidence.

## Environment Assumptions

- Runtime: Colab A100.
- GPU memory is not the limiting factor.
- Default export/deployment baseline: full checkpoint fp16.
- Do not use 4-bit as the primary ONNX/TensorRT export baseline, because bitsandbytes quantization is not a stable export target.
- Batch size default: `1`.
- Depth mode default: `normal`.
- Benchmark sample sizes:
  - smoke: `5`
  - profiling: `50`
  - stronger report: `100`

## Phased Deliverables

### Phase 1: PyTorch Optimization + ONNX Export (must-do, ~6.5h)

This phase establishes the core deployment evidence: ONNX export, parity verification, and multi-backend benchmarking.

#### 1a. torch.compile Benchmark (~2h)

Add `scripts/benchmark_torch_compile.py`.

- Wrap the existing `model.generate()` call with `torch.compile(mode="reduce-overhead")`.
- Run warmup passes (3-5 iterations) to trigger compilation, then benchmark 50 examples.
- Record: mean/median/p90 latency, throughput, peak CUDA memory, compilation wall-clock time.
- Compare against the V1 eager fp16 baseline.
- Note: `torch.compile` may partially fail on VLM `generate()` due to dynamic shapes and graph breaks. Record graph break count if available.

#### 1b. ONNX Export: Vision Tower (~2h)

Add `scripts/export_onnx_modules.py`.

Export the vision tower (SigLIP/CLIP image encoder) as the primary target:

```python
torch.onnx.export(..., dynamo=True, external_data=True)
```

Write export artifacts locally under:

```text
artifacts/onnx/
```

Do not commit exported `.onnx` files.

If the vision tower exports cleanly, also attempt the multimodal projector MLP as a second submodule to broaden the "exportable submodules" claim.

#### 1c. Parity Check (~1h)

Add `scripts/check_export_parity.py`.

Parity metrics:

- `mae`
- `max_abs`
- `cosine_similarity`

Default acceptance:

- `cosine_similarity >= 0.999` for fp16/fp32 comparable outputs, or
- record the mismatch clearly in `deployment_report.md`.

Important wording rule:

- Submodule parity is not end-to-end answer parity.
- Do not claim full PhysVLM ONNX deployment unless the full answer generation path is actually exported and tested.

#### 1d. ONNX Runtime CUDA Benchmark (~1.5h)

Add `scripts/benchmark_onnx.py`.

- Load exported ONNX model with `CUDAExecutionProvider`.
- Benchmark on 50 examples using the same input data as the PyTorch baseline.
- Record: mean/median/p90 latency, throughput, peak memory.
- Compute speedup ratio vs PyTorch eager.

### Phase 2: TensorRT + FastAPI (high ROI, ~5.5h)

This phase adds TensorRT feasibility analysis and end-to-end serving capability. Both are commonly expected in embodied intelligence algorithm roles that involve model deployment.

#### 2a. TensorRT EP Attempt (~3h)

Add `scripts/benchmark_tensorrt_ep.py`.

- Try ONNX Runtime `TensorrtExecutionProvider` on the exported vision tower ONNX.
- Provider order:

```python
[
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
```

- If TRT EP succeeds: benchmark latency and compare with ORT CUDA.
- If TRT EP fails: capture the exact error, list unsupported operators, and document the failure boundary clearly. A well-documented failure analysis is still useful deployment evidence.

TensorRT artifacts stay local:

```text
artifacts/tensorrt/
```

Do not commit `.engine`, `.plan`, timing cache, or raw build logs.

Note: the standalone `trtexec` engine build is intentionally omitted. It duplicates the TRT EP test with extra environment setup overhead and little additional evidence for this benchmark.

#### 2b. FastAPI Serving + Benchmark (~2.5h)

Add:

- `scripts/serve_physvlm.py`
- `scripts/benchmark_api.py`

Minimum API:

- `GET /health`
- `POST /predict`

`/predict` input:

- image path or base64
- depth path or depth mode
- question
- max new tokens

`benchmark_api.py` must record:

- sequential latency (10 requests)
- concurrency-2 latency (10 requests)
- success count
- error count

API latency must be reported separately from pure model latency so the serving overhead is visible.

Note: concurrency-4 and above is omitted. For a single-GPU demonstration, concurrency-2 is sufficient to show request-level serving behavior without turning this into a load-testing project.

### Phase 3: Report + Extras (nice-to-have, ~2.5h)

#### 3a. Multimodal Projector ONNX Export (~1.5h)

If not already done in Phase 1b, export the `mm_projector` MLP:

- ONNX export + parity check + ORT CUDA benchmark.
- This broadens the exportable submodules story from one component to two.

#### 3b. Deployment Report + README Update (~1h)

Add:

```text
results/deployment/deployment_summary.csv
results/deployment/deployment_report.md
results/deployment/backend_results/*.json
```

`deployment_report.md` should include:

- environment information (GPU, driver, CUDA, PyTorch, ORT versions)
- backend comparison table (latency, throughput, memory per backend)
- speedup table relative to PyTorch eager
- parity results
- TensorRT success or failure analysis
- VLM deployment boundary discussion (why `generate()` is not fully exportable)
- final deployment conclusion

## Result Schema

Every backend result JSON must include:

- `backend`: one of `pytorch_eager`, `torch_compile`, `onnx_cuda`, `onnx_trt_ep`, `fastapi`
- `status`: `success` or `failed`
- `num_examples`
- `mean_ms`
- `median_ms`
- `p90_ms`
- `examples_per_second`
- `cuda_peak_memory_gb`
- `notes`: free text for graph breaks, unsupported ops, compilation time, etc.

## Implementation Sequence

1. Commit this V2 plan file and update `.gitignore` for deployment artifacts.
2. Keep existing V1 scripts stable; do not refactor unless necessary.
3. Implement `benchmark_torch_compile.py`, smoke test on 5 examples, then run 50.
4. Implement `export_onnx_modules.py` for vision tower, smoke test on 1 example.
5. Implement `check_export_parity.py`, verify cosine similarity.
6. Implement `benchmark_onnx.py` with CUDAExecutionProvider, run 50 examples.
7. Implement `benchmark_tensorrt_ep.py`, attempt TRT EP, document outcome.
8. Implement `serve_physvlm.py` and `benchmark_api.py`, run API benchmark.
9. (If time allows) Export multimodal projector, repeat parity + benchmark.
10. Generate `results/deployment/` report files.
11. Update README deployment section.
12. Commit and push V2.

## Test Plan

Smoke tests:

```bash
python scripts/benchmark_torch_compile.py --limit 5
python scripts/export_onnx_modules.py --module vision_tower
python scripts/check_export_parity.py --module vision_tower
python scripts/benchmark_onnx.py --limit 5
python scripts/benchmark_api.py --limit 5
```

Existing V1 checks should still pass:

```bash
python3 -m py_compile scripts/standalone_inference.py scripts/eval_standalone.py scripts/analyze_ablation.py scripts/benchmark_inference.py
```

Git hygiene checks:

```bash
git diff --check
find . -type f -size +10M -not -path './.git/*' -print
git status -sb --ignored
```

Expected outcome:

- No large model/export artifacts staged.
- No local absolute paths in tracked reports.
- No raw Colab archives or complete generated image folders staged.

## README Update Target

Add a section titled `Deployment Optimization`.

The section should say:

- PyTorch fp16 remains the end-to-end functional baseline.
- `torch.compile` was tested as a low-friction PyTorch optimization.
- ONNX Runtime CUDA / TensorRT were tested on exportable submodules (vision tower, multimodal projector).
- Full VLM generation deployment has known boundaries because `generate()` includes dynamic decoding, tokenizer interactions, custom multimodal inputs, and model-specific control flow.
- The value of V2 is the measured deployment feasibility and reproducible boundary analysis, not an exaggerated claim of full TensorRT conversion.

## Future: V2.5 Directions (not part of this plan)

These are independent tracks that can be done after V2:

1. **Qwen2-VL zero-shot baseline**: Run a general-purpose VLM on EQA-phys without S-P Map to establish a cross-model comparison. This is algorithm understanding work, not deployment work, and belongs in a separate README section.
2. **Gradient ablation experiments**: Gaussian blur, edge-only S-P Map, random noise replacement to further probe what spatial information the model uses.
3. **Generic VLM comparison**: InternVL or other open VLMs as additional baselines.

## Technical References

- PyTorch ONNX exporter: <https://docs.pytorch.org/docs/stable/onnx.html>
- ONNX Runtime CUDA Execution Provider: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- ONNX Runtime TensorRT Execution Provider: <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html>
- NVIDIA TensorRT Quick Start: <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html>
