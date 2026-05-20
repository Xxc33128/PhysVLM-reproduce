# PhysVLM V2 Implementation Plan: Deployment Optimization Loop

Last updated: 2026-05-20

## Summary

V1 has completed the PhysVLM reproduction, S-P Map ablation, error analysis, and PyTorch inference profiling. V2 upgrades the project from "reproducible + diagnosable" to "reproducible + diagnosable + deployment-aware".

The target story:

> This project reproduces PhysVLM for robotic physical reachability reasoning, diagnoses its S-P Map dependency and Yes-bias, and further builds a deployment optimization loop on Colab A100 covering PyTorch fp16, `torch.compile`, ONNX Runtime CUDA, ONNX Runtime TensorRT EP / TensorRT engine feasibility, and FastAPI serving benchmark.

The deployment goal is deliberately high but still technically honest. We should try full-model optimization where possible, but success does not depend on forcing the entire VLM `generate()` path into TensorRT. A valuable V2 outcome can be either:

- successful acceleration of the full inference path, or
- successful ONNX/TensorRT acceleration of exportable submodules plus a clear, reproducible failure analysis for the non-exportable VLM generation path.

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
  - optional longer run: `400`

## Key Deliverables

### 1. Unified Deployment Benchmark

Add `scripts/benchmark_deployment.py`.

Required backends:

- `pytorch`: current eager fp16 baseline.
- `torch_compile`: compiled PyTorch attempt.
- `onnx_cuda`: ONNX Runtime CUDA Execution Provider for exported submodules.
- `onnx_trt_ep`: ONNX Runtime TensorRT Execution Provider for exported submodules.
- `trtexec`: TensorRT engine benchmark where `trtexec` is available.
- `api`: FastAPI end-to-end serving benchmark.

Every backend result must include:

- `backend`
- `status`: `success`, `failed`, `skipped_env_missing`, or `skipped_unsupported_model`
- `num_examples`
- `mean_ms`
- `median_ms`
- `p90_ms`
- `examples_per_second`
- `cuda_peak_memory_gb`
- `notes`

### 2. ONNX Export And Parity

Add `scripts/export_onnx_modules.py`.

Export priority:

1. vision tower / image encoder
2. RGB + S-P Map image preprocessing path if separable
3. multimodal projector
4. feasibility probe for larger PhysVLM forward subgraph

Use:

```python
torch.onnx.export(..., dynamo=True, external_data=True)
```

Write export artifacts locally under:

```text
artifacts/onnx/
```

Do not commit exported `.onnx` files.

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

### 3. TensorRT Feasibility

Add `scripts/build_tensorrt_engine.py`.

Preferred order:

1. Try ONNX Runtime TensorRT Execution Provider.
2. If available, try `trtexec` FP16 engine build.
3. If both fail, capture failure reason and unsupported operators.

Provider order for ORT:

```python
[
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
```

TensorRT artifacts stay local:

```text
artifacts/tensorrt/
```

Do not commit `.engine`, `.plan`, timing cache, or raw build logs.

### 4. FastAPI Serving

Add:

- `scripts/serve_physvlm.py`
- `scripts/benchmark_api.py`

Minimum API:

- `GET /health`
- `POST /predict`

`/predict` input:

- image path
- depth path or depth mode
- question
- max new tokens

`benchmark_api.py` must record:

- sequential latency
- concurrency 2 latency
- concurrency 4 latency
- success count
- error count

API latency must be reported separately from pure model latency.

### 5. Result Reports

Add:

```text
results/deployment/deployment_summary.csv
results/deployment/deployment_report.md
results/deployment/backend_results/*.json
```

`deployment_report.md` should include:

- environment information
- backend table
- speedup table relative to PyTorch eager
- memory comparison
- parity results
- TensorRT success or failure analysis
- final deployment conclusion

## Implementation Sequence

1. Commit this V2 plan file first.
2. Add `.gitignore` rules for deployment artifacts.
3. Refactor current benchmark utilities only if needed; keep existing V1 scripts stable.
4. Implement PyTorch and `torch_compile` backend in `benchmark_deployment.py`.
5. Run smoke tests on 5 examples.
6. Implement ONNX export for the easiest stable submodule.
7. Add parity check.
8. Add ONNX Runtime CUDA benchmark.
9. Add ONNX Runtime TensorRT EP or `trtexec` path.
10. Add FastAPI serving and API benchmark.
11. Generate final `results/deployment/` report files.
12. Update README deployment section.
13. Update `CLAUDE.md` with V2 completion notes.
14. Commit and push V2.

## Test Plan

Smoke tests:

```bash
python scripts/benchmark_deployment.py --backend pytorch --limit 5
python scripts/benchmark_deployment.py --backend torch_compile --limit 5
python scripts/export_onnx_modules.py --limit 1
python scripts/check_export_parity.py --module vision_tower
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
- ONNX Runtime CUDA / TensorRT were tested on exportable modules.
- Full VLM generation deployment has known boundaries because `generate()` includes dynamic decoding, tokenizer interactions, custom multimodal inputs, and model-specific control flow.
- The value of V2 is the measured deployment feasibility, not an exaggerated claim of full TensorRT conversion.

## Resume Wording

If ONNX + TensorRT submodule acceleration succeeds:

> 构建 PhysVLM 部署优化实验闭环，在 Colab A100 上对 PyTorch fp16、`torch.compile`、ONNX Runtime CUDA 与 TensorRT FP16 后端进行 latency / throughput / 显存 benchmark，并通过输出一致性校验验证视觉编码与多模态投影模块的部署可靠性。

If ONNX succeeds but TensorRT fails:

> 完成 PhysVLM 部署可行性分析，将视觉编码/多模态投影子模块导出至 ONNX Runtime CUDA，建立 PyTorch 与 ONNX 输出一致性校验和推理 benchmark；同时记录 TensorRT 转换瓶颈，形成可复现的 VLM 部署边界分析。

If FastAPI serving is also completed:

> 进一步封装 FastAPI 推理服务与压测脚本，形成从离线评测、错误诊断到 GPU 推理部署 profiling 的完整工程闭环。

## Technical References

- PyTorch ONNX exporter: <https://docs.pytorch.org/docs/stable/onnx.html>
- ONNX Runtime CUDA Execution Provider: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- ONNX Runtime TensorRT Execution Provider: <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html>
- NVIDIA TensorRT Quick Start: <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html>
- NVIDIA TensorRT-LLM: <https://docs.nvidia.com/tensorrt-llm/>

