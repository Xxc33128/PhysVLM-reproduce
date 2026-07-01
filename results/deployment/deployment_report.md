# PhysVLM Deployment Optimization Report

This V2 report adds a deployment-oriented loop on top of the V1 PhysVLM reproduction and S-P Map ablation work. The goal is deployment feasibility with honest boundaries: full answer generation remains a PyTorch path, while ONNX Runtime CUDA is evaluated on exportable submodules.

Colab entrypoint: `notebooks/02_colab_v2_deployment.ipynb`.

## Environment

- Runtime: Colab A100
- GPU: NVIDIA A100-SXM4-40GB
- Main model path: `$PHYSVLM_MODEL_PATH`
- Data root: `$PHYSVLM_DATA_ROOT`
- Deployment sample set: 50 normal-depth examples from the generated UR5 smoke QA file
- Batch size: 1
- ONNX export: `torch.onnx.export(..., dynamo=True, external_data=True)`
- ONNX Runtime providers for CUDA runs: `CUDAExecutionProvider`, `CPUExecutionProvider`

## Backend Results

| Backend | Scope | Status | Examples | Mean ms | Median ms | P90 ms | Throughput | Peak CUDA memory |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch eager fp16 | full `generate()` | success | 50 | 165.274 | 155.228 | 161.754 | 6.0505/s | 8.41 GB |
| `torch.compile` | full `generate()` | success, not a useful speedup | 50 | 3443.13 | 83.129 | 371.142 | 0.2904/s | 8.4169 GB |
| ONNX Runtime CUDA | `vision_tower` | success | 50 | 14.653 | 14.435 | 14.517 | 68.246/s | 10.0957 GB |
| ONNX Runtime CUDA | `mm_projector` | success | 50 | 0.850 | 0.846 | 0.876 | 1175.9577/s | 9.1211 GB |
| ONNX Runtime TensorRT EP | `vision_tower` | failed | 50 | n/a | n/a | n/a | n/a | 10.1035 GB |
| FastAPI | full API path | not run | n/a | n/a | n/a | n/a | n/a | n/a |

Source JSON files live under `results/deployment/backend_results/`.

## Speedup Context

| Comparison | Result | Interpretation |
| :--- | :--- | :--- |
| `torch.compile` mean vs PyTorch eager mean | 0.048x | Not a speedup; dynamic `generate()` causes graph breaks and outliers. |
| `torch.compile` median vs PyTorch eager median | 1.87x | Median improved, but the mean and graph-break count make this unreliable as a deployment claim. |
| `vision_tower` ONNX CUDA vs full PyTorch eager mean | 11.28x lower latency | Not end-to-end comparable; this is submodule latency, not answer generation latency. |
| `mm_projector` ONNX CUDA vs full PyTorch eager mean | 194.44x lower latency | Not end-to-end comparable; this is a lightweight projector submodule benchmark. |

The `torch.compile` run completed, but recorded `graph_break_delta=67` and `warmup_s=91.859`. Because the mean latency was much worse than PyTorch eager, V2 treats this as a measured deployment boundary rather than a successful acceleration.

## ONNX Export And Parity

| Module | Export status | Input shape | Output shape | MAE | Max abs | Cosine similarity |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `vision_tower` | success | `[1, 3, 384, 384]` | `[1, 729, 1152]` | 0.006217 | 1.59375 | 0.9999799 |
| `mm_projector` | success | `[1, 243, 1152]` | `[1, 243, 2048]` | 0.000218 | 0.015625 | 0.9999991 |

Both exported submodules pass the default parity threshold of `cosine_similarity >= 0.999`. This is submodule parity only; it is not end-to-end answer parity.

## TensorRT EP Attempt

TensorRT EP was attempted through ONNX Runtime provider order:

```python
[
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
```

In this Colab environment, TensorRT EP failed to load because `libnvinfer.so.10` was missing. ONNX Runtime then fell back to `CUDAExecutionProvider`. The fallback latency was preserved in `fallback_*` fields inside `onnx_trt_ep.json`, but it is not reported as TensorRT latency.

Final TensorRT conclusion: no valid TensorRT latency is claimed for this run. The result is a reproducible environment boundary.

## Deployment Boundary

The full PhysVLM `generate()` path is not treated as a stable ONNX/TensorRT target because it includes dynamic autoregressive decoding, tokenizer-dependent control flow, custom multimodal keyword arguments, cache updates, and model-specific branching. V2 therefore separates the deployment story into:

1. End-to-end PyTorch fp16 and `torch.compile` measurements for the real answer-generation path.
2. ONNX Runtime CUDA measurements for exportable submodules: `vision_tower` and `mm_projector`.
3. TensorRT EP feasibility analysis, including failure when runtime libraries are unavailable.
4. FastAPI serving scripts for future online request benchmarking.

This boundary is intentional: the project claims measured deployment feasibility and boundary analysis, not a full TensorRT conversion.

## Project Summary

> 构建 PhysVLM 部署可行性分析闭环，在 Colab A100 上对 PyTorch fp16、`torch.compile`、ONNX Runtime CUDA 与 TensorRT EP 可用性进行 benchmark；将视觉编码与多模态投影子模块导出至 ONNX，并通过输出一致性校验验证导出可靠性（cosine similarity 分别为 0.99998 / 0.999999）；同时记录 TensorRT 运行时依赖缺失导致的部署边界，避免夸大为完整 TensorRT 转换。

Short version:

> 完成 PhysVLM 部署可行性分析，将视觉编码/多模态投影子模块导出至 ONNX Runtime CUDA 并建立输出一致性校验；记录 TensorRT EP 依赖缺失与 VLM `generate()` 动态解码边界，形成可复现的部署分析报告。
