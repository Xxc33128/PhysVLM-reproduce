# PhysVLM Inference Profiling Report

## Configuration

- GPU/device: NVIDIA A100-SXM4-40GB
- Depth mode: `normal`
- 4-bit loading: False
- 8-bit loading: False
- Flash attention: False
- Examples: 50
- Max new tokens: 64

## Latency

| Metric | Value |
| --- | ---: |
| Mean latency | 165.274 ms |
| Median latency | 155.228 ms |
| P90 latency | 161.754 ms |
| Min latency | 152.496 ms |
| Max latency | 619.177 ms |
| Examples / second | 6.0505 |
| Generated tokens / second | 6.0505 |
| Peak CUDA memory | 8.41 GB |

## Notes

This benchmark measures end-to-end single-sample inference through `PhysVLMPredictor.predict()`, including image loading, RGB/S-P preprocessing, prompt tokenization, and `model.generate()`. It is intended as a deployment-oriented profiling baseline rather than an ONNX/TensorRT export claim.
