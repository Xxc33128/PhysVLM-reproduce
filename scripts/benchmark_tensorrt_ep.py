from __future__ import annotations

import json

from benchmark_onnx import build_arg_parser, run_onnx_benchmark
from deployment_utils import DEFAULT_BACKEND_DIR, DEFAULT_TENSORRT_DIR, write_json


def main() -> None:
    parser = build_arg_parser(
        default_providers="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
    )
    parser.description = "Benchmark exported PhysVLM ONNX submodules with ONNX Runtime TensorRT EP."
    parser.set_defaults(output_json=str(DEFAULT_BACKEND_DIR / "onnx_trt_ep.json"))
    parser.add_argument("--trt-artifact-dir", default=str(DEFAULT_TENSORRT_DIR))
    args = parser.parse_args()

    # ORT TensorRT EP will create caches/engines only when configured by the runtime environment.
    result = run_onnx_benchmark(args, backend="onnx_trt_ep")
    result["tensorrt_artifact_dir"] = args.trt_artifact_dir
    if result["status"] == "failed":
        result["notes"] += "; TensorRT EP failure is an expected reportable boundary when unsupported ops or environment gaps appear."
    write_json(args.output_json, result)
    print(json.dumps({key: result[key] for key in ("backend", "status", "num_examples", "mean_ms", "p90_ms")}, indent=2))


if __name__ == "__main__":
    main()
