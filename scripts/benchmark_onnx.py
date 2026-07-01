from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from deployment_utils import (
    DEFAULT_BACKEND_DIR,
    DEFAULT_ONNX_DIR,
    DEPLOYMENT_MODULES,
    add_dataset_args,
    add_model_args,
    build_dummy_input,
    build_mismatch_depth_map,
    build_module_input_from_row,
    build_rows,
    deployment_result,
    gpu_memory_used_gb,
    module_input_name,
    parse_providers,
    row_paths,
    tensor_to_numpy,
    load_predictor,
    write_json,
)


def prepare_inputs(args: argparse.Namespace, predictor: Any) -> list[Any]:
    if args.synthetic:
        return [tensor_to_numpy(build_dummy_input(predictor, args.module, batch_size=args.batch_size)) for _ in range(args.limit)]

    rows, selected_rows = build_rows(args)
    mismatch_depth_map = build_mismatch_depth_map(rows, args.data_root) if args.depth_mode == "mismatch" else {}
    inputs = []
    for row in selected_rows:
        image_path, depth_path, _ = row_paths(args, row, mismatch_depth_map)
        module_input = build_module_input_from_row(predictor, args.module, image_path, depth_path)
        inputs.append(tensor_to_numpy(module_input))
    return inputs


def run_onnx_benchmark(args: argparse.Namespace, backend: str = "onnx_cuda") -> dict[str, Any]:
    import onnxruntime as ort

    predictor = load_predictor(args)
    onnx_path = Path(args.onnx_path or (DEFAULT_ONNX_DIR / f"{args.module}.onnx"))
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    providers = parse_providers(args.providers)
    session_options = ort.SessionOptions()
    if args.graph_optimization_level:
        session_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, args.graph_optimization_level)

    provider_options = None
    if "TensorrtExecutionProvider" in providers and hasattr(args, "trt_artifact_dir"):
        trt_artifact_dir = Path(args.trt_artifact_dir)
        trt_artifact_dir.mkdir(parents=True, exist_ok=True)
        provider_options = []
        for provider in providers:
            if provider == "TensorrtExecutionProvider":
                provider_options.append(
                    {
                        "trt_engine_cache_enable": "1",
                        "trt_engine_cache_path": str(trt_artifact_dir),
                        "trt_timing_cache_enable": "1",
                        "trt_timing_cache_path": str(trt_artifact_dir),
                        "trt_fp16_enable": "1",
                    }
                )
            else:
                provider_options.append({})

    try:
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )
    except Exception as exc:
        return deployment_result(
            backend=backend,
            status="failed",
            timings_ms=[],
            num_examples=0,
            cuda_peak_memory_gb_value=gpu_memory_used_gb(),
            notes=f"Failed to create ONNX Runtime session: {type(exc).__name__}: {exc}",
            extra={"onnx_path": str(onnx_path), "providers_requested": providers, "provider_options": provider_options},
        )

    active_providers = session.get_providers()
    if backend == "onnx_trt_ep" and "TensorrtExecutionProvider" not in active_providers:
        return deployment_result(
            backend=backend,
            status="failed",
            timings_ms=[],
            num_examples=0,
            cuda_peak_memory_gb_value=gpu_memory_used_gb(),
            notes=(
                "TensorRTExecutionProvider was requested but not active; ONNX Runtime fell back to "
                f"{active_providers}. Install matching TensorRT libraries and make sure libnvinfer is "
                "visible through LD_LIBRARY_PATH before claiming TensorRT results."
            ),
            extra={
                "onnx_path": str(onnx_path),
                "providers_requested": providers,
                "providers_active": active_providers,
                "provider_options": provider_options,
            },
        )

    input_name = args.input_name or module_input_name(args.module)
    inputs = prepare_inputs(args, predictor)
    if not inputs:
        raise ValueError("No ONNX benchmark inputs prepared.")

    warmup_inputs = inputs[: max(0, args.warmup)]
    try:
        for item in warmup_inputs:
            session.run(None, {input_name: item})
    except Exception as exc:
        return deployment_result(
            backend=backend,
            status="failed",
            timings_ms=[],
            num_examples=0,
            cuda_peak_memory_gb_value=gpu_memory_used_gb(),
            notes=f"ONNX Runtime warmup failed: {type(exc).__name__}: {exc}",
            extra={"onnx_path": str(onnx_path), "providers_active": session.get_providers()},
        )

    timings_ms: list[float] = []
    peak_memory = gpu_memory_used_gb()
    try:
        for item in inputs:
            started = time.perf_counter()
            session.run(None, {input_name: item})
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings_ms.append(elapsed_ms)
            current_memory = gpu_memory_used_gb()
            if current_memory is not None:
                peak_memory = current_memory if peak_memory is None else max(peak_memory, current_memory)
    except Exception as exc:
        return deployment_result(
            backend=backend,
            status="failed",
            timings_ms=timings_ms,
            num_examples=len(timings_ms),
            cuda_peak_memory_gb_value=peak_memory,
            notes=f"ONNX Runtime benchmark failed after {len(timings_ms)} examples: {type(exc).__name__}: {exc}",
            extra={"onnx_path": str(onnx_path), "providers_active": session.get_providers()},
        )

    notes = (
        f"module={args.module}; onnx_path={onnx_path}; providers_active={session.get_providers()}; "
        "memory is sampled with nvidia-smi and may include non-ORT process memory"
    )
    return deployment_result(
        backend=backend,
        status="success",
        timings_ms=timings_ms,
        num_examples=len(timings_ms),
        cuda_peak_memory_gb_value=peak_memory,
        notes=notes,
        extra={
            "module": args.module,
            "onnx_path": str(onnx_path),
            "providers_requested": providers,
            "providers_active": session.get_providers(),
            "provider_options": provider_options,
            "synthetic": args.synthetic,
            "depth_mode": None if args.synthetic else args.depth_mode,
        },
    )


def build_arg_parser(default_providers: str = "CUDAExecutionProvider,CPUExecutionProvider") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark exported PhysVLM ONNX submodules with ONNX Runtime.")
    add_model_args(parser)
    add_dataset_args(parser, default_limit=50)
    parser.add_argument("--module", choices=DEPLOYMENT_MODULES, default="vision_tower")
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--providers", default=default_providers)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--graph-optimization-level",
        default="ORT_ENABLE_ALL",
        choices=["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"],
    )
    parser.add_argument("--output-json", default=str(DEFAULT_BACKEND_DIR / "onnx_cuda.json"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_onnx_benchmark(args, backend="onnx_cuda")
    write_json(args.output_json, result)
    print(json.dumps({key: result[key] for key in ("backend", "status", "num_examples", "mean_ms", "p90_ms")}, indent=2))


if __name__ == "__main__":
    main()
