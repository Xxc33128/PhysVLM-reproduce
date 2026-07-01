from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from deployment_utils import (
    DEFAULT_BACKEND_DIR,
    add_dataset_args,
    add_model_args,
    build_mismatch_depth_map,
    build_rows,
    cuda_device_name,
    cuda_peak_memory_gb,
    cuda_synchronize,
    deployment_result,
    load_predictor,
    row_paths,
    write_json,
)


def graph_break_count(torch_module: Any) -> int | None:
    dynamo = getattr(torch_module, "_dynamo", None)
    if dynamo is None:
        return None
    utils = getattr(dynamo, "utils", None)
    counters = getattr(utils, "counters", None)
    if counters is None:
        return None
    graph_breaks = counters.get("graph_break", {})
    return int(sum(graph_breaks.values())) if hasattr(graph_breaks, "values") else None


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    rows, selected_rows = build_rows(args)
    predictor = load_predictor(args)
    torch_module = predictor.torch
    device = predictor.device

    if not hasattr(torch_module, "compile"):
        return deployment_result(
            backend="torch_compile",
            status="failed",
            timings_ms=[],
            num_examples=0,
            cuda_peak_memory_gb_value=None,
            notes="torch.compile is unavailable in this PyTorch build.",
        )

    original_generate = predictor.model.generate
    compile_started = time.perf_counter()
    try:
        predictor.model.generate = torch_module.compile(
            original_generate,
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
            dynamic=args.dynamic,
        )
    except Exception as exc:
        return deployment_result(
            backend="torch_compile",
            status="failed",
            timings_ms=[],
            num_examples=0,
            cuda_peak_memory_gb_value=None,
            notes=f"torch.compile setup failed: {type(exc).__name__}: {exc}",
        )

    compile_setup_s = time.perf_counter() - compile_started
    mismatch_depth_map = build_mismatch_depth_map(rows, args.data_root) if args.depth_mode == "mismatch" else {}

    if device == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    graph_breaks_before = graph_break_count(torch_module)
    warmup_rows = selected_rows[: args.warmup]
    warmup_started = time.perf_counter()
    try:
        for row in warmup_rows:
            image_path, depth_path, question = row_paths(args, row, mismatch_depth_map)
            predictor.predict(
                image_path=image_path,
                depth_path=depth_path,
                question=question,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
        cuda_synchronize(torch_module, device)
    except Exception as exc:
        return deployment_result(
            backend="torch_compile",
            status="failed",
            timings_ms=[],
            num_examples=0,
            cuda_peak_memory_gb_value=cuda_peak_memory_gb(torch_module, device),
            notes=f"torch.compile warmup failed: {type(exc).__name__}: {exc}",
            extra={"compile_setup_s": round(compile_setup_s, 3)},
        )
    warmup_s = time.perf_counter() - warmup_started

    timings_ms: list[float] = []
    samples: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected_rows):
            image_path, depth_path, question = row_paths(args, row, mismatch_depth_map)
            cuda_synchronize(torch_module, device)
            started = time.perf_counter()
            prediction = predictor.predict(
                image_path=image_path,
                depth_path=depth_path,
                question=question,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            cuda_synchronize(torch_module, device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings_ms.append(elapsed_ms)
            if len(samples) < args.keep_samples:
                samples.append(
                    {
                        "index": index,
                        "image": image_path,
                        "depth": depth_path,
                        "question": question,
                        "answer": prediction.answer,
                        "latency_ms": round(elapsed_ms, 3),
                    }
                )
    except Exception as exc:
        return deployment_result(
            backend="torch_compile",
            status="failed",
            timings_ms=timings_ms,
            num_examples=len(timings_ms),
            cuda_peak_memory_gb_value=cuda_peak_memory_gb(torch_module, device),
            notes=f"torch.compile benchmark failed after {len(timings_ms)} examples: {type(exc).__name__}: {exc}",
            extra={"compile_setup_s": round(compile_setup_s, 3), "warmup_s": round(warmup_s, 3)},
        )

    graph_breaks_after = graph_break_count(torch_module)
    graph_break_delta = None
    if graph_breaks_before is not None and graph_breaks_after is not None:
        graph_break_delta = graph_breaks_after - graph_breaks_before

    notes = [
        f"GPU/device: {cuda_device_name(torch_module, device)}",
        f"compile_mode={args.compile_mode}",
        f"fullgraph={args.fullgraph}",
        f"dynamic={args.dynamic}",
        f"compile_setup_s={compile_setup_s:.3f}",
        f"warmup_s={warmup_s:.3f}",
    ]
    if graph_break_delta is not None:
        notes.append(f"graph_break_delta={graph_break_delta}")

    return deployment_result(
        backend="torch_compile",
        status="success",
        timings_ms=timings_ms,
        num_examples=len(timings_ms),
        cuda_peak_memory_gb_value=cuda_peak_memory_gb(torch_module, device),
        notes="; ".join(notes),
        extra={
            "model_path": str(args.model_path),
            "qa_json": str(args.qa_json),
            "data_root": str(args.data_root),
            "depth_mode": args.depth_mode,
            "max_new_tokens": args.max_new_tokens,
            "compile_setup_s": round(compile_setup_s, 3),
            "warmup_s": round(warmup_s, 3),
            "graph_break_delta": graph_break_delta,
            "samples": samples,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PhysVLM model.generate() under torch.compile.")
    add_model_args(parser)
    add_dataset_args(parser, default_limit=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--keep-samples", type=int, default=3)
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_BACKEND_DIR / "torch_compile.json"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = benchmark(args)
    write_json(args.output_json, result)
    print(json.dumps({key: result[key] for key in ("backend", "status", "num_examples", "mean_ms", "p90_ms")}, indent=2))


if __name__ == "__main__":
    main()
