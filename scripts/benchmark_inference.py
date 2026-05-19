from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


_SCRIPT_DIR = Path(__file__).resolve().parent
_standalone_spec = importlib.util.spec_from_file_location(
    "standalone_inference", _SCRIPT_DIR / "standalone_inference.py"
)
_standalone = importlib.util.module_from_spec(_standalone_spec)
sys.modules[_standalone_spec.name] = _standalone
_standalone_spec.loader.exec_module(_standalone)

_eval_spec = importlib.util.spec_from_file_location("eval_standalone", _SCRIPT_DIR / "eval_standalone.py")
_eval = importlib.util.module_from_spec(_eval_spec)
sys.modules[_eval_spec.name] = _eval
_eval_spec.loader.exec_module(_eval)

DEFAULT_CONV_MODE = _standalone.DEFAULT_CONV_MODE
DEFAULT_MODEL_NAME = _standalone.DEFAULT_MODEL_NAME
PhysVLMPredictor = _standalone.PhysVLMPredictor
default_physvlm_root = _standalone.default_physvlm_root
write_json = _standalone.write_json
read_json = _eval.read_json
resolve_data_path = _eval.resolve_data_path
build_mismatch_depth_map = _eval.build_mismatch_depth_map
select_depth_path = _eval.select_depth_path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def cuda_synchronize(torch_module: Any, device: str) -> None:
    if device == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def cuda_peak_memory_gb(torch_module: Any, device: str) -> float | None:
    if device != "cuda" or not torch_module.cuda.is_available():
        return None
    return round(torch_module.cuda.max_memory_allocated() / 1024**3, 4)


def cuda_device_name(torch_module: Any, device: str) -> str:
    if device == "cuda" and torch_module.cuda.is_available():
        return torch_module.cuda.get_device_name(0)
    return device


def select_rows(rows: list[dict[str, Any]], limit: int, stride: int) -> list[dict[str, Any]]:
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    selected = rows[::stride]
    if limit:
        selected = selected[:limit]
    return selected


def summarize_timings(timings_ms: list[float]) -> dict[str, float]:
    return {
        "count": len(timings_ms),
        "mean_ms": round(statistics.mean(timings_ms), 3) if timings_ms else 0.0,
        "median_ms": round(statistics.median(timings_ms), 3) if timings_ms else 0.0,
        "p90_ms": round(percentile(timings_ms, 90), 3),
        "min_ms": round(min(timings_ms), 3) if timings_ms else 0.0,
        "max_ms": round(max(timings_ms), 3) if timings_ms else 0.0,
    }


def evaluate_depth_path(
    args: argparse.Namespace,
    row: dict[str, Any],
    image_path: str,
    original_depth_path: str | None,
    mismatch_depth_map: dict[str, str],
) -> str | None:
    used_depth = select_depth_path(args.depth_mode, image_path, original_depth_path, mismatch_depth_map)
    if args.depth_mode == "normal" and (not used_depth or not Path(used_depth).exists()):
        raise FileNotFoundError(f"Depth path does not exist: {used_depth}")
    if args.depth_mode == "mismatch" and (not used_depth or not Path(used_depth).exists()):
        raise FileNotFoundError(f"Mismatch depth path does not exist: {used_depth}")
    return used_depth


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_json(args.qa_json)
    if not isinstance(rows, list):
        raise ValueError("QA JSON must contain a list of examples.")

    selected_rows = select_rows(rows, args.limit, args.stride)
    if not selected_rows:
        raise ValueError("No rows selected for benchmarking.")

    predictor = PhysVLMPredictor(
        model_path=args.model_path,
        physvlm_root=args.physvlm_root,
        model_base=args.model_base,
        model_name=args.model_name,
        conv_mode=args.conv_mode,
        device=args.device,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        use_flash_attn=args.use_flash_attn,
    )
    torch_module = predictor.torch
    device = predictor.device

    mismatch_depth_map = build_mismatch_depth_map(rows, args.data_root) if args.depth_mode == "mismatch" else {}

    if device == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    timings_ms: list[float] = []
    generated_tokens: list[int] = []
    samples: list[dict[str, Any]] = []

    for index, row in enumerate(selected_rows):
        image_path = resolve_data_path(args.data_root, row.get("image") or row.get("image_path"))
        original_depth_path = resolve_data_path(args.data_root, row.get("depth") or row.get("depth_path"))
        question = row.get("question") or row.get("query")
        if not image_path or not question:
            raise ValueError(f"Example {index} is missing image/question fields: {row}")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        used_depth_path = evaluate_depth_path(args, row, image_path, original_depth_path, mismatch_depth_map)

        cuda_synchronize(torch_module, device)
        started = time.perf_counter()
        prediction = predictor.predict(
            image_path=image_path,
            depth_path=used_depth_path,
            question=str(question),
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        cuda_synchronize(torch_module, device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        timings_ms.append(elapsed_ms)
        token_count = len(predictor.tokenizer(prediction.answer).input_ids)
        generated_tokens.append(token_count)

        if len(samples) < args.keep_samples:
            samples.append(
                {
                    "index": index,
                    "image": image_path,
                    "depth": original_depth_path,
                    "used_depth": used_depth_path,
                    "question": question,
                    "answer": prediction.answer,
                    "latency_ms": round(elapsed_ms, 3),
                    "generated_tokens": token_count,
                }
            )

    total_time_s = sum(timings_ms) / 1000
    total_tokens = sum(generated_tokens)
    result = {
        "model_path": str(args.model_path),
        "qa_json": str(args.qa_json),
        "data_root": str(args.data_root) if args.data_root else None,
        "depth_mode": args.depth_mode,
        "device": device,
        "gpu": cuda_device_name(torch_module, device),
        "load_8bit": args.load_8bit,
        "load_4bit": args.load_4bit,
        "use_flash_attn": args.use_flash_attn,
        "num_examples": len(selected_rows),
        "max_new_tokens": args.max_new_tokens,
        "latency": summarize_timings(timings_ms),
        "total_time_s": round(total_time_s, 3),
        "examples_per_second": round(len(selected_rows) / total_time_s, 4) if total_time_s else 0.0,
        "generated_tokens_total": total_tokens,
        "generated_tokens_per_second": round(total_tokens / total_time_s, 4) if total_time_s else 0.0,
        "generated_tokens_mean": round(statistics.mean(generated_tokens), 3) if generated_tokens else 0.0,
        "cuda_peak_memory_gb": cuda_peak_memory_gb(torch_module, device),
        "samples": samples,
    }
    return result


def write_markdown_report(result: dict[str, Any], output_path: str | Path) -> None:
    peak_memory = result["cuda_peak_memory_gb"]
    peak_memory_text = "n/a" if peak_memory is None else f"{peak_memory:.2f} GB"
    report = f"""# PhysVLM Inference Profiling Report

## Configuration

- GPU/device: {result['gpu']}
- Depth mode: `{result['depth_mode']}`
- 4-bit loading: {result['load_4bit']}
- 8-bit loading: {result['load_8bit']}
- Flash attention: {result['use_flash_attn']}
- Examples: {result['num_examples']}
- Max new tokens: {result['max_new_tokens']}

## Latency

| Metric | Value |
| --- | ---: |
| Mean latency | {result['latency']['mean_ms']} ms |
| Median latency | {result['latency']['median_ms']} ms |
| P90 latency | {result['latency']['p90_ms']} ms |
| Min latency | {result['latency']['min_ms']} ms |
| Max latency | {result['latency']['max_ms']} ms |
| Examples / second | {result['examples_per_second']} |
| Generated tokens / second | {result['generated_tokens_per_second']} |
| Peak CUDA memory | {peak_memory_text} |

## Notes

This benchmark measures end-to-end single-sample inference through `PhysVLMPredictor.predict()`, including image loading, RGB/S-P preprocessing, prompt tokenization, and `model.generate()`. It is intended as a deployment-oriented profiling baseline rather than an ONNX/TensorRT export claim.
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark standalone PhysVLM inference latency and memory.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--physvlm-root", default=str(default_physvlm_root()))
    parser.add_argument("--conv-mode", default=DEFAULT_CONV_MODE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--qa-json", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--depth-mode", choices=["normal", "black", "mismatch"], default="normal")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--keep-samples", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = benchmark(args)
    write_json(args.output_json, result)
    if args.output_md:
        write_markdown_report(result, args.output_md)
    print(json.dumps({key: result[key] for key in ("gpu", "depth_mode", "num_examples", "latency", "cuda_peak_memory_gb")}, indent=2))


if __name__ == "__main__":
    main()
