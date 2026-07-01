from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from typing import Any

from deployment_utils import (
    DEFAULT_BACKEND_DIR,
    add_dataset_args,
    build_mismatch_depth_map,
    build_rows,
    deployment_result,
    row_paths,
    summarize_timings,
    write_json,
)


def request_payload(args: argparse.Namespace, row: dict[str, Any], mismatch_depth_map: dict[str, str]) -> dict[str, Any]:
    image_path, depth_path, question = row_paths(args, row, mismatch_depth_map)
    payload = {
        "image_path": image_path,
        "question": question,
        "depth_mode": args.depth_mode if args.depth_mode in {"normal", "black"} else "normal",
        "max_new_tokens": args.max_new_tokens,
    }
    if payload["depth_mode"] == "normal":
        payload["depth_path"] = depth_path
    return payload


def post_predict(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    import requests

    started = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    body = response.json()
    return {
        "ok": True,
        "latency_ms": elapsed_ms,
        "server_latency_ms": body.get("latency_ms"),
        "answer": body.get("answer"),
    }


def run_requests(endpoint: str, payloads: list[dict[str, Any]], timeout: float, concurrency: int) -> list[dict[str, Any]]:
    if concurrency <= 1:
        results = []
        for payload in payloads:
            try:
                results.append(post_predict(endpoint, payload, timeout))
            except Exception as exc:
                results.append({"ok": False, "latency_ms": 0.0, "error": f"{type(exc).__name__}: {exc}"})
        return results

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(post_predict, endpoint, payload, timeout) for payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"ok": False, "latency_ms": 0.0, "error": f"{type(exc).__name__}: {exc}"})
    return results


def summarize_api(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [item for item in results if item.get("ok")]
    latencies = [float(item["latency_ms"]) for item in ok_results]
    server_latencies = [float(item["server_latency_ms"]) for item in ok_results if item.get("server_latency_ms") is not None]
    return {
        "success_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "client_latency": summarize_timings(latencies),
        "server_latency_mean_ms": round(statistics.mean(server_latencies), 3) if server_latencies else 0.0,
        "errors": [item.get("error") for item in results if not item.get("ok")][:5],
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import requests

    rows, selected_rows = build_rows(args)
    selected_rows = selected_rows[: args.limit]
    mismatch_depth_map = build_mismatch_depth_map(rows, args.data_root) if args.depth_mode == "mismatch" else {}
    payloads = [request_payload(args, row, mismatch_depth_map) for row in selected_rows]

    health_url = args.base_url.rstrip("/") + "/health"
    endpoint = args.base_url.rstrip("/") + "/predict"
    health = requests.get(health_url, timeout=args.timeout)
    health.raise_for_status()

    sequential = run_requests(endpoint, payloads, args.timeout, concurrency=1)
    concurrency_2 = run_requests(endpoint, payloads, args.timeout, concurrency=2)

    sequential_summary = summarize_api(sequential)
    concurrency_summary = summarize_api(concurrency_2)
    latencies = [float(item["latency_ms"]) for item in sequential if item.get("ok")]
    success_count = sequential_summary["success_count"] + concurrency_summary["success_count"]
    error_count = sequential_summary["error_count"] + concurrency_summary["error_count"]
    notes = (
        f"base_url={args.base_url}; sequential_success={sequential_summary['success_count']}; "
        f"concurrency2_success={concurrency_summary['success_count']}; API latency includes HTTP and serving overhead"
    )
    return deployment_result(
        backend="fastapi",
        status="success" if success_count else "failed",
        timings_ms=latencies,
        num_examples=len(payloads),
        cuda_peak_memory_gb_value=None,
        notes=notes,
        extra={
            "base_url": args.base_url,
            "depth_mode": args.depth_mode,
            "sequential": sequential_summary,
            "concurrency_2": concurrency_summary,
            "success_count": success_count,
            "error_count": error_count,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the PhysVLM FastAPI serving endpoint.")
    add_dataset_args(parser, default_limit=10)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", default=str(DEFAULT_BACKEND_DIR / "fastapi.json"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = benchmark(args)
    write_json(args.output_json, result)
    print(json.dumps({key: result[key] for key in ("backend", "status", "num_examples", "mean_ms", "p90_ms")}, indent=2))


if __name__ == "__main__":
    main()
