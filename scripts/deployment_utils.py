from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"
DEFAULT_TENSORRT_DIR = REPO_ROOT / "artifacts" / "tensorrt"
DEFAULT_BACKEND_DIR = REPO_ROOT / "results" / "deployment" / "backend_results"


def _load_script_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_standalone = _load_script_module("standalone_inference", "standalone_inference.py")
_eval = _load_script_module("eval_standalone", "eval_standalone.py")
_benchmark = _load_script_module("benchmark_inference", "benchmark_inference.py")

DEFAULT_CONV_MODE = _standalone.DEFAULT_CONV_MODE
DEFAULT_MODEL_NAME = _standalone.DEFAULT_MODEL_NAME
PhysVLMPredictor = _standalone.PhysVLMPredictor
default_physvlm_root = _standalone.default_physvlm_root
read_json = _eval.read_json
resolve_data_path = _eval.resolve_data_path
build_mismatch_depth_map = _eval.build_mismatch_depth_map
select_depth_path = _eval.select_depth_path
select_rows = _benchmark.select_rows
summarize_timings = _benchmark.summarize_timings
cuda_synchronize = _benchmark.cuda_synchronize
cuda_peak_memory_gb = _benchmark.cuda_peak_memory_gb
cuda_device_name = _benchmark.cuda_device_name


DEPLOYMENT_MODULES = ("vision_tower", "mm_projector", "mm_depth_projector", "depth_tower")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def add_model_args(parser: Any) -> None:
    parser.add_argument("--model-path", default=os.environ.get("PHYSVLM_MODEL_PATH"))
    parser.add_argument("--model-base", default=os.environ.get("PHYSVLM_MODEL_BASE"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--physvlm-root", default=os.environ.get("PHYSVLM_ROOT", str(default_physvlm_root())))
    parser.add_argument("--conv-mode", default=DEFAULT_CONV_MODE)
    parser.add_argument("--device", default=os.environ.get("PHYSVLM_DEVICE", "auto"))
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")


def add_dataset_args(parser: Any, default_limit: int = 50) -> None:
    parser.add_argument("--qa-json", default=os.environ.get("PHYSVLM_QA_JSON"))
    parser.add_argument("--data-root", default=os.environ.get("PHYSVLM_DATA_ROOT"))
    parser.add_argument("--depth-mode", choices=["normal", "black", "mismatch"], default="normal")
    parser.add_argument("--limit", type=int, default=default_limit)
    parser.add_argument("--stride", type=int, default=1)


def validate_model_args(args: Any) -> None:
    missing = []
    if not args.model_path:
        missing.append("--model-path or PHYSVLM_MODEL_PATH")
    if missing:
        raise ValueError("Missing required model configuration: " + ", ".join(missing))


def validate_dataset_args(args: Any) -> None:
    missing = []
    if not args.qa_json:
        missing.append("--qa-json or PHYSVLM_QA_JSON")
    if not args.data_root:
        missing.append("--data-root or PHYSVLM_DATA_ROOT")
    if missing:
        raise ValueError("Missing required dataset configuration: " + ", ".join(missing))


def load_predictor(args: Any) -> Any:
    validate_model_args(args)
    return PhysVLMPredictor(
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


def get_model_core(model: Any) -> Any:
    return model.get_model() if hasattr(model, "get_model") else model


def get_deployment_module(predictor: Any, module_name: str) -> Any:
    model = predictor.model
    core = get_model_core(model)
    if module_name == "vision_tower":
        module = model.get_vision_tower()
    elif module_name == "depth_tower":
        module = model.get_depth_tower()
    elif module_name == "mm_projector":
        module = core.mm_projector
    elif module_name == "mm_depth_projector":
        module = core.mm_depth_projector
    else:
        raise ValueError(f"Unsupported module: {module_name}. Choose from {', '.join(DEPLOYMENT_MODULES)}")

    module.eval()
    return module


def module_input_name(module_name: str) -> str:
    if module_name in {"vision_tower", "depth_tower"}:
        return "pixel_values"
    return "features"


def module_output_name(module_name: str) -> str:
    if module_name in {"vision_tower", "depth_tower"}:
        return "hidden_states"
    return "projected_features"


def first_parameter_dtype_device(module: Any, fallback_device: str = "cpu") -> tuple[Any, Any]:
    import torch

    for parameter in module.parameters():
        return parameter.dtype, parameter.device
    return torch.float32, torch.device(fallback_device)


def model_hidden_sizes(predictor: Any) -> tuple[int, int]:
    model = predictor.model
    config = model.config
    hidden_size = int(getattr(config, "hidden_size", 896))
    mm_hidden_size = int(getattr(config, "mm_hidden_size", 1152))
    return hidden_size, mm_hidden_size


def vision_image_size(predictor: Any, module_name: str = "vision_tower") -> int:
    tower = predictor.model.get_depth_tower() if module_name == "depth_tower" else predictor.model.get_vision_tower()
    config = getattr(tower, "config", None)
    return int(getattr(config, "image_size", 384))


def build_dummy_input(
    predictor: Any,
    module_name: str,
    batch_size: int = 1,
    dtype: Any | None = None,
    device: Any | None = None,
) -> Any:
    import torch

    module = get_deployment_module(predictor, module_name)
    default_dtype, default_device = first_parameter_dtype_device(module, predictor.device)
    dtype = dtype or default_dtype
    device = device or default_device

    if module_name in {"vision_tower", "depth_tower"}:
        image_size = vision_image_size(predictor, module_name)
        return torch.randn(batch_size, 3, image_size, image_size, dtype=dtype, device=device)

    _, mm_hidden_size = model_hidden_sizes(predictor)
    feature_dim = mm_hidden_size * 2 if module_name == "mm_depth_projector" else mm_hidden_size
    return torch.randn(batch_size, 243, feature_dim, dtype=dtype, device=device)


def ensure_batched_tensor(tensor: Any) -> Any:
    if isinstance(tensor, list):
        if not tensor:
            raise ValueError("Expected at least one tensor, got an empty list.")
        tensor = tensor[0]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 5:
        tensor = tensor.reshape(-1, *tensor.shape[-3:])
    return tensor


def build_rows(args: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_dataset_args(args)
    rows = read_json(args.qa_json)
    if not isinstance(rows, list):
        raise ValueError("QA JSON must contain a list of examples.")
    selected_rows = select_rows(rows, args.limit, args.stride)
    if not selected_rows:
        raise ValueError("No rows selected.")
    return rows, selected_rows


def row_paths(args: Any, row: dict[str, Any], mismatch_depth_map: dict[str, str]) -> tuple[str, str | None, str]:
    image_path = resolve_data_path(args.data_root, row.get("image") or row.get("image_path"))
    depth_path = resolve_data_path(args.data_root, row.get("depth") or row.get("depth_path"))
    question = row.get("question") or row.get("query") or ""
    if not image_path or not question:
        raise ValueError(f"Example is missing image/question fields: {row}")
    used_depth = select_depth_path(args.depth_mode, image_path, depth_path, mismatch_depth_map)
    return image_path, used_depth, str(question)


def build_module_input_from_row(
    predictor: Any,
    module_name: str,
    image_path: str,
    depth_path: str | None,
) -> Any:
    torch = predictor.torch
    image_tensor, depth_tensor, _, _ = predictor.preprocess_pair(image_path, depth_path)
    image_tensor = ensure_batched_tensor(image_tensor)
    depth_tensor = ensure_batched_tensor(depth_tensor)

    with torch.inference_mode():
        if module_name == "vision_tower":
            return image_tensor
        if module_name == "depth_tower":
            return depth_tensor
        image_features = predictor.model.encode_images(image_tensor)
        down_sample_image_features = predictor.model.encode_down_sample(image_features)
        if module_name == "mm_projector":
            return down_sample_image_features
        if module_name == "mm_depth_projector":
            depth_features = predictor.model.encode_depths(depth_tensor)
            down_sample_depth_features = predictor.model.encode_down_sample(depth_features)
            return torch.cat([down_sample_image_features, down_sample_depth_features], dim=-1)
    raise ValueError(f"Unsupported module: {module_name}")


def tensor_to_numpy(tensor: Any) -> Any:
    return tensor.detach().cpu().numpy()


def normalize_tensor_output(output: Any) -> Any:
    if isinstance(output, dict):
        if "last_hidden_state" in output:
            return output["last_hidden_state"]
        return next(iter(output.values()))
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def output_metrics(torch_output: Any, ort_output: Any) -> dict[str, float | list[int]]:
    import numpy as np

    torch_array = tensor_to_numpy(normalize_tensor_output(torch_output)).astype("float32")
    ort_array = normalize_tensor_output(ort_output).astype("float32")
    diff = np.abs(torch_array - ort_array)
    torch_flat = torch_array.reshape(-1)
    ort_flat = ort_array.reshape(-1)
    denom = float(np.linalg.norm(torch_flat) * np.linalg.norm(ort_flat))
    cosine = float(np.dot(torch_flat, ort_flat) / denom) if denom else 0.0
    return {
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "cosine_similarity": cosine,
        "torch_shape": list(torch_array.shape),
        "onnx_shape": list(ort_array.shape),
    }


def parse_providers(text: str | None) -> list[str]:
    if not text:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [item.strip() for item in text.split(",") if item.strip()]


def gpu_memory_used_gb() -> float | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line) / 1024)
        except ValueError:
            continue
    return round(max(values), 4) if values else None


def deployment_result(
    backend: str,
    status: str,
    timings_ms: list[float],
    num_examples: int,
    cuda_peak_memory_gb_value: float | None,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latency = summarize_timings(timings_ms)
    total_time_s = sum(timings_ms) / 1000
    result = {
        "backend": backend,
        "status": status,
        "num_examples": num_examples,
        "mean_ms": latency["mean_ms"],
        "median_ms": latency["median_ms"],
        "p90_ms": latency["p90_ms"],
        "examples_per_second": round(num_examples / total_time_s, 4) if total_time_s else 0.0,
        "cuda_peak_memory_gb": cuda_peak_memory_gb_value,
        "notes": notes,
    }
    if extra:
        result.update(extra)
    return result


def mean_or_zero(values: list[float]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0
