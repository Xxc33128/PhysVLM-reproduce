from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deployment_utils import (
    DEFAULT_ONNX_DIR,
    DEPLOYMENT_MODULES,
    add_model_args,
    build_dummy_input,
    get_deployment_module,
    module_input_name,
    module_output_name,
    normalize_tensor_output,
    tensor_to_numpy,
    load_predictor,
    write_json,
)


def export_module(args: argparse.Namespace) -> dict[str, Any]:
    predictor = load_predictor(args)
    torch = predictor.torch
    module = get_deployment_module(predictor, args.module)
    sample_input = build_dummy_input(predictor, args.module, batch_size=args.batch_size)
    input_name = module_input_name(args.module)
    output_name = module_output_name(args.module)

    output_path = Path(args.output or (Path(args.output_dir) / f"{args.module}.onnx"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata or output_path.with_suffix(".metadata.json"))

    with torch.inference_mode():
        sample_output = normalize_tensor_output(module(sample_input))

    export_kwargs = {
        "opset_version": args.opset,
        "input_names": [input_name],
        "output_names": [output_name],
        "dynamo": True,
        "external_data": True,
    }

    status = "success"
    notes = "Exported with torch.onnx.export(dynamo=True, external_data=True)."
    try:
        torch.onnx.export(module, (sample_input,), str(output_path), **export_kwargs)
    except TypeError as exc:
        if not args.allow_legacy_fallback:
            status = "failed"
            notes = f"ONNX dynamo export TypeError: {exc}"
            if not args.allow_failure:
                raise
        else:
            legacy_kwargs = {
                "opset_version": args.opset,
                "input_names": [input_name],
                "output_names": [output_name],
                "dynamic_axes": {input_name: {0: "batch"}, output_name: {0: "batch"}},
            }
            torch.onnx.export(module, (sample_input,), str(output_path), **legacy_kwargs)
            notes = f"Dynamo/export external_data kwargs unsupported; exported with legacy ONNX path after TypeError: {exc}"
    except Exception as exc:
        status = "failed"
        notes = f"ONNX export failed: {type(exc).__name__}: {exc}"
        if not args.allow_failure:
            metadata = build_metadata(args, status, notes, output_path, metadata_path, sample_input, sample_output)
            write_json(metadata_path, metadata)
            raise

    metadata = build_metadata(args, status, notes, output_path, metadata_path, sample_input, sample_output)
    write_json(metadata_path, metadata)
    return metadata


def build_metadata(
    args: argparse.Namespace,
    status: str,
    notes: str,
    output_path: Path,
    metadata_path: Path,
    sample_input: Any,
    sample_output: Any,
) -> dict[str, Any]:
    sample_input_array = tensor_to_numpy(sample_input)
    sample_output_array = tensor_to_numpy(sample_output)
    return {
        "module": args.module,
        "status": status,
        "onnx_path": str(output_path),
        "metadata_path": str(metadata_path),
        "input_name": module_input_name(args.module),
        "output_name": module_output_name(args.module),
        "input_shape": list(sample_input.shape),
        "output_shape": list(sample_output.shape),
        "input_dtype": str(sample_input.dtype),
        "output_dtype": str(sample_output.dtype),
        "input_min": float(sample_input_array.min()),
        "input_max": float(sample_input_array.max()),
        "output_min": float(sample_output_array.min()),
        "output_max": float(sample_output_array.max()),
        "opset": args.opset,
        "notes": notes,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PhysVLM deployment submodules to ONNX.")
    add_model_args(parser)
    parser.add_argument("--module", choices=DEPLOYMENT_MODULES, default="vision_tower")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-dir", default=str(DEFAULT_ONNX_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--allow-legacy-fallback", action="store_true", default=True)
    parser.add_argument("--no-legacy-fallback", dest="allow_legacy_fallback", action="store_false")
    parser.add_argument("--allow-failure", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metadata = export_module(args)
    print(json.dumps({key: metadata[key] for key in ("module", "status", "onnx_path", "input_shape", "output_shape")}, indent=2))


if __name__ == "__main__":
    main()
