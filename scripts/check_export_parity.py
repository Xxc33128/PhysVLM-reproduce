from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deployment_utils import (
    DEFAULT_BACKEND_DIR,
    DEFAULT_ONNX_DIR,
    DEPLOYMENT_MODULES,
    add_model_args,
    build_dummy_input,
    get_deployment_module,
    module_input_name,
    output_metrics,
    parse_providers,
    tensor_to_numpy,
    load_predictor,
    write_json,
)


def check_parity(args: argparse.Namespace) -> dict[str, Any]:
    import onnxruntime as ort

    predictor = load_predictor(args)
    torch = predictor.torch
    module = get_deployment_module(predictor, args.module)
    sample_input = build_dummy_input(predictor, args.module, batch_size=args.batch_size)

    onnx_path = Path(args.onnx_path or (DEFAULT_ONNX_DIR / f"{args.module}.onnx"))
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    providers = parse_providers(args.providers)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = args.input_name or module_input_name(args.module)

    with torch.inference_mode():
        torch_output = module(sample_input)
    ort_outputs = session.run(None, {input_name: tensor_to_numpy(sample_input)})
    metrics = output_metrics(torch_output, ort_outputs[0])
    passed = bool(metrics["cosine_similarity"] >= args.min_cosine)

    result = {
        "module": args.module,
        "status": "success" if passed else "failed",
        "onnx_path": str(onnx_path),
        "providers_requested": providers,
        "providers_active": session.get_providers(),
        "input_name": input_name,
        "batch_size": args.batch_size,
        "min_cosine": args.min_cosine,
        "metrics": metrics,
        "notes": "Submodule parity only; this is not end-to-end PhysVLM answer parity.",
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check PyTorch vs ONNX Runtime parity for exported PhysVLM submodules.")
    add_model_args(parser)
    parser.add_argument("--module", choices=DEPLOYMENT_MODULES, default="vision_tower")
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output-json", default=str(DEFAULT_BACKEND_DIR / "parity_vision_tower.json"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.output_json.endswith("parity_vision_tower.json") and args.module != "vision_tower":
        args.output_json = str(DEFAULT_BACKEND_DIR / f"parity_{args.module}.json")
    result = check_parity(args)
    write_json(args.output_json, result)
    print(json.dumps({"module": result["module"], "status": result["status"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
