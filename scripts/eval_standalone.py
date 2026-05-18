from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Dynamic import so the script works regardless of cwd or sys.path.
_SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "standalone_inference", _SCRIPT_DIR / "standalone_inference.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

DEFAULT_CONV_MODE = _mod.DEFAULT_CONV_MODE
DEFAULT_MODEL_NAME = _mod.DEFAULT_MODEL_NAME
PhysVLMPredictor = _mod.PhysVLMPredictor
default_physvlm_root = _mod.default_physvlm_root
write_json = _mod.write_json


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_data_path(data_root: str | Path | None, maybe_relative: str | None) -> str | None:
    if not maybe_relative:
        return None
    candidate = Path(maybe_relative).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if data_root is None:
        return str(candidate)
    return str(Path(data_root).expanduser() / candidate)


def robot_name(row: dict[str, Any]) -> str:
    explicit = row.get("robot") or row.get("robot_name")
    if explicit:
        return str(explicit).upper()

    haystack = " ".join(str(row.get(key, "")) for key in ("image", "depth", "scene", "id")).upper()
    for name in ("UR5", "CR5", "FR5", "PANDA", "UR3", "XARM6"):
        if name in haystack:
            return name
    return "UNKNOWN"


def normalize_answer(answer: Any) -> str:
    return str(answer).strip().lower()


def _first_word(text: str) -> str:
    """Extract the first word, stripping punctuation."""
    word = normalize_answer(text).split()[0] if normalize_answer(text).split() else ""
    return word.rstrip(".,;:!?")


def is_correct(prediction: str, label: str) -> bool:
    return _first_word(prediction) == _first_word(label)


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = defaultdict(int)
    corrects: dict[str, int] = defaultdict(int)

    for item in predictions:
        robot = item["robot"]
        totals[robot] += 1
        totals["ALL"] += 1
        if item["correct"]:
            corrects[robot] += 1
            corrects["ALL"] += 1

    metrics = {}
    for robot in sorted(totals):
        total = totals[robot]
        correct = corrects[robot]
        metrics[robot] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }
    return metrics


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_json(args.qa_json)
    if not isinstance(rows, list):
        raise ValueError("QA JSON must contain a list of examples.")

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

    selected = rows[: args.limit] if args.limit else rows
    predictions: list[dict[str, Any]] = []

    for index, row in enumerate(selected):
        image_path = resolve_data_path(args.data_root, row.get("image") or row.get("image_path"))
        depth_path = resolve_data_path(args.data_root, row.get("depth") or row.get("depth_path"))
        question = row.get("question") or row.get("query")
        label = row.get("answer") or row.get("label")

        if not image_path or not question or label is None:
            raise ValueError(f"Example {index} is missing image/question/answer fields: {row}")

        prediction = predictor.predict(
            image_path=image_path,
            depth_path=depth_path,
            question=str(question),
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

        pred_row = {
            "index": index,
            "robot": robot_name(row),
            "image": image_path,
            "depth": depth_path,
            "question": question,
            "label": label,
            "prediction": prediction.answer,
            "correct": is_correct(prediction.answer, str(label)),
            "raw": row,
        }
        predictions.append(pred_row)

    return {
        "qa_json": str(args.qa_json),
        "data_root": str(args.data_root) if args.data_root else None,
        "model_path": str(args.model_path),
        "num_examples": len(predictions),
        "metrics": summarize(predictions),
        "predictions": predictions,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline PhysVLM EQA-phys evaluation.")
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate(args)
    write_json(args.output_json, result)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
