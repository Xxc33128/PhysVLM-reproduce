from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEPTH_MODES = ("normal", "black", "mismatch")
MODE_ORDER = {mode: index for index, mode in enumerate(DEPTH_MODES)}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_answer(answer: Any) -> str:
    output = str(answer).strip()
    special_tokens = ["<|endoftext|>", "<|im_end|>", "</s>"]
    changed = True
    while changed:
        changed = False
        for token in special_tokens:
            if output.startswith(token):
                output = output[len(token):].strip()
                changed = True
            if output.endswith(token):
                output = output[: -len(token)].strip()
                changed = True
    return output.lower()


def first_yesno(answer: Any) -> str:
    words = normalize_answer(answer).split()
    if not words:
        return "EMPTY"
    first = words[0].strip(".,;:!?")
    if first.startswith("yes"):
        return "Yes"
    if first.startswith("no"):
        return "No"
    return "OTHER"


def question_type(question: str) -> str:
    lowered = question.lower()
    if "reachable space" in lowered:
        return "reachable_space"
    if "directly pick" in lowered:
        return "direct_pick"
    return "other"


def robot_name(row: dict[str, Any]) -> str:
    explicit = row.get("robot") or row.get("robot_name")
    if explicit:
        return str(explicit).upper()
    haystack = " ".join(str(row.get(key, "")) for key in ("image", "depth", "id")).upper()
    for name in ("CR5", "FR5", "PANDA", "UR5"):
        if name in haystack:
            return name
    return "UNKNOWN"


def is_correct(prediction: Any, label: Any) -> bool:
    return first_yesno(prediction) == first_yesno(label)


def load_mode(path: str | Path, fallback_mode: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a predictions list")

    mode = payload.get("depth_mode") or fallback_mode
    enriched: list[dict[str, Any]] = []
    for row in rows:
        prediction = row.get("prediction", "")
        label = row.get("label", "")
        question = str(row.get("question", ""))
        enriched.append(
            {
                **row,
                "mode": mode,
                "robot": robot_name(row),
                "question_type": question_type(question),
                "label_yesno": first_yesno(label),
                "pred_yesno": first_yesno(prediction),
                "correct_clean": is_correct(prediction, label),
            }
        )
    return enriched


def summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows: list[dict[str, Any]] = []

    def sort_key(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[Any, ...]:
        key = item[0]
        values = []
        for index, value in enumerate(key):
            if group_keys[index] == "mode":
                values.append(MODE_ORDER.get(str(value), len(MODE_ORDER)))
            else:
                values.append(value)
        return tuple(values)

    for key, subset in sorted(grouped.items(), key=sort_key):
        total = len(subset)
        correct = sum(1 for row in subset if row["correct_clean"])
        output = {group_keys[index]: value for index, value in enumerate(key)}
        output.update(
            {
                "correct": correct,
                "total": total,
                "accuracy": round(correct / total, 4) if total else 0.0,
            }
        )
        summary_rows.append(output)
    return summary_rows


def add_all_robot_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    if "robot" not in group_keys:
        return summarize(rows, group_keys)

    all_rows = []
    for row in rows:
        copied = dict(row)
        copied["robot"] = "ALL"
        all_rows.append(copied)
    return summarize(rows + all_rows, group_keys)


def confusion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for mode in DEPTH_MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        for robot in ["ALL", "CR5", "FR5", "PANDA", "UR5"]:
            subset = mode_rows if robot == "ALL" else [row for row in mode_rows if row["robot"] == robot]
            counts = Counter((row["label_yesno"], row["pred_yesno"]) for row in subset)
            for label in ("Yes", "No"):
                for pred in ("Yes", "No", "OTHER", "EMPTY"):
                    count = counts.get((label, pred), 0)
                    if count:
                        output.append(
                            {
                                "mode": mode,
                                "robot": robot,
                                "label": label,
                                "prediction": pred,
                                "count": count,
                            }
                        )
    return output


def markdown_table(headers: list[str], body: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, summary: list[dict[str, Any]], by_robot: list[dict[str, Any]], confusion: list[dict[str, Any]]) -> None:
    summary_body = [
        [
            row["mode"],
            row["correct"],
            row["total"],
            f"{row['accuracy'] * 100:.2f}%",
        ]
        for row in summary
    ]

    robot_body = [
        [
            row["mode"],
            row["robot"],
            row["correct"],
            row["total"],
            f"{row['accuracy'] * 100:.2f}%",
        ]
        for row in by_robot
        if row["robot"] != "ALL"
    ]

    false_positive = {
        row["mode"]: row["count"]
        for row in confusion
        if row["robot"] == "ALL" and row["label"] == "No" and row["prediction"] == "Yes"
    }
    false_negative = {
        row["mode"]: row["count"]
        for row in confusion
        if row["robot"] == "ALL" and row["label"] == "Yes" and row["prediction"] == "No"
    }
    error_body = [
        [mode, false_positive.get(mode, 0), false_negative.get(mode, 0)]
        for mode in DEPTH_MODES
    ]

    report = f"""# S-P Map Ablation Report

## Overall Accuracy

{markdown_table(["Depth Mode", "Correct", "Total", "Accuracy"], summary_body)}

## Per-Robot Accuracy

{markdown_table(["Depth Mode", "Robot", "Correct", "Total", "Accuracy"], robot_body)}

## Error Direction

{markdown_table(["Depth Mode", "False Positive (No -> Yes)", "False Negative (Yes -> No)"], error_body)}

## Interpretation Guide

- `normal` uses the original RGB + aligned S-P Map input.
- `black` keeps the `<depth>` token but replaces the S-P Map with an all-black placeholder.
- `mismatch` keeps the RGB image but uses the next scene's S-P Map from the same robot split.
- If `normal` is stronger than `black`, the S-P Map contributes useful physical information.
- If `mismatch` drops or changes the confusion pattern, the model is sensitive to incorrect physical context rather than only RGB or language priors.
"""
    (output_dir / "ablation_report.md").write_text(report, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PhysVLM S-P Map ablation result JSON files.")
    parser.add_argument("--normal-json", required=True)
    parser.add_argument("--black-json", required=True)
    parser.add_argument("--mismatch-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = (
        load_mode(args.normal_json, "normal")
        + load_mode(args.black_json, "black")
        + load_mode(args.mismatch_json, "mismatch")
    )

    summary = summarize(rows, ["mode"])
    by_robot = add_all_robot_rows(rows, ["mode", "robot"])
    by_question_type = add_all_robot_rows(rows, ["mode", "robot", "question_type"])
    confusion = confusion_rows(rows)

    write_csv(output_dir / "ablation_summary.csv", summary, ["mode", "correct", "total", "accuracy"])
    write_csv(output_dir / "ablation_by_robot.csv", by_robot, ["mode", "robot", "correct", "total", "accuracy"])
    write_csv(
        output_dir / "ablation_by_question_type.csv",
        by_question_type,
        ["mode", "robot", "question_type", "correct", "total", "accuracy"],
    )
    write_csv(output_dir / "ablation_confusion.csv", confusion, ["mode", "robot", "label", "prediction", "count"])
    write_report(output_dir, summary, by_robot, confusion)
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
