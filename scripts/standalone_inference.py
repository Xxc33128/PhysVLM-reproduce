from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_MODEL_NAME = "physvlm-qwen2"
DEFAULT_CONV_MODE = "qwen2"
DEFAULT_SYSTEM_PROMPT_PREFIX = "<image>\n<depth>\n"


@dataclass
class Prediction:
    image_path: str
    depth_path: str | None
    question: str
    answer: str
    prompt: str
    model_path: str


def default_physvlm_root() -> Path:
    return Path(__file__).resolve().parents[2] / "github_repos" / "PhysVLM" / "physvlm-main"


def add_physvlm_to_path(physvlm_root: str | Path) -> Path:
    root = Path(physvlm_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"PhysVLM root does not exist: {root}")
    sys.path.insert(0, str(root))
    return root


def choose_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def black_depth_like(image: Image.Image) -> Image.Image:
    return Image.new("RGB", image.size, (0, 0, 0))


def normalize_stop_text(text: str, stop_str: str | None) -> str:
    output = text.strip()
    if stop_str and output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()
    if "ASSISTANT:" in output:
        output = output.split("ASSISTANT:", 1)[-1].strip()
    return output


def parse_bbox(answer: str, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    match = re.fullmatch(
        r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
        answer.strip(),
    )
    if not match:
        return None

    values = [float(v) for v in match.groups()]
    width, height = image_size
    if all(0.0 <= v <= 1.0 for v in values):
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]

    x1, y1, x2, y2 = [round(v) for v in values]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def save_visualization(image_path: str | Path, answer: str, output_path: str | Path) -> None:
    image = load_rgb(image_path)
    bbox = parse_bbox(answer, image.size)
    if bbox is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, outline=(0, 255, 0), width=4)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


class PhysVLMPredictor:
    def __init__(
        self,
        model_path: str | Path,
        physvlm_root: str | Path | None = None,
        model_base: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        conv_mode: str = DEFAULT_CONV_MODE,
        device: str = "auto",
        load_8bit: bool = False,
        load_4bit: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        root = add_physvlm_to_path(physvlm_root or default_physvlm_root())
        self.physvlm_root = root
        self.model_path = str(Path(model_path).expanduser())
        self.model_base = model_base
        self.model_name = model_name
        self.conv_mode = conv_mode
        self.device = choose_device(device)

        if self.device != "cuda" and (load_8bit or load_4bit):
            raise ValueError("8-bit/4-bit loading is only supported on CUDA in this reproduction script.")

        import torch
        from physvlm.conversation import SeparatorStyle, conv_templates
        from physvlm.model.builder import load_pretrained_model

        self.torch = torch
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates

        loaded = load_pretrained_model(
            model_path=self.model_path,
            model_base=self.model_base,
            model_name=self.model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device=self.device,
            use_flash_attn=use_flash_attn,
        )
        if loaded is None:
            raise RuntimeError(
                "load_pretrained_model returned None. Apply the local builder.py compatibility patch first."
            )

        self.tokenizer, self.model, self.image_processor, self.context_len = loaded
        self.model.eval()

    def build_prompt(self, question: str) -> tuple[str, str | None]:
        conv = self.conv_templates[self.conv_mode].copy()
        user_message = DEFAULT_SYSTEM_PROMPT_PREFIX + question.strip()
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], None)
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        return conv.get_prompt(), stop_str

    def preprocess_pair(
        self,
        image_path: str | Path,
        depth_path: str | Path | None,
    ) -> tuple[Any, Any, list[tuple[int, int]], Image.Image]:
        from physvlm.mm_utils import process_images

        image = load_rgb(image_path)
        depth = load_rgb(depth_path) if depth_path else black_depth_like(image)
        image_sizes = [image.size]

        image_tensor = process_images([image], self.image_processor, self.model.config)
        depth_tensor = process_images([depth], self.image_processor, self.model.config)

        dtype = next(self.model.parameters()).dtype
        device = self.model.device

        if isinstance(image_tensor, list):
            image_tensor = [item.to(device=device, dtype=dtype) for item in image_tensor]
        else:
            image_tensor = image_tensor.to(device=device, dtype=dtype)

        if isinstance(depth_tensor, list):
            depth_tensor = [item.to(device=device, dtype=dtype) for item in depth_tensor]
        else:
            depth_tensor = depth_tensor.to(device=device, dtype=dtype)

        return image_tensor, depth_tensor, image_sizes, image

    def predict(
        self,
        image_path: str | Path,
        question: str,
        depth_path: str | Path | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 128,
    ) -> Prediction:
        from physvlm.constants import IMAGE_TOKEN_INDEX
        from physvlm.mm_utils import tokenizer_image_token

        prompt, stop_str = self.build_prompt(question)
        image_tensor, depth_tensor, image_sizes, _ = self.preprocess_pair(image_path, depth_path)

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        do_sample = temperature > 0.001
        generation_kwargs: dict[str, Any] = {
            "inputs": input_ids,
            "images": image_tensor,
            "depth_images": depth_tensor,
            "image_sizes": image_sizes,
            "do_sample": do_sample,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with self.torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)

        # Only decode newly generated tokens (skip the input prompt)
        new_token_ids = output_ids[0, input_ids.shape[1]:]
        answer = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        answer = normalize_stop_text(answer, stop_str)
        return Prediction(
            image_path=str(image_path),
            depth_path=str(depth_path) if depth_path else None,
            question=question,
            answer=answer,
            prompt=prompt,
            model_path=self.model_path,
        )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone PhysVLM inference without FastAPI server.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--physvlm-root", default=str(default_physvlm_root()))
    parser.add_argument("--conv-mode", default=DEFAULT_CONV_MODE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--depth-path", default=None)
    parser.add_argument("--question", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--visualization-path", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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
    prediction = predictor.predict(
        image_path=args.image_path,
        depth_path=args.depth_path,
        question=args.question,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    print(prediction.answer)
    if args.output_json:
        write_json(args.output_json, asdict(prediction))
    if args.visualization_path:
        save_visualization(args.image_path, prediction.answer, args.visualization_path)


if __name__ == "__main__":
    main()
