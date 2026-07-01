from __future__ import annotations

import argparse
import base64
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from deployment_utils import add_model_args, load_predictor


def decode_base64_image(payload: str, suffix: str = ".jpg") -> str:
    if "," in payload and payload.split(",", 1)[0].startswith("data:"):
        payload = payload.split(",", 1)[1]
    data = base64.b64decode(payload)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(data)
    handle.close()
    return handle.name


def create_app(args: argparse.Namespace) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("Install FastAPI serving extras first: pip install fastapi uvicorn pydantic") from exc

    app = FastAPI(title="PhysVLM Serving", version="0.2")
    predictor = load_predictor(args)
    lock = threading.Lock()

    class PredictRequest(BaseModel):
        image_path: str | None = None
        image_base64: str | None = None
        depth_path: str | None = None
        depth_base64: str | None = None
        depth_mode: str = "normal"
        question: str
        temperature: float = 0.0
        top_p: float = 1.0
        max_new_tokens: int = 64

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path": str(args.model_path),
            "device": predictor.device,
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        image_path = request.image_path
        depth_path = request.depth_path
        temp_paths: list[str] = []

        try:
            if request.image_base64:
                image_path = decode_base64_image(request.image_base64)
                temp_paths.append(image_path)
            if request.depth_base64:
                depth_path = decode_base64_image(request.depth_base64)
                temp_paths.append(depth_path)
            if not image_path:
                raise HTTPException(status_code=400, detail="Provide image_path or image_base64.")
            if request.depth_mode not in {"normal", "black"}:
                raise HTTPException(status_code=400, detail="depth_mode must be normal or black.")
            if request.depth_mode == "black":
                depth_path = None
            elif not depth_path:
                raise HTTPException(status_code=400, detail="Provide depth_path/depth_base64 or use depth_mode=black.")
            if not Path(image_path).exists():
                raise HTTPException(status_code=404, detail=f"image_path does not exist: {image_path}")
            if depth_path and not Path(depth_path).exists():
                raise HTTPException(status_code=404, detail=f"depth_path does not exist: {depth_path}")

            started = time.perf_counter()
            with lock:
                prediction = predictor.predict(
                    image_path=image_path,
                    depth_path=depth_path,
                    question=request.question,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=request.max_new_tokens,
                )
            latency_ms = (time.perf_counter() - started) * 1000
            return {
                "answer": prediction.answer,
                "latency_ms": round(latency_ms, 3),
                "image_path": image_path,
                "depth_path": depth_path,
                "depth_mode": request.depth_mode,
            }
        finally:
            for temp_path in temp_paths:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve PhysVLM inference through FastAPI.")
    add_model_args(parser)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install uvicorn first: pip install uvicorn") from exc
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
