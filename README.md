# PhysVLM Reproduction

This folder contains local reproduction scripts for PhysVLM without relying on the missing `start_physvlm_server.py` entrypoint.

## Files

- `notebooks/01_colab_pro_reproduction.ipynb`: self-contained Colab Pro runbook.
- `scripts/standalone_inference.py`: direct RGB + S-P/depth map inference.
- `scripts/eval_standalone.py`: offline EQA-phys style evaluation over a QA JSON file.

The scripts expect the official PhysVLM source tree at:

```text
../github_repos/PhysVLM/physvlm-main
```

Override this with `--physvlm-root` if the source tree is elsewhere.

## Example Commands

```bash
python scripts/standalone_inference.py \
  --model-path ../github_repos/PhysVLM/physvlm-main/checkpoints/PhysVLM-Qwen2.5-3B \
  --image-path /path/to/rgb.png \
  --depth-path /path/to/sp_map.png \
  --question "Can the robot reach the red cup on the table?" \
  --output-json results/demo_prediction.json \
  --visualization-path results/visualization/demo_prediction.png
```

```bash
python scripts/eval_standalone.py \
  --model-path ../github_repos/PhysVLM/physvlm-main/checkpoints/PhysVLM-Qwen2.5-3B \
  --qa-json /path/to/phys_bench_sim_qas.json \
  --data-root /path/to/EQA-phys-simulator/output \
  --output-json results/eval_results.json
```

## Notes

- With Colab Pro, prefer A100 or L4. Use 4-bit loading on L4/T4; fp16 is mainly for A100.
- Use `--device cuda` for Colab/Kaggle GPU runs.
- Use `--load-4bit` only on CUDA machines with bitsandbytes support.
- If `--depth-path` is omitted, the inference script uses a black placeholder depth map and still inserts the `<depth>` token. This is useful for smoke tests only, not valid benchmark reproduction.
- The local official source tree has two compatibility fixes: full HF checkpoint loading no longer returns early, and the released `Siglip/...` tower alias is mapped to `google/siglip-so400m-patch14-384`.
