"""Small helpers for optional TensorRT release workflows."""

from __future__ import annotations

from pathlib import Path


def expected_engine_paths(export_dir: str | Path) -> dict[str, Path]:
    root = Path(export_dir)
    return {
        "nerf": root / "nerf_fp16.trt",
        "sampler": root / "minmaxrays_net_fp16.trt",
        "refine": root / "refine_net_fp16.trt",
    }
