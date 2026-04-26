"""Unified command line interface for the supported LLFF Fern release path.

The research scripts are still kept at the repository root to preserve the
original training behavior. This CLI provides stable, documented commands that
dispatch to those scripts with release configs and safer defaults.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE1_SCRIPT = REPO_ROOT / "run_S_eS_eN_alter_base.py"
STAGE2_SCRIPT = REPO_ROOT / "run_S_eS_eN_alter_base_refine2.py"
TRT_SCRIPT = REPO_ROOT / "run_S_eS_eN_alter_trt.py"

DEFAULT_STAGE1_CONFIG = REPO_ROOT / "configs/llff/fern/fern_epi.txt"
DEFAULT_STAGE2_CONFIG = REPO_ROOT / "configs/llff/fern/fern_refine.txt"
DEFAULT_TRT_CONFIG = REPO_ROOT / "configs/llff/fern/fern_trt.txt"


def _repo_relative(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _append_flag(argv: list[str], flag: str, value: str | int | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _append_bool(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def _extra_args(args: argparse.Namespace) -> list[str]:
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def _run_script(script: Path, args: Iterable[str]) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    old_argv = sys.argv[:]
    sys.argv = [str(script), *args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def _parse_config(path: str | Path) -> dict[str, str]:
    """Parse the simple configargparse key-value format used by this repo."""
    values: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def train_stage1(args: argparse.Namespace) -> None:
    argv = ["--config", _repo_relative(args.config)]
    _append_bool(argv, "--no_reload", args.no_reload)
    _append_flag(argv, "--max_steps", args.max_steps)
    argv.extend(_extra_args(args))
    _run_script(STAGE1_SCRIPT, argv)


def train_stage2(args: argparse.Namespace) -> None:
    argv = ["--config", _repo_relative(args.config)]
    _append_flag(argv, "--pretrain_path", args.pretrain_path)
    _append_bool(argv, "--no_reload", args.no_reload)
    _append_flag(argv, "--max_steps", args.max_steps)
    argv.extend(_extra_args(args))
    _run_script(STAGE2_SCRIPT, argv)


def infer(args: argparse.Namespace) -> None:
    argv = ["--config", _repo_relative(args.config)]
    _append_flag(argv, "--ft_path", args.checkpoint)
    _append_bool(argv, "--render_test", args.render_test)
    _append_bool(argv, "--use_trt", args.use_trt)
    _append_flag(argv, "--max_images", args.max_images)
    argv.extend(_extra_args(args))
    _run_script(TRT_SCRIPT, argv)


def export_trt(args: argparse.Namespace) -> None:
    config = Path(_repo_relative(args.config))
    config_values = _parse_config(config)
    basedir = Path(config_values.get("basedir", "logs_trt"))
    expname = config_values.get("expname", "fern_8samples_trtinfer")
    export_dir = REPO_ROOT / basedir / expname

    argv = ["--config", str(config), "--export_only"]
    _append_flag(argv, "--ft_path", args.checkpoint)
    argv.extend(_extra_args(args))
    _run_script(TRT_SCRIPT, argv)

    if args.onnx_only:
        print(f"ONNX export complete: {export_dir}")
        return

    try:
        from onnx2trt import get_engine
    except Exception as exc:  # TensorRT often is not installed on training hosts.
        print(f"ONNX export complete: {export_dir}")
        print(f"Skipping TensorRT engine build because TensorRT is unavailable: {exc}")
        return

    n_samples = int(config_values.get("N_samples", 8))
    n_point_ray_enc = int(config_values.get("N_point_ray_enc", 48))
    num_neighbor = int(config_values.get("num_neighbor", 4))
    image_batch = args.height * args.width

    get_engine(
        onnx_file_path=str(export_dir / "nerf.onnx"),
        engine_file_path=str(export_dir / "nerf_fp16.trt"),
        max_batch_size=image_batch * n_samples,
        in_ch=[63, 27],
        fp16_mode=True,
        is_nerf=True,
    )
    get_engine(
        onnx_file_path=str(export_dir / "minmaxrays_net.onnx"),
        engine_file_path=str(export_dir / "minmaxrays_net_fp16.trt"),
        max_batch_size=image_batch,
        in_ch=6 * n_point_ray_enc,
        fp16_mode=True,
        is_nerf=False,
    )
    get_engine(
        onnx_file_path=str(export_dir / "refine_net.onnx"),
        engine_file_path=str(export_dir / "refine_net_fp16.trt"),
        max_batch_size=image_batch,
        in_ch=(3 * num_neighbor) * n_samples + 6 * n_samples,
        fp16_mode=True,
        is_nerf=False,
    )
    print(f"TensorRT engines written to: {export_dir}")


def eval_model(args: argparse.Namespace) -> None:
    args.render_test = True
    infer(args)


def _add_common_passthrough(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="additional arguments forwarded to the underlying research script; prefix with --",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pronerf.cli",
        description="Clean release CLI for the LLFF Fern ProNeRF/MMNeRF pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("train-stage1", help="run alternating PAS sampler/NeRF training")
    p.add_argument("--config", default=str(DEFAULT_STAGE1_CONFIG))
    p.add_argument("--no-reload", action="store_true", dest="no_reload")
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    _add_common_passthrough(p)
    p.set_defaults(func=train_stage1)

    p = subparsers.add_parser("train-stage2", help="run refinement training from a stage 1 checkpoint")
    p.add_argument("--config", default=str(DEFAULT_STAGE2_CONFIG))
    p.add_argument("--pretrain-path", default=None, dest="pretrain_path")
    p.add_argument("--no-reload", action="store_true", dest="no_reload")
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    _add_common_passthrough(p)
    p.set_defaults(func=train_stage2)

    p = subparsers.add_parser("infer", help="render held-out/test views")
    p.add_argument("--config", default=str(DEFAULT_TRT_CONFIG))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--render-test", action="store_true", dest="render_test")
    p.add_argument("--use-trt", action="store_true", dest="use_trt")
    p.add_argument("--max-images", type=int, default=None, dest="max_images")
    _add_common_passthrough(p)
    p.set_defaults(func=infer)

    p = subparsers.add_parser("eval", help="render test split through the inference path")
    p.add_argument("--config", default=str(DEFAULT_TRT_CONFIG))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--use-trt", action="store_true", dest="use_trt")
    p.add_argument("--max-images", type=int, default=None, dest="max_images")
    _add_common_passthrough(p)
    p.set_defaults(func=eval_model)

    p = subparsers.add_parser("export-trt", help="export ONNX and optionally build TensorRT FP16 engines")
    p.add_argument("--config", default=str(DEFAULT_TRT_CONFIG))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--onnx-only", action="store_true", dest="onnx_only")
    p.add_argument("--height", type=int, default=756)
    p.add_argument("--width", type=int, default=1008)
    _add_common_passthrough(p)
    p.set_defaults(func=export_trt)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    mpl_config = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/mmnerf-matplotlib"))
    mpl_config.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
