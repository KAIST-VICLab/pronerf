# Release Refactor Plan

This repository is being prepared as a clean GitHub release for the LLFF Fern
pipeline inspired by ProNeRF: stage 1 projection-aware sparse ray sampling,
stage 2 refinement, and optional TensorRT inference.

## Supported Scope

- Maintained dataset path: `data/nerf_llff_data/fern`.
- Maintained workflow: stage 1 training, stage 2 refinement, PyTorch inference,
  ONNX export, and optional TensorRT FP16 engine build.
- Legacy experiment scripts and configs are removed from the release tree.

## Public Commands

```bash
python -m pronerf.cli train-stage1 --config configs/llff/fern/fern_epi.txt
python -m pronerf.cli train-stage2 --config configs/llff/fern/fern_refine.txt --pretrain-path <stage1.tar>
python -m pronerf.cli infer --config configs/llff/fern/fern_trt.txt --checkpoint <stage2.tar> --render-test
python -m pronerf.cli export-trt --config configs/llff/fern/fern_trt.txt --checkpoint <stage2.tar>
python -m pronerf.cli infer --config configs/llff/fern/fern_trt.txt --use-trt --render-test
```

## Cleanup Policy

- Only the three supported configs are kept:
  `fern_epi.txt`, `fern_refine.txt`, and `fern_trt.txt`.
- Supported scripts no longer set `CUDA_VISIBLE_DEVICES`; select GPUs outside
  Python, e.g. `CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli ...`.
- Heavy artifacts are ignored by Git: datasets, checkpoints, renders, ONNX, and
  TensorRT engines.

## Validation

Run short smoke tests before long training:

```bash
python -m pronerf.cli train-stage1 --max-steps 2 --no-reload -- --i_weights 2
python -m pronerf.cli train-stage2 --max-steps 2 --pretrain-path <stage1.tar> --no-reload -- --i_weights 2
python -m pronerf.cli infer --render-test --max-images 1 --checkpoint <stage2.tar>
```
