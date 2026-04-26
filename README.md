# ProNeRF: Learning Efficient Projection-Aware Ray Sampling for Fine-Grained Implicit Neural Radiance Fields

<p align="center">
  <strong>IEEE Access 2024</strong>
</p>

<p align="center">
  Juan Luis Gonzalez Bello* &nbsp;&nbsp;&nbsp;
  Minh-Quan Viet Bui* &nbsp;&nbsp;&nbsp;
  Munchurl Kim
</p>

<p align="center">
  KAIST, South Korea<br>
  *Co-first authors (equal contribution)
</p>

<p align="center">
  <a href="https://kaist-viclab.github.io/pronerf-site/">Project Page</a> |
  <a href="https://ieeexplore.ieee.org/document/10504815">Paper</a> |
  <a href="https://github.com/KAIST-VICLab/pronerf">Code</a>
</p>

<p align="center">
  <a href="full_pipeline.pdf">Pipeline Figure</a>
</p>

This repository contains a cleaned release path for efficient NeRF rendering on
the LLFF forward-facing `fern` scene. The supported workflow follows the
ProNeRF idea of projection-aware sparse ray sampling: train a sampler and NeRF,
refine sparse samples using neighboring-view color projections, then optionally
export the trained networks to TensorRT.

The original environment and NeRF training layout are based on
[yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). This
release keeps that config-driven style but exposes a cleaner CLI for the
maintained LLFF Fern pipeline.

## News

- **2024**: ProNeRF was published in IEEE Access, Volume 12, pages 56799-56814.
- Code release repository: <https://github.com/KAIST-VICLab/pronerf>

## Installation

Clone this repository:

```bash
git clone https://github.com/KAIST-VICLab/pronerf.git
cd pronerf
```

Create the base environment:

```bash
conda env create -f environment.yml
conda activate nerf_pl
pip install -r requirements.txt
```

The code was developed around Python 3.7, PyTorch 1.11, and CUDA 11.3. TensorRT
is optional and only needed for engine export/inference:

```bash
pip install -r requirements-trt.txt
```

If TensorRT or PyCUDA are not available, PyTorch training and rendering still
work.

## Data

Download the NeRF example data, including LLFF `fern`:

```bash
bash download_example_data.sh
```

Expected layout:

```text
data/nerf_llff_data/fern/
  images/
  images_4/
  poses_bounds.npy
```

This release documents and supports LLFF Fern as the maintained path.

## Training

Select GPUs outside Python:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli train-stage1 \
  --config configs/llff/fern/fern_epi.txt
```

Stage 1 alternates between NeRF exploration updates and sampler exploitation
updates. With the provided config, it writes checkpoints under
`logs_epi_RR/fern_sampler_e2e_donerf_8samples_cc/`.

Then run stage 2 refinement from a stage 1 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli train-stage2 \
  --config configs/llff/fern/fern_refine.txt \
  --pretrain-path logs_epi_RR/fern_sampler_e2e_donerf_8samples_cc/500000.tar
```

Stage 2 refines sparse samples with neighboring-view color projections and
writes checkpoints under `logs_epi_RR/fern_refine_8samples_v2/`.

For a quick smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli train-stage1 --max-steps 2 --no-reload -- --i_weights 2
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli train-stage2 --max-steps 2 \
  --pretrain-path logs_epi_RR/fern_sampler_e2e_donerf_8samples_cc/000002.tar --no-reload -- --i_weights 2
```

## Rendering and Evaluation

Render held-out test views with PyTorch:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli infer \
  --config configs/llff/fern/fern_trt.txt \
  --checkpoint logs_epi_RR/fern_refine_8samples_v2/500000.tar \
  --render-test
```

For a one-image smoke render:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli infer --render-test --max-images 1 \
  --checkpoint logs_epi_RR/fern_refine_8samples_v2/500000.tar
```

Rendered images are saved under the configured experiment directory, for example
`logs_minmax/fern_8samples_trtinfer/renderonly_test_*`.

## TensorRT Export

Export ONNX models and build TensorRT FP16 engines:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli export-trt \
  --config configs/llff/fern/fern_trt.txt \
  --checkpoint logs_epi_RR/fern_refine_8samples_v2/500000.tar
```

If TensorRT is unavailable, the command still exports ONNX and reports that the
engine build was skipped. To export ONNX only:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli export-trt --onnx-only \
  --checkpoint logs_epi_RR/fern_refine_8samples_v2/500000.tar
```

Run TensorRT inference after engines are available:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pronerf.cli infer \
  --config configs/llff/fern/fern_trt.txt \
  --checkpoint logs_epi_RR/fern_refine_8samples_v2/500000.tar \
  --use-trt --render-test
```

## Repository Layout

```text
pronerf/                         # release CLI and small helpers
configs/llff/fern/               # supported Fern configs only
docs/release_refactor_plan.md    # cleanup plan and validation notes
run_S_eS_eN_alter_base.py        # original stage 1 implementation
run_S_eS_eN_alter_base_refine2.py# original stage 2 implementation
run_S_eS_eN_alter_trt.py         # original TRT/PyTorch inference implementation
```

The root research scripts are preserved for reproducibility; the recommended
public entrypoint is `python -m pronerf.cli`.

## Citation

If this code or paper is useful for your research, please cite ProNeRF:

```bibtex
@ARTICLE{10504815,
      author={Bello, Juan Luis Gonzalez and Bui, Minh-Quan Viet and Kim, Munchurl},
      journal={IEEE Access}, 
      title={ProNeRF: Learning Efficient Projection-Aware Ray Sampling for Fine-Grained Implicit Neural Radiance Fields}, 
      year={2024},
      volume={12},
      number={},
      pages={56799-56814},
      keywords={Rendering (computer graphics);Three-dimensional displays;Training;Image color analysis;Geometry;Pipelines;Neural radiance field;3D reconstruction;neural radiance field;neural rendering;view synthesis;ray sampling},
      doi={10.1109/ACCESS.2024.3390753}}
```

This release also builds on the environment and code style of
[`nerf-pytorch`](https://github.com/yenchenlin/nerf-pytorch).
