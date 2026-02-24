# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAMURAI (Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory) is a visual object tracking system built on Meta's SAM 2.1. It requires no training — it uses pre-trained SAM 2.1 weights with a Kalman filter for motion-aware memory selection.

## Setup

```bash
# Install SAM2 (requires Python >=3.10, PyTorch >=2.3.1)
cd sam2 && pip install -e . && pip install -e ".[notebooks]"

# Download model checkpoints
cd checkpoints && ./download_ckpts.sh && cd ../..

# Additional dependencies
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
```

## Key Commands

**Benchmark inference (LaSOT):**
```bash
python scripts/main_inference.py
```

**Multi-GPU chunked inference:**
```bash
python scripts/main_inference_chunk.py --dataset_path data/LaSOT-ext --tracker_name samurai --model_name large --chunk_idx 0 --num_chunks 8
```

**Single-object demo on custom video:**
```bash
python scripts/demo.py --video_path <video.mp4> --txt_path <bbox.txt> --model_path sam2/checkpoints/sam2.1_hiera_base_plus.pt
```

**Multi-object demo with caching:**
```bash
python scripts/demo_multi.py --video_path <video.mp4> --txt_path <bbox.txt> --txt_bbox_format xyxy --video_output_path out.mp4 --result_path result.json --cache_path cache.json
```

**Interactive review GUI (Tkinter):**
```bash
python scripts/review_gui.py --video_path <video.mp4> --cache_path <cache.json>
```

## Architecture

### Directory Structure

- `sam2/` — Modified SAM 2.1 framework (the core deep learning model). Installed as a package.
  - `sam2/sam2/modeling/` — Model architecture: `sam2_base.py` (base model), `sam2_video_predictor.py` (main video tracking predictor with SAMURAI extensions)
  - `sam2/sam2/configs/samurai/` — SAMURAI-specific YAML configs for 4 model sizes (tiny, small, base_plus, large)
- `scripts/` — Main entry points for inference and demos
- `lib/` — Supporting library code (evaluation datasets, trackers, training infrastructure, utilities)
- `data/` — Expected location for datasets (LaSOT format)
- `demo/` — Sample demo videos and bounding box files

### Model Sizes and Config Mapping

| Size | Config | Checkpoint |
|------|--------|------------|
| tiny | `configs/samurai/sam2.1_hiera_t.yaml` | `sam2.1_hiera_tiny.pt` |
| small | `configs/samurai/sam2.1_hiera_s.yaml` | `sam2.1_hiera_small.pt` |
| base_plus | `configs/samurai/sam2.1_hiera_b+.yaml` | `sam2.1_hiera_base_plus.pt` |
| large | `configs/samurai/sam2.1_hiera_l.yaml` | `sam2.1_hiera_large.pt` |

### Key Flow

1. `build_sam2_video_predictor()` (from `sam2.build_sam`) creates the model from a YAML config + checkpoint
2. `predictor.init_state()` initializes tracking state for a video
3. `predictor.add_new_points_or_box()` provides the initial bounding box prompt
4. `predictor.propagate_in_video()` runs frame-by-frame tracking with motion-aware memory

### SAMURAI-Specific Parameters (in YAML configs)

- `samurai_mode: true` — Enables Kalman filter motion estimation
- `stable_frames_threshold` — Frames before memory stabilizes (default: 15)
- `memory_bank_iou_threshold` — IoU threshold for memory bank updates (default: 0.5)
- `kf_score_weight` — Kalman filter score blending weight

### Bbox Format Convention

- Input `.txt` files for `demo.py`: `x,y,w,h` format (one bbox per line)
- Input `.txt` files for `demo_multi.py`: supports `xyxy` format via `--txt_bbox_format xyxy`
- SAM 2 internally uses `x1,y1,x2,y2` format

### Caching System

`demo_multi.py` supports a JSON cache (`--cache_path`) that stores per-frame inference state. This allows:
- Resuming interrupted runs
- Incremental re-processing from a specific frame (`--start_frame`, `--end_frame`)
- The review GUI (`review_gui.py`) reads and writes to the same cache for interactive corrections

## Git Workflow

- `main` is the active development branch
- `master` is the base branch for PRs
