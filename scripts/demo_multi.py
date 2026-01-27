import argparse
import os
import os.path as osp
import json
import numpy as np
import cv2
import torch
import gc
import sys
import tempfile
import shutil
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

# Colors for different object IDs (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Orange
    (255, 128, 0),  # Light blue
    (0, 128, 255),  # Orange-red
    (128, 255, 0),  # Light green
]

# Default chunk size (number of frames per chunk)
DEFAULT_CHUNK_SIZE = 500


def load_txt_multi(gt_path, bbox_format="xywh"):
    """
    Load multiple bounding boxes from a text file.
    Each line represents one object to track.

    Args:
        gt_path: Path to the text file
        bbox_format: "xywh" for x,y,width,height or "xyxy" for x1,y1,x2,y2

    Returns a dict mapping object_id to (bbox_xyxy, label)
    """
    print(f"[DEBUG] Loading bounding boxes from: {gt_path}")
    print(f"[DEBUG] Bounding box format: {bbox_format}")

    with open(gt_path, 'r') as f:
        lines = f.readlines()

    print(f"[DEBUG] Found {len(lines)} lines in bbox file")

    prompts = {}
    for obj_id, line in enumerate(lines):
        line = line.strip()
        if not line:
            print(f"[DEBUG] Skipping empty line {obj_id}")
            continue
        coords = list(map(float, line.split(',')))
        print(f"[DEBUG] Line {obj_id}: raw coords = {coords}")

        if bbox_format == "xywh":
            # x,y,w,h format (top-left corner + width/height)
            x, y, w, h = coords
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            print(f"[DEBUG] Object {obj_id}: xywh=({x},{y},{w},{h}) -> xyxy=({x1},{y1},{x2},{y2})")
        elif bbox_format == "xyxy":
            # x1,y1,x2,y2 format (top-left and bottom-right corners)
            x1, y1, x2, y2 = map(int, coords)
            print(f"[DEBUG] Object {obj_id}: xyxy=({x1},{y1},{x2},{y2})")
        else:
            raise ValueError(f"Unknown bbox_format: {bbox_format}. Use 'xywh' or 'xyxy'.")

        prompts[obj_id] = ((x1, y1, x2, y2), obj_id)

    print(f"[DEBUG] Successfully loaded {len(prompts)} bounding boxes")
    return prompts


def determine_model_cfg(model_path):
    print(f"[DEBUG] Determining model config for: {model_path}")
    if "large" in model_path:
        cfg = "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        cfg = "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        cfg = "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")
    print(f"[DEBUG] Selected model config: {cfg}")
    return cfg


def prepare_frames_or_path(video_path):
    print(f"[DEBUG] Preparing frames/path for: {video_path}")
    if video_path.endswith(".mp4"):
        print(f"[DEBUG] Input is an MP4 video file")
        return video_path
    elif osp.isdir(video_path):
        print(f"[DEBUG] Input is a directory of frames")
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def get_video_info(video_path):
    """Get video frame count and dimensions."""
    if osp.isdir(video_path):
        frames = sorted([f for f in os.listdir(video_path)
                        if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
        num_frames = len(frames)
        if num_frames > 0:
            first_frame = cv2.imread(osp.join(video_path, frames[0]))
            height, width = first_frame.shape[:2]
            frame_rate = 30  # Default for image sequences
        else:
            raise ValueError("No frames found in directory")
    else:
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    return num_frames, width, height, frame_rate


def extract_chunk_frames(video_path, start_frame, end_frame, temp_dir):
    """Extract frames from video/directory for a specific chunk into a temp directory."""
    os.makedirs(temp_dir, exist_ok=True)

    if osp.isdir(video_path):
        # Copy frames from source directory
        frames = sorted([f for f in os.listdir(video_path)
                        if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
        for i, frame_idx in enumerate(range(start_frame, end_frame)):
            if frame_idx < len(frames):
                src = osp.join(video_path, frames[frame_idx])
                dst = osp.join(temp_dir, f"{i:05d}.jpg")
                shutil.copy2(src, dst)
    else:
        # Extract frames from video file
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(osp.join(temp_dir, f"{i:05d}.jpg"), frame)
        cap.release()

    return temp_dir


def load_chunk_frames_for_vis(video_path, start_frame, end_frame):
    """Load frames for visualization."""
    frames = []
    if osp.isdir(video_path):
        frame_files = sorted([f for f in os.listdir(video_path)
                             if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
        for frame_idx in range(start_frame, end_frame):
            if frame_idx < len(frame_files):
                frame = cv2.imread(osp.join(video_path, frame_files[frame_idx]))
                frames.append(frame)
    else:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    return frames


def process_chunk(predictor, chunk_path, prompts, chunk_start_frame, save_to_video,
                  video_frames=None, height=None, width=None):
    """Process a single chunk of video and return results and updated prompts for next chunk."""
    chunk_results = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(chunk_path, offload_video_to_cpu=True)

        # Add all objects to track with their current bboxes
        for obj_id, (bbox, label) in prompts.items():
            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=obj_id)

        # Propagate through chunk
        last_bboxes = {}  # Store last valid bbox for each object
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            global_frame_idx = chunk_start_frame + frame_idx
            mask_to_vis = {}
            bbox_to_vis = {}
            frame_results = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)

                if len(non_zero_indices) == 0:
                    # Use last known bbox if available
                    if obj_id in last_bboxes:
                        bbox = last_bboxes[obj_id]
                    else:
                        bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # x,y,w,h
                    last_bboxes[obj_id] = bbox

                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask
                frame_results[int(obj_id)] = {
                    "bbox": bbox,
                    "center": [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
                }

            chunk_results[global_frame_idx] = frame_results

            # Draw on frame if saving video
            if save_to_video and video_frames is not None and frame_idx < len(video_frames):
                img = video_frames[frame_idx].copy()

                # Draw masks
                for obj_id, mask in mask_to_vis.items():
                    color = COLORS[obj_id % len(COLORS)]
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color
                    img = cv2.addWeighted(img, 1, mask_img, 0.3, 0)

                # Draw bounding boxes and labels
                for obj_id, bbox in bbox_to_vis.items():
                    color = COLORS[obj_id % len(COLORS)]
                    x, y, w, h = bbox
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    label = f"ID:{obj_id}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x, y - label_h - 10), (x + label_w + 4, y), color, -1)
                    cv2.putText(img, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                video_frames[frame_idx] = img  # Update in place for writing later

        # Prepare prompts for next chunk using last detected bboxes (convert xywh to xyxy)
        next_prompts = {}
        for obj_id, (_, label) in prompts.items():
            if obj_id in last_bboxes:
                x, y, w, h = last_bboxes[obj_id]
                bbox_xyxy = (x, y, x + w, y + h)
                next_prompts[obj_id] = (bbox_xyxy, label)
            else:
                # Object lost - keep the original bbox
                next_prompts[obj_id] = prompts[obj_id]

        # Clean up
        del state

    return chunk_results, next_prompts, video_frames


def main(args):
    # Compute frame_output_path if not specified (same directory as video output)
    if args.frame_output_path is None:
        video_output_dir = osp.dirname(args.video_output_path)
        if video_output_dir == "":
            video_output_dir = "."
        args.frame_output_path = osp.join(video_output_dir, "frame_images")

    print("=" * 60)
    print("[DEBUG] Starting multi-object tracking (chunked processing)")
    print("=" * 60)
    print(f"[DEBUG] Arguments:")
    print(f"[DEBUG]   video_path: {args.video_path}")
    print(f"[DEBUG]   txt_path: {args.txt_path}")
    print(f"[DEBUG]   bbox_format: {args.bbox_format}")
    print(f"[DEBUG]   model_path: {args.model_path}")
    print(f"[DEBUG]   video_output_path: {args.video_output_path}")
    print(f"[DEBUG]   frame_output_path: {args.frame_output_path}")
    print(f"[DEBUG]   result_path: {args.result_path}")
    print(f"[DEBUG]   csv_path: {args.csv_path}")
    print(f"[DEBUG]   save_to_video: {args.save_to_video}")
    print(f"[DEBUG]   chunk_size: {args.chunk_size}")
    print("-" * 60)

    print("\n[DEBUG] Step 1: Loading model configuration...")
    model_cfg = determine_model_cfg(args.model_path)

    print("\n[DEBUG] Step 2: Building SAM2 video predictor...")
    print(f"[DEBUG] Using device: cuda:0")
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    print("[DEBUG] Predictor built successfully")

    print("\n[DEBUG] Step 3: Getting video information...")
    num_frames, width, height, frame_rate = get_video_info(args.video_path)
    print(f"[DEBUG] Video: {num_frames} frames, {width}x{height}, {frame_rate} FPS")

    print("\n[DEBUG] Step 4: Loading bounding boxes...")
    prompts = load_txt_multi(args.txt_path, args.bbox_format)

    num_objects = len(prompts)
    print(f"\n[DEBUG] Step 5: Ready to track {num_objects} objects")

    # Calculate chunks
    chunk_size = args.chunk_size
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    print(f"[DEBUG] Processing video in {num_chunks} chunks of up to {chunk_size} frames each")
    print("-" * 60)

    # Results storage: frame_idx -> {obj_id: bbox}
    all_results = {}

    # Setup video writer
    out = None
    if args.save_to_video:
        print(f"\n[DEBUG] Setting up video writer...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))
        if out.isOpened():
            print(f"[DEBUG] Video writer initialized: {args.video_output_path}")
        else:
            print(f"[DEBUG] WARNING: Video writer may not be properly initialized")

        # Create frame output directory
        os.makedirs(args.frame_output_path, exist_ok=True)
        print(f"[DEBUG] Frame output directory: {args.frame_output_path}")

    # Create temp directory for chunk frames
    temp_base_dir = tempfile.mkdtemp(prefix="samurai_chunks_")
    print(f"[DEBUG] Using temp directory: {temp_base_dir}")

    current_prompts = prompts
    try:
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, num_frames)
            chunk_frames_count = chunk_end - chunk_start

            print(f"\n[DEBUG] Processing chunk {chunk_idx + 1}/{num_chunks}: frames {chunk_start}-{chunk_end - 1} ({chunk_frames_count} frames)")

            # Extract chunk frames to temp directory
            chunk_temp_dir = osp.join(temp_base_dir, f"chunk_{chunk_idx:04d}")
            print(f"[DEBUG] Extracting frames to: {chunk_temp_dir}")
            extract_chunk_frames(args.video_path, chunk_start, chunk_end, chunk_temp_dir)

            # Load frames for visualization if needed
            video_frames = None
            if args.save_to_video:
                video_frames = load_chunk_frames_for_vis(args.video_path, chunk_start, chunk_end)
                print(f"[DEBUG] Loaded {len(video_frames)} frames for visualization")

            # Process chunk
            print(f"[DEBUG] Running tracking on chunk...")
            chunk_results, next_prompts, video_frames = process_chunk(
                predictor, chunk_temp_dir, current_prompts, chunk_start,
                args.save_to_video, video_frames, height, width
            )

            # Merge results
            all_results.update(chunk_results)

            # Write frames to video and save as images
            if args.save_to_video and video_frames is not None:
                for i, frame in enumerate(video_frames):
                    out.write(frame)
                    # Save frame as image
                    global_frame_idx = chunk_start + i
                    frame_filename = osp.join(args.frame_output_path, f"frame_image_{global_frame_idx:06d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                print(f"[DEBUG] Wrote {len(video_frames)} frames to output video and frame_images folder")

            # Update prompts for next chunk
            current_prompts = next_prompts

            # Clean up chunk temp directory
            shutil.rmtree(chunk_temp_dir)

            # Clean up GPU memory between chunks
            gc.collect()
            torch.cuda.empty_cache()

            print(f"[DEBUG] Chunk {chunk_idx + 1} complete. Total frames processed: {len(all_results)}")

    finally:
        # Clean up temp directory
        if osp.exists(temp_base_dir):
            shutil.rmtree(temp_base_dir)
            print(f"[DEBUG] Cleaned up temp directory")

    print(f"\n[DEBUG] Saving outputs...")
    if out is not None:
        out.release()
        print(f"[DEBUG] Video writer released")
        print(f"[DEBUG] Video saved to: {args.video_output_path}")

    # Save results to JSON
    if args.result_path:
        print(f"[DEBUG] Saving results to JSON: {args.result_path}")
        # Convert keys to strings for JSON serialization
        json_results = {str(k): v for k, v in all_results.items()}
        output_data = {
            "video_path": args.video_path,
            "num_objects": num_objects,
            "num_frames": len(all_results),
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "format": {
                "bbox": "x, y, width, height (top-left corner)",
                "center": "center_x, center_y"
            },
            "frames": json_results
        }
        with open(args.result_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"[DEBUG] JSON results saved successfully ({len(all_results)} frames)")

    # Also save to simple CSV format for easy parsing
    if args.csv_path:
        print(f"[DEBUG] Saving results to CSV: {args.csv_path}")
        total_rows = 0
        with open(args.csv_path, 'w') as f:
            f.write("frame,object_id,x,y,width,height,center_x,center_y\n")
            for frame_idx in sorted(all_results.keys()):
                for obj_id, data in all_results[frame_idx].items():
                    bbox = data["bbox"]
                    center = data["center"]
                    f.write(f"{frame_idx},{obj_id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{center[0]},{center[1]}\n")
                    total_rows += 1
        print(f"[DEBUG] CSV saved successfully ({total_rows} data rows)")

    print(f"\n[DEBUG] Cleaning up resources...")
    del predictor
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("[DEBUG] Multi-object tracking completed successfully!")
    print(f"[DEBUG] Summary:")
    print(f"[DEBUG]   Objects tracked: {num_objects}")
    print(f"[DEBUG]   Frames processed: {len(all_results)}")
    print(f"[DEBUG]   Chunks processed: {num_chunks}")
    if args.save_to_video:
        print(f"[DEBUG]   Output video: {args.video_output_path}")
        print(f"[DEBUG]   Frame images: {args.frame_output_path}/")
    if args.result_path:
        print(f"[DEBUG]   JSON results: {args.result_path}")
    if args.csv_path:
        print(f"[DEBUG]   CSV results: {args.csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-person tracking with SAMURAI (chunked processing for long videos)")
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to bounding box file (one bbox per line).")
    parser.add_argument("--bbox_format", default="xywh", choices=["xywh", "xyxy"],
                        help="Bounding box format: 'xywh' for x,y,width,height or 'xyxy' for x1,y1,x2,y2 (default: xywh)")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo_multi.mp4", help="Path to save the output video.")
    parser.add_argument("--frame_output_path", default=None, help="Path to folder for saving output frame images. Defaults to 'frame_images' in same directory as video output.")
    parser.add_argument("--result_path", default=None, help="Path to save tracking results as JSON.")
    parser.add_argument("--csv_path", default=None, help="Path to save tracking results as CSV.")
    parser.add_argument("--save_to_video", type=lambda x: x.lower() == 'true', default=True, help="Save results to a video.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Number of frames per chunk for memory-efficient processing (default: {DEFAULT_CHUNK_SIZE}). "
                             "Lower values use less memory but may affect tracking continuity at chunk boundaries.")
    args = parser.parse_args()
    main(args)
