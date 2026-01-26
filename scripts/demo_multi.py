import argparse
import os
import os.path as osp
import json
import numpy as np
import cv2
import torch
import gc
import sys
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


def load_txt_multi(gt_path, bbox_format="xywh"):
    """
    Load multiple bounding boxes from a text file.
    Each line represents one object to track.

    Args:
        gt_path: Path to the text file
        bbox_format: "xywh" for x,y,width,height or "xyxy" for x1,y1,x2,y2

    Returns a dict mapping object_id to (bbox_xyxy, label)
    """
    with open(gt_path, 'r') as f:
        lines = f.readlines()

    prompts = {}
    for obj_id, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        coords = list(map(float, line.split(',')))

        if bbox_format == "xywh":
            # x,y,w,h format (top-left corner + width/height)
            x, y, w, h = coords
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        elif bbox_format == "xyxy":
            # x1,y1,x2,y2 format (top-left and bottom-right corners)
            x1, y1, x2, y2 = map(int, coords)
        else:
            raise ValueError(f"Unknown bbox_format: {bbox_format}. Use 'xywh' or 'xyxy'.")

        prompts[obj_id] = ((x1, y1, x2, y2), obj_id)

    return prompts


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt_multi(args.txt_path, args.bbox_format)

    num_objects = len(prompts)
    print(f"Tracking {num_objects} objects")

    # Results storage: frame_idx -> {obj_id: bbox}
    all_results = {}

    frame_rate = 30
    loaded_frames = None
    height, width = None, None

    # Load frames if saving to video
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path)
                           if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

        print(f"Loaded {len(loaded_frames)} frames ({width}x{height}) at {frame_rate} FPS")

    # Setup video writer
    out = None
    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)

        # Add all objects to track
        for obj_id, (bbox, label) in prompts.items():
            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=obj_id)
            print(f"  Added object {obj_id} with bbox {bbox}")

        # Propagate through video
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}
            frame_results = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)

                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # x,y,w,h format

                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask
                frame_results[int(obj_id)] = {
                    "bbox": bbox,  # x, y, w, h
                    "center": [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]  # center x, y
                }

            all_results[frame_idx] = frame_results

            # Draw on frame if saving video
            if args.save_to_video and loaded_frames is not None:
                img = loaded_frames[frame_idx].copy()

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
                    # Draw ID label
                    label = f"ID:{obj_id}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x, y - label_h - 10), (x + label_w + 4, y), color, -1)
                    cv2.putText(img, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                out.write(img)

            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"  Processed frame {frame_idx}")

    if out is not None:
        out.release()
        print(f"Video saved to: {args.video_output_path}")

    # Save results to JSON
    if args.result_path:
        output_data = {
            "video_path": args.video_path,
            "num_objects": num_objects,
            "num_frames": len(all_results),
            "format": {
                "bbox": "x, y, width, height (top-left corner)",
                "center": "center_x, center_y"
            },
            "frames": all_results
        }
        with open(args.result_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.result_path}")

    # Also save to simple CSV format for easy parsing
    if args.csv_path:
        with open(args.csv_path, 'w') as f:
            f.write("frame,object_id,x,y,width,height,center_x,center_y\n")
            for frame_idx in sorted(all_results.keys()):
                for obj_id, data in all_results[frame_idx].items():
                    bbox = data["bbox"]
                    center = data["center"]
                    f.write(f"{frame_idx},{obj_id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{center[0]},{center[1]}\n")
        print(f"CSV saved to: {args.csv_path}")

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-person tracking with SAMURAI")
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to bounding box file (one bbox per line).")
    parser.add_argument("--bbox_format", default="xywh", choices=["xywh", "xyxy"],
                        help="Bounding box format: 'xywh' for x,y,width,height or 'xyxy' for x1,y1,x2,y2 (default: xywh)")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo_multi.mp4", help="Path to save the output video.")
    parser.add_argument("--result_path", default=None, help="Path to save tracking results as JSON.")
    parser.add_argument("--csv_path", default=None, help="Path to save tracking results as CSV.")
    parser.add_argument("--save_to_video", type=lambda x: x.lower() == 'true', default=True, help="Save results to a video.")
    args = parser.parse_args()
    main(args)
