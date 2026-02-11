#!/usr/bin/env python3
"""Interactive GUI for reviewing and correcting SAMURAI tracking predictions.

Displays video frames with predicted bboxes overlaid. Users can draw corrected
bounding boxes and trigger automatic re-processing from the corrected frame.

Usage:
    python scripts/review_gui.py \
        --video_path ./demo/cortis_go/cortis_go_cut.mp4 \
        --cache_path ./demo/cortis_go/cortis_go_cache.json
"""

import argparse
import collections
import json
import os
import os.path as osp
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import cv2
from PIL import Image, ImageTk

# Colors matching demo_multi.py COLORS (BGR) converted to RGB hex for tkinter
COLORS_HEX = [
    "#0000FF",  # Blue       (demo_multi BGR: 255,0,0)
    "#00FF00",  # Green      (demo_multi BGR: 0,255,0)
    "#FF0000",  # Red        (demo_multi BGR: 0,0,255)
    "#00FFFF",  # Cyan       (demo_multi BGR: 255,255,0)
    "#FF00FF",  # Magenta    (demo_multi BGR: 255,0,255)
    "#FFFF00",  # Yellow     (demo_multi BGR: 0,255,255)
    "#FF0080",  # Pink       (demo_multi BGR: 128,0,255)
    "#0080FF",  # Sky blue   (demo_multi BGR: 255,128,0)
    "#FF8000",  # Orange     (demo_multi BGR: 0,128,255)
    "#00FF80",  # Light green(demo_multi BGR: 128,255,0)
]

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


# ---------------------------------------------------------------------------
# FrameCache: on-demand frame extraction with LRU cache
# ---------------------------------------------------------------------------
class FrameCache:
    """Extract individual frames from video on-demand with bounded LRU cache."""

    def __init__(self, video_path, max_cache_size=50):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._cache = collections.OrderedDict()
        self.max_cache_size = max_cache_size

    def get_frame(self, frame_idx):
        """Return PIL Image (RGB) for the given frame index."""
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = self.cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_idx}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._cache[frame_idx] = img
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return img

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------------------
# TrackingData: in-memory cache JSON with load/save/truncate/merge
# ---------------------------------------------------------------------------
class TrackingData:
    """In-memory representation of tracking cache/result JSON."""

    def __init__(self):
        self.video_path = ""
        self.num_objects = 0
        self.chunk_size = 500
        self.frames = {}  # {frame_idx(int): {obj_id(int): {"bbox":[x,y,w,h], "center":[cx,cy]}}}
        self.source_path = None

    def load(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.video_path = data.get("video_path", "")
        self.num_objects = data.get("num_objects", 0)
        self.chunk_size = data.get("chunk_size", 500)
        self.source_path = json_path
        self.frames = {}
        for fk, fv in data.get("frames", {}).items():
            frame_idx = int(fk)
            self.frames[frame_idx] = {}
            for ok, ov in fv.items():
                self.frames[frame_idx][int(ok)] = ov

    def save(self, json_path):
        json_frames = {}
        for fk in sorted(self.frames.keys()):
            json_frames[str(fk)] = {str(ok): ov for ok, ov in self.frames[fk].items()}
        output = {
            "video_path": self.video_path,
            "num_objects": self.num_objects,
            "num_frames": len(self.frames),
            "chunk_size": self.chunk_size,
            "num_chunks": (len(self.frames) + self.chunk_size - 1) // self.chunk_size if self.chunk_size > 0 else 1,
            "format": {
                "bbox": "x, y, width, height (top-left corner)",
                "center": "center_x, center_y"
            },
            "frames": json_frames
        }
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

    def get_frame_data(self, frame_idx):
        """Return {obj_id: {"bbox":..., "center":...}} or None."""
        return self.frames.get(frame_idx)

    def truncate_from(self, frame_idx):
        """Remove all data from frame_idx onward (inclusive)."""
        to_remove = [k for k in self.frames if k >= frame_idx]
        for k in to_remove:
            del self.frames[k]

    def merge_results(self, result_json_path):
        """Merge result JSON into this data. Frames in result overwrite."""
        with open(result_json_path) as f:
            data = json.load(f)
        for fk, fv in data.get("frames", {}).items():
            frame_idx = int(fk)
            self.frames[frame_idx] = {}
            for ok, ov in fv.items():
                self.frames[frame_idx][int(ok)] = ov

    def has_frame(self, frame_idx):
        return frame_idx in self.frames

    def max_frame_idx(self):
        return max(self.frames.keys()) if self.frames else -1


# ---------------------------------------------------------------------------
# CanvasView: frame display + bbox overlays + drawing
# ---------------------------------------------------------------------------
class CanvasView(tk.Canvas):
    """Canvas that displays a video frame with bbox overlays and supports drawing."""

    def __init__(self, parent, on_correction_change=None, **kwargs):
        super().__init__(parent, bg="black", highlightthickness=0, **kwargs)
        self.pil_image = None
        self.photo_image = None  # prevent GC
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.fit_scale = 1.0

        # Bbox data
        self.predicted_bboxes = {}   # {obj_id: [x,y,w,h]}
        self.pending_corrections = {}  # {obj_id: [x,y,w,h]}

        # Drawing state
        self.draw_mode = False
        self.selected_obj_id = 0
        self._drag_start = None
        self._drag_rect_id = None
        self._on_correction_change = on_correction_change

        # Pan state
        self._pan_start = None

        # Mouse bindings
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", self._on_resize)
        # Zoom
        self.bind("<Button-4>", self._on_scroll_up)
        self.bind("<Button-5>", self._on_scroll_down)
        self.bind("<MouseWheel>", self._on_mousewheel)
        # Pan with right-click drag
        self.bind("<ButtonPress-3>", self._on_pan_start)
        self.bind("<B3-Motion>", self._on_pan_drag)

    def set_frame(self, pil_image, predicted_bboxes, pending_corrections=None):
        self.pil_image = pil_image
        self.predicted_bboxes = predicted_bboxes or {}
        if pending_corrections is not None:
            self.pending_corrections = pending_corrections
        self._compute_fit_scale()
        self._redraw()

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._redraw()

    def _compute_fit_scale(self):
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if self.pil_image and canvas_w > 1 and canvas_h > 1:
            img_w, img_h = self.pil_image.size
            self.fit_scale = min(canvas_w / img_w, canvas_h / img_h)

    def _get_image_origin(self):
        """Return (left, top) of image in canvas coordinates."""
        if self.pil_image is None:
            return 0, 0
        eff_scale = self.fit_scale * self.zoom_level
        img_w, img_h = self.pil_image.size
        cx = self.winfo_width() / 2 + self.pan_x
        cy = self.winfo_height() / 2 + self.pan_y
        left = cx - (img_w * eff_scale) / 2
        top = cy - (img_h * eff_scale) / 2
        return left, top

    def _redraw(self):
        self.delete("all")
        if self.pil_image is None:
            return
        eff_scale = self.fit_scale * self.zoom_level
        new_w = max(1, int(self.pil_image.width * eff_scale))
        new_h = max(1, int(self.pil_image.height * eff_scale))
        resized = self.pil_image.resize((new_w, new_h), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized)
        cx = self.winfo_width() / 2 + self.pan_x
        cy = self.winfo_height() / 2 + self.pan_y
        self.create_image(cx, cy, image=self.photo_image, anchor="center")

        img_left, img_top = self._get_image_origin()

        # Draw predicted bboxes
        for obj_id, bbox in self.predicted_bboxes.items():
            if obj_id in self.pending_corrections:
                self._draw_bbox(bbox, obj_id, eff_scale, img_left, img_top, style="faded")
            else:
                self._draw_bbox(bbox, obj_id, eff_scale, img_left, img_top, style="solid")

        # Draw pending corrections
        for obj_id, bbox in self.pending_corrections.items():
            self._draw_bbox(bbox, obj_id, eff_scale, img_left, img_top, style="correction")

    def _draw_bbox(self, bbox_xywh, obj_id, scale, img_left, img_top, style):
        x, y, w, h = bbox_xywh
        color = COLORS_HEX[obj_id % len(COLORS_HEX)]
        cx1 = img_left + x * scale
        cy1 = img_top + y * scale
        cx2 = img_left + (x + w) * scale
        cy2 = img_top + (y + h) * scale

        if style == "solid":
            self.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2)
        elif style == "faded":
            self.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=1, dash=(4, 4))
        elif style == "correction":
            self.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=3, dash=(6, 3))

        # Label
        label = f"ID:{obj_id}"
        font_size = max(8, int(10 * scale / self.fit_scale)) if self.fit_scale > 0 else 10
        self.create_text(cx1 + 4, cy1 - 4, text=label, fill=color,
                         anchor="sw", font=("Helvetica", font_size, "bold"))

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        eff_scale = self.fit_scale * self.zoom_level
        img_left, img_top = self._get_image_origin()
        ix = (canvas_x - img_left) / eff_scale
        iy = (canvas_y - img_top) / eff_scale
        return int(round(ix)), int(round(iy))

    def _on_press(self, event):
        if not self.draw_mode or self.pil_image is None:
            return
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        if self._drag_rect_id:
            self.delete(self._drag_rect_id)
        color = COLORS_HEX[self.selected_obj_id % len(COLORS_HEX)]
        self._drag_rect_id = self.create_rectangle(
            self._drag_start[0], self._drag_start[1], event.x, event.y,
            outline=color, width=2, dash=(4, 2))

    def _on_release(self, event):
        if self._drag_start is None:
            return
        if self._drag_rect_id:
            self.delete(self._drag_rect_id)
            self._drag_rect_id = None
        ix1, iy1 = self._canvas_to_image_coords(self._drag_start[0], self._drag_start[1])
        ix2, iy2 = self._canvas_to_image_coords(event.x, event.y)
        x1, x2 = min(ix1, ix2), max(ix1, ix2)
        y1, y2 = min(iy1, iy2), max(iy1, iy2)
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.pil_image.width, x2)
        y2 = min(self.pil_image.height, y2)
        w, h = x2 - x1, y2 - y1
        self._drag_start = None
        if w < 5 or h < 5:
            return
        self.pending_corrections[self.selected_obj_id] = [x1, y1, w, h]
        self._redraw()
        if self._on_correction_change:
            self._on_correction_change()

    def _on_resize(self, event):
        self._compute_fit_scale()
        self._redraw()

    def _on_scroll_up(self, event):
        self.zoom_level = min(4.0, self.zoom_level * 1.15)
        self._redraw()

    def _on_scroll_down(self, event):
        self.zoom_level = max(0.25, self.zoom_level / 1.15)
        self._redraw()

    def _on_mousewheel(self, event):
        if event.delta > 0:
            self._on_scroll_up(event)
        else:
            self._on_scroll_down(event)

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.pan_x += dx
        self.pan_y += dy
        self._pan_start = (event.x, event.y)
        self._redraw()


# ---------------------------------------------------------------------------
# ReprocessDialog: modal dialog for running demo_multi.py as subprocess
# ---------------------------------------------------------------------------
class ReprocessDialog(tk.Toplevel):
    """Modal dialog that truncates cache, writes bbox file, runs demo_multi.py,
    and merges results back."""

    def __init__(self, parent, tracking_data, correction_frame,
                 corrected_bboxes, video_path, model_path,
                 cache_save_path, end_frame=None):
        super().__init__(parent)
        self.title("Reprocessing...")
        self.geometry("800x500")
        self.transient(parent)
        self.grab_set()

        self.tracking_data = tracking_data
        self.correction_frame = correction_frame
        self.corrected_bboxes = corrected_bboxes
        self.video_path = video_path
        self.model_path = model_path
        self.cache_save_path = cache_save_path
        self.end_frame = end_frame
        self.success = False
        self.process = None
        self.temp_dir = None
        self.backup_path = None
        self._log_queue = queue.Queue()

        # UI
        self.log_text = scrolledtext.ScrolledText(self, state='disabled', wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.status_label = tk.Label(btn_frame, text="Preparing...")
        self.status_label.pack(side=tk.LEFT)
        self.cancel_btn = tk.Button(btn_frame, text="Cancel", command=self._cancel)
        self.cancel_btn.pack(side=tk.RIGHT)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.after(100, self._start_reprocessing)

    def _log(self, text):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def _start_reprocessing(self):
        self.temp_dir = tempfile.mkdtemp(prefix="review_gui_")
        try:
            # Backup cache before truncation
            if self.cache_save_path and osp.exists(self.cache_save_path):
                ts = time.strftime("%Y%m%d_%H%M%S")
                self.backup_path = self.cache_save_path + f".backup_{ts}"
                shutil.copy2(self.cache_save_path, self.backup_path)
                self._log(f"Backup saved: {self.backup_path}\n")

            # Truncate cache from correction frame
            self._log(f"Truncating cache from frame {self.correction_frame}...\n")
            self.tracking_data.truncate_from(self.correction_frame)

            # Save truncated cache
            cache_path = osp.join(self.temp_dir, "cache.json")
            self.tracking_data.save(cache_path)
            self._log(f"Truncated cache saved ({len(self.tracking_data.frames)} frames remaining)\n")

            # Write bbox file
            bbox_path = osp.join(self.temp_dir, "bbox_corrections.txt")
            self._write_bbox_file(bbox_path)

            # Result output
            result_path = osp.join(self.temp_dir, "result.json")
            self.result_path = result_path

            # Build command
            cmd = [
                sys.executable, "scripts/demo_multi.py",
                "--video_path", self.video_path,
                "--txt_path", bbox_path,
                "--txt_bbox_format", "xywh",
                "--model_path", self.model_path,
                "--video_output_path", osp.join(self.temp_dir, "temp_video.mp4"),
                "--result_path", result_path,
                "--cache_path", cache_path,
                "--save_to_video", "false",
                "--start_frame", str(self.correction_frame),
            ]
            if self.end_frame is not None:
                cmd.extend(["--end_frame", str(self.end_frame)])

            self._log(f"\nCommand:\n{' '.join(cmd)}\n\n")
            self.status_label.config(text=f"Running demo_multi.py from frame {self.correction_frame}...")

            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=PROJECT_ROOT
            )
            self._read_thread = threading.Thread(target=self._read_output, daemon=True)
            self._read_thread.start()
            self._poll_process()

        except Exception as e:
            self._log(f"ERROR: {e}\n")
            self.status_label.config(text="Error!")
            self._restore_backup()

    def _write_bbox_file(self, bbox_path):
        """Write bbox file with one line per object.
        Corrected objects use user's bbox; others use last known bbox."""
        prev_frame = self.correction_frame - 1
        prev_data = self.tracking_data.get_frame_data(prev_frame)
        # Determine num_objects: use tracking data if available, otherwise
        # derive from user corrections (covers new video with empty cache)
        num_objects = self.tracking_data.num_objects
        if num_objects == 0 and self.corrected_bboxes:
            num_objects = max(self.corrected_bboxes.keys()) + 1
            self.tracking_data.num_objects = num_objects
        lines = []
        self._log(f"Writing bbox file for {num_objects} objects at frame {self.correction_frame}:\n")
        for obj_id in range(num_objects):
            if obj_id in self.corrected_bboxes:
                bbox = self.corrected_bboxes[obj_id]
                self._log(f"  ID {obj_id}: {bbox} (user correction)\n")
            elif prev_data and obj_id in prev_data:
                bbox = prev_data[obj_id]["bbox"]
                self._log(f"  ID {obj_id}: {bbox} (from frame {prev_frame})\n")
            else:
                bbox = [0, 0, 0, 0]
                self._log(f"  ID {obj_id}: {bbox} (fallback - no data)\n")
            lines.append(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
        with open(bbox_path, 'w') as f:
            f.write("\n".join(lines) + "\n")

    def _read_output(self):
        try:
            for line in self.process.stdout:
                self._log_queue.put(line)
            self.process.stdout.close()
        except Exception:
            pass

    def _poll_process(self):
        while not self._log_queue.empty():
            try:
                self._log(self._log_queue.get_nowait())
            except queue.Empty:
                break
        if self.process is None:
            return
        rc = self.process.poll()
        if rc is None:
            self.after(100, self._poll_process)
        else:
            self._on_complete(rc)

    def _on_complete(self, return_code):
        if return_code == 0:
            self.status_label.config(text="Merging results...")
            try:
                self.tracking_data.merge_results(self.result_path)
                # Save merged data to the original cache path
                if self.cache_save_path:
                    self.tracking_data.save(self.cache_save_path)
                    self._log(f"\nCache saved to: {self.cache_save_path}\n")
                self.success = True
                self.status_label.config(text="Done!")
                self._log("\n=== Reprocessing complete. Results merged. ===\n")
            except Exception as e:
                self._log(f"\nERROR merging results: {e}\n")
                self.status_label.config(text="Merge failed!")
                self._restore_backup()
        else:
            self._log(f"\nProcess exited with code {return_code}\n")
            self.status_label.config(text=f"Failed (exit code {return_code})")
            self._restore_backup()
        self.cancel_btn.config(text="Close")
        self._cleanup_temp()

    def _restore_backup(self):
        if self.backup_path and osp.exists(self.backup_path):
            self._log(f"Restoring cache from backup: {self.backup_path}\n")
            try:
                self.tracking_data.load(self.backup_path)
                self._log("Cache restored successfully.\n")
            except Exception as e:
                self._log(f"ERROR restoring backup: {e}\n")

    def _cleanup_temp(self):
        if self.temp_dir and osp.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _cancel(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            self._restore_backup()
        self._cleanup_temp()
        self.destroy()


# ---------------------------------------------------------------------------
# ReviewApp: main window
# ---------------------------------------------------------------------------
class ReviewApp(tk.Tk):
    """Main application window for tracking review."""

    def __init__(self, video_path=None, cache_path=None, model_path=None):
        super().__init__()
        self.title("SAMURAI Tracking Review")
        self.geometry("1400x900")

        self.frame_cache = None
        self.tracking_data = TrackingData()
        self.current_frame_idx = 0
        self.num_frames = 0
        self.video_path = ""
        self.cache_path = ""
        self.model_path = model_path or "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        self.is_playing = False

        self._build_menu()
        self._build_canvas()
        self._build_nav_bar()
        self._build_object_panel()
        self._build_action_bar()
        self._bind_keys()

        if video_path:
            self.load_video(video_path)
        if cache_path:
            self.load_cache(cache_path)

    # --- Menu ---
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video...", command=self._menu_open_video)
        file_menu.add_command(label="Open Cache/Result JSON...", command=self._menu_open_cache)
        file_menu.add_separator()
        file_menu.add_command(label="Save Cache...", command=self._menu_save_cache)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    # --- Canvas ---
    def _build_canvas(self):
        self.canvas_view = CanvasView(self, on_correction_change=self._on_correction_change)
        self.canvas_view.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

    # --- Navigation Bar ---
    def _build_nav_bar(self):
        nav = tk.Frame(self)
        nav.pack(fill=tk.X, padx=5, pady=2)

        # Frame label and entry
        tk.Label(nav, text="Frame:").pack(side=tk.LEFT)
        self.frame_entry = tk.Entry(nav, width=8)
        self.frame_entry.pack(side=tk.LEFT, padx=2)
        self.frame_entry.insert(0, "0")
        self.frame_entry.bind("<Return>", self._on_frame_entry)

        self.frame_total_label = tk.Label(nav, text="/ 0")
        self.frame_total_label.pack(side=tk.LEFT, padx=(0, 10))

        # Navigation buttons
        tk.Button(nav, text="|<", width=3, command=self._goto_first).pack(side=tk.LEFT)
        tk.Button(nav, text="<<", width=3, command=lambda: self._step_frames(-100)).pack(side=tk.LEFT)
        tk.Button(nav, text="<", width=3, command=lambda: self._step_frames(-1)).pack(side=tk.LEFT)
        tk.Button(nav, text=">", width=3, command=lambda: self._step_frames(1)).pack(side=tk.LEFT)
        tk.Button(nav, text=">>", width=3, command=lambda: self._step_frames(100)).pack(side=tk.LEFT)
        tk.Button(nav, text=">|", width=3, command=self._goto_last).pack(side=tk.LEFT)

        tk.Label(nav, text="  ").pack(side=tk.LEFT)
        self.play_btn = tk.Button(nav, text="Play", width=5, command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(nav, text="Fit", width=4, command=self.canvas_view.reset_view).pack(side=tk.LEFT)

        # Slider
        slider_frame = tk.Frame(self)
        slider_frame.pack(fill=tk.X, padx=5, pady=0)
        self.frame_slider = tk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                     command=self._on_slider, showvalue=False)
        self.frame_slider.pack(fill=tk.X)

    # --- Object Panel ---
    def _build_object_panel(self):
        panel = tk.LabelFrame(self, text="Object & Drawing")
        panel.pack(fill=tk.X, padx=5, pady=2)

        row1 = tk.Frame(panel)
        row1.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(row1, text="Object ID:").pack(side=tk.LEFT)
        self.obj_id_var = tk.IntVar(value=0)
        self.obj_id_spin = tk.Spinbox(row1, from_=0, to=9, width=4,
                                       textvariable=self.obj_id_var,
                                       command=self._on_obj_id_change)
        self.obj_id_spin.pack(side=tk.LEFT, padx=5)

        self.color_swatch = tk.Label(row1, text="  ", width=3,
                                      bg=COLORS_HEX[0], relief=tk.RIDGE)
        self.color_swatch.pack(side=tk.LEFT, padx=5)

        tk.Label(row1, text="  Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="view")
        tk.Radiobutton(row1, text="View", variable=self.mode_var,
                       value="view", command=self._on_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(row1, text="Draw Bbox", variable=self.mode_var,
                       value="draw", command=self._on_mode_change).pack(side=tk.LEFT)

        tk.Button(row1, text="Clear Corrections", command=self._clear_corrections).pack(side=tk.RIGHT, padx=5)

        # Corrections list
        row2 = tk.Frame(panel)
        row2.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(row2, text="Corrections:").pack(side=tk.LEFT)
        self.corrections_label = tk.Label(row2, text="(none)", fg="gray")
        self.corrections_label.pack(side=tk.LEFT, padx=5)

    # --- Action Bar ---
    def _build_action_bar(self):
        action = tk.Frame(self)
        action.pack(fill=tk.X, padx=5, pady=(2, 5))

        self.reprocess_btn = tk.Button(action, text="Apply Corrections & Reprocess",
                                        command=self._on_apply_corrections,
                                        bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
        self.reprocess_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(action, text="End frame (optional):").pack(side=tk.LEFT, padx=(20, 2))
        self.end_frame_entry = tk.Entry(action, width=8)
        self.end_frame_entry.pack(side=tk.LEFT, padx=2)

        self.status_label = tk.Label(action, text="Ready", fg="gray")
        self.status_label.pack(side=tk.RIGHT, padx=5)

    # --- Key Bindings ---
    def _bind_keys(self):
        self.bind("<Left>", lambda e: self._step_frames(-1))
        self.bind("<Right>", lambda e: self._step_frames(1))
        self.bind("<Shift-Left>", lambda e: self._step_frames(-10))
        self.bind("<Shift-Right>", lambda e: self._step_frames(10))
        self.bind("<Home>", lambda e: self._goto_first())
        self.bind("<End>", lambda e: self._goto_last())
        self.bind("<space>", lambda e: self._toggle_play())
        self.bind("<d>", lambda e: self._toggle_draw_mode())
        self.bind("<D>", lambda e: self._toggle_draw_mode())
        self.bind("<Escape>", lambda e: self._exit_draw_mode())
        self.bind("<f>", lambda e: self.canvas_view.reset_view())
        self.bind("<F>", lambda e: self.canvas_view.reset_view())
        self.bind("<plus>", lambda e: self.canvas_view._on_scroll_up(e))
        self.bind("<minus>", lambda e: self.canvas_view._on_scroll_down(e))
        self.bind("<equal>", lambda e: self.canvas_view._on_scroll_up(e))
        self.bind("<Control-s>", lambda e: self._menu_save_cache())
        self.bind("<Control-z>", lambda e: self._undo_last_correction())
        # Number keys for quick object selection
        for i in range(10):
            self.bind(str(i), lambda e, idx=i: self._quick_select_obj(idx))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- Load/Save ---
    def load_video(self, video_path):
        try:
            self.frame_cache = FrameCache(video_path)
            self.video_path = video_path
            self.num_frames = self.frame_cache.num_frames
            self.frame_slider.config(to=max(0, self.num_frames - 1))
            self.frame_total_label.config(text=f"/ {self.num_frames - 1}")
            self.title(f"SAMURAI Tracking Review - {osp.basename(video_path)}")
            self._show_frame(0)
            self.status_label.config(text=f"Loaded video: {self.num_frames} frames", fg="black")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open video:\n{e}")

    def load_cache(self, cache_path):
        if not osp.exists(cache_path):
            create = messagebox.askyesno(
                "Cache not found",
                f"Cache file does not exist:\n{cache_path}\n\n"
                "Create a new empty cache file?")
            if create:
                self.tracking_data = TrackingData()
                self.tracking_data.video_path = self.video_path
                self.tracking_data.save(cache_path)
                self.cache_path = cache_path
                self.status_label.config(text=f"Created new cache: {cache_path}", fg="black")
            return

        try:
            self.tracking_data.load(cache_path)
            self.cache_path = cache_path
            num_objects = self.tracking_data.num_objects
            if num_objects > 0:
                self.obj_id_spin.config(to=num_objects - 1)
            self.status_label.config(
                text=f"Loaded cache: {len(self.tracking_data.frames)} frames, {num_objects} objects",
                fg="black")
            self._show_frame(self.current_frame_idx)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load cache:\n{e}")

    def _menu_open_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.load_video(path)

    def _menu_open_cache(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.load_cache(path)

    def _menu_save_cache(self):
        if not self.cache_path:
            path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])
            if path:
                self.cache_path = path
        if self.cache_path:
            self.tracking_data.save(self.cache_path)
            self.status_label.config(text=f"Cache saved: {self.cache_path}", fg="green")

    # --- Frame Display ---
    def _show_frame(self, frame_idx):
        if self.frame_cache is None:
            return
        frame_idx = max(0, min(frame_idx, self.num_frames - 1))
        # Clear corrections when navigating to a different frame
        if frame_idx != self.current_frame_idx:
            self.canvas_view.pending_corrections.clear()
        self.current_frame_idx = frame_idx
        try:
            pil_image = self.frame_cache.get_frame(frame_idx)
        except Exception:
            return

        # Get predicted bboxes from tracking data
        predicted = {}
        frame_data = self.tracking_data.get_frame_data(frame_idx)
        if frame_data:
            for obj_id, data in frame_data.items():
                predicted[obj_id] = data["bbox"]

        self.canvas_view.set_frame(pil_image, predicted)
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(frame_idx))
        self.frame_slider.set(frame_idx)
        self._update_corrections_label()

    # --- Navigation ---
    def _on_slider(self, value):
        frame_idx = int(float(value))
        if frame_idx != self.current_frame_idx:
            self._show_frame(frame_idx)

    def _on_frame_entry(self, event=None):
        try:
            frame_idx = int(self.frame_entry.get())
            self._show_frame(frame_idx)
        except ValueError:
            pass

    def _step_frames(self, delta):
        self._show_frame(self.current_frame_idx + delta)

    def _goto_first(self):
        self._show_frame(0)

    def _goto_last(self):
        self._show_frame(self.num_frames - 1)

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Stop" if self.is_playing else "Play")
        if self.is_playing:
            self._play_tick()

    def _play_tick(self):
        if not self.is_playing:
            return
        if self.current_frame_idx < self.num_frames - 1:
            self._show_frame(self.current_frame_idx + 1)
            self.after(33, self._play_tick)  # ~30fps
        else:
            self.is_playing = False
            self.play_btn.config(text="Play")

    # --- Object & Drawing ---
    def _on_obj_id_change(self):
        obj_id = self.obj_id_var.get()
        self.canvas_view.selected_obj_id = obj_id
        color = COLORS_HEX[obj_id % len(COLORS_HEX)]
        self.color_swatch.config(bg=color)

    def _on_mode_change(self):
        self.canvas_view.draw_mode = (self.mode_var.get() == "draw")
        cursor = "crosshair" if self.canvas_view.draw_mode else ""
        self.canvas_view.config(cursor=cursor)

    def _toggle_draw_mode(self):
        if self.mode_var.get() == "draw":
            self.mode_var.set("view")
        else:
            self.mode_var.set("draw")
        self._on_mode_change()

    def _exit_draw_mode(self):
        self.mode_var.set("view")
        self._on_mode_change()

    def _quick_select_obj(self, obj_id):
        self.obj_id_var.set(obj_id)
        self._on_obj_id_change()

    def _clear_corrections(self):
        self.canvas_view.pending_corrections.clear()
        self.canvas_view._redraw()
        self._update_corrections_label()

    def _undo_last_correction(self):
        if self.canvas_view.pending_corrections:
            last_key = list(self.canvas_view.pending_corrections.keys())[-1]
            del self.canvas_view.pending_corrections[last_key]
            self.canvas_view._redraw()
            self._update_corrections_label()

    def _on_correction_change(self):
        self._update_corrections_label()

    def _update_corrections_label(self):
        corrections = self.canvas_view.pending_corrections
        if not corrections:
            self.corrections_label.config(text="(none)", fg="gray")
        else:
            parts = []
            for obj_id, bbox in sorted(corrections.items()):
                parts.append(f"ID:{obj_id} [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
            self.corrections_label.config(text="  ".join(parts), fg="black")

    # --- Reprocessing ---
    def _on_apply_corrections(self):
        corrections = dict(self.canvas_view.pending_corrections)
        if not corrections:
            messagebox.showwarning("No Corrections",
                                   "Draw at least one corrected bbox first.\n"
                                   "Select 'Draw Bbox' mode, choose an object ID, "
                                   "then click and drag on the frame.")
            return

        if not self.video_path:
            messagebox.showerror("Error", "No video loaded.")
            return

        if not self.cache_path:
            path = filedialog.asksaveasfilename(
                title="Save cache as...",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")])
            if not path:
                return
            self.cache_path = path

        # Check frame 0 edge case
        if self.current_frame_idx == 0:
            if len(corrections) < self.tracking_data.num_objects:
                resp = messagebox.askyesno(
                    "Frame 0 Warning",
                    f"You are correcting frame 0 but only drew {len(corrections)} "
                    f"of {self.tracking_data.num_objects} objects.\n"
                    f"Missing objects will use [0,0,0,0] as bbox.\n"
                    f"Continue anyway?")
                if not resp:
                    return

        # Parse optional end frame
        end_frame = None
        end_frame_text = self.end_frame_entry.get().strip()
        if end_frame_text:
            try:
                end_frame = int(end_frame_text)
            except ValueError:
                messagebox.showerror("Error", "End frame must be a number.")
                return

        frame_idx = self.current_frame_idx
        dialog = ReprocessDialog(
            self,
            tracking_data=self.tracking_data,
            correction_frame=frame_idx,
            corrected_bboxes=corrections,
            video_path=self.video_path,
            model_path=self.model_path,
            cache_save_path=self.cache_path,
            end_frame=end_frame
        )
        self.wait_window(dialog)

        if dialog.success:
            self.canvas_view.pending_corrections.clear()
            # Refresh num_objects and spinbox (may have been set for first time)
            num_objects = self.tracking_data.num_objects
            if num_objects > 0:
                self.obj_id_spin.config(to=num_objects - 1)
            self._show_frame(frame_idx)
            self.status_label.config(text="Reprocessing complete", fg="green")

    # --- Cleanup ---
    def _on_close(self):
        if self.frame_cache:
            self.frame_cache.close()
        self.destroy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interactive GUI for reviewing and correcting SAMURAI tracking predictions")
    parser.add_argument("--video_path", default=None, help="Path to video file (.mp4)")
    parser.add_argument("--cache_path", default=None, help="Path to cache/result JSON file")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt",
                        help="Path to SAM2 model checkpoint (for reprocessing)")
    args = parser.parse_args()

    app = ReviewApp(
        video_path=args.video_path,
        cache_path=args.cache_path,
        model_path=args.model_path
    )
    app.mainloop()


if __name__ == "__main__":
    main()
