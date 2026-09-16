"""
Pluto - Adaptive 3D Image Segmentation Desktop Application

This application wraps the custom adaptive segmentation logic from pluto.py
into an interactive desktop GUI built with Tkinter and Pillow.

Core Features:
1. Data Loading: Loads 2D image stacks (TIFF, PNG, etc.) from ./data/{folder_name}
   ordered numerically by filename.
2. User Interface:
   - Scrubbing through slices with slider, arrow keys, and mouse wheel.
   - Zoom and Pan navigation (Ctrl+Wheel or Zoom buttons).
   - Interactive drawing tools (Brush, Eraser, Polygon/Lasso) to draw masks by hand.
   - Visual mask overlay with adjustable opacity.
3. Algorithm Execution:
   - Propagates mask consecutively across slices using pluto.py adaptive logic.
4. Empty Mask Handling:
   - Pauses immediately if an empty mask (or non-growing seed) is encountered.
   - Notifies user with two options: 'Stop Execution' or 'Segment by Hand'.
   - Resumes seamlessly after manual correction.
5. Saving:
   - Exports all masks strictly as binary PNG files into ./masks/{folder_name}.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageTk, ImageDraw

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.utils import (
    load_image_stack,
    save_masks,
    characterize_mask,
    propagate_slice,
    fill_mask_holes,
)


class PlutoApp(tk.Tk):
    """Main Application Window for Pluto 3D Image Segmentation."""

    def __init__(self):
        super().__init__()
        self.title("Pluto - Adaptive 3D Image Segmentation")
        self.geometry("1100x820")
        self.minsize(850, 650)

        # Style configuration
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Application State
        self.stack: Optional[np.ndarray] = None
        self.filenames: List[str] = []
        self.folder_name: str = ""
        self.current_slice_idx: int = 0
        self.num_slices: int = 0
        self.masks: Dict[int, np.ndarray] = {}  # slice_idx -> 2D binary uint8 mask
        self.mask_history: Dict[int, List[np.ndarray]] = {}  # For undo support

        # Algorithm Parameters
        self.alpha_var = tk.DoubleVar(value=0.8)
        self.k_std_var = tk.DoubleVar(value=2.0)
        self.max_iter_var = tk.IntVar(value=40)
        self.m_seeds_var = tk.IntVar(value=5)
        self.start_slice_var = tk.IntVar(value=0)
        self.end_slice_var = tk.IntVar(value=0)
        self._syncing_slice: bool = False
        self.start_slice_var.trace_add("write", self._on_start_slice_changed)

        # Execution State
        self.is_running: bool = False
        self.is_paused_for_manual: bool = False
        self.running_current_slice: int = 0
        self.active_mean: float = 0.0
        self.active_std: float = 0.0
        self.active_yc: int = 0
        self.active_xc: int = 0
        self.active_spatial_std_y: float = 2.0
        self.active_spatial_std_x: float = 2.0

        # Drawing / Interaction State
        self.tool_mode = tk.StringVar(value="brush")  # 'brush', 'eraser', 'lasso'
        self.brush_radius = tk.IntVar(value=6)
        self.overlay_alpha = tk.DoubleVar(value=0.45)
        self.show_overlay = tk.BooleanVar(value=True)

        # Zoom & Pan State
        self.zoom_scale: float = 1.0
        self.pan_x: int = 0
        self.pan_y: int = 0
        self.last_mouse_x: int = 0
        self.last_mouse_y: int = 0
        self.lasso_points: List[Tuple[int, int]] = []
        self.is_panning: bool = False

        # Image display cache
        self.photo_img: Optional[ImageTk.PhotoImage] = None

        # Build UI layout
        self._create_menu()
        self._create_widgets()
        self._bind_events()

        # Attempt default load of data/cropped_3_3_1 if available
        default_data = os.path.join("data", "cropped_3_3_1")
        if os.path.isdir(default_data):
            self.load_dataset(default_data)

    # =========================================================================
    # UI Creation
    # =========================================================================

    def _create_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Stack Folder...", command=self.on_open_folder)
        file_menu.add_command(label="Save Masks...", command=self.on_save_masks)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo Stroke (Ctrl+Z)", command=self.undo_mask)
        edit_menu.add_command(label="Clear Current Mask", command=self.clear_current_mask)
        edit_menu.add_command(label="Clear All Masks", command=self.clear_all_masks)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About Pluto", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _create_widgets(self):
        # Top toolbar
        toolbar = ttk.Frame(self, padding=(8, 4))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_open = ttk.Button(toolbar, text="📁 Open Stack", command=self.on_open_folder)
        btn_open.pack(side=tk.LEFT, padx=4)

        self.lbl_folder = ttk.Label(toolbar, text="No dataset loaded", font=("Segoe UI", 9, "bold"))
        self.lbl_folder.pack(side=tk.LEFT, padx=8)

        # Drawing Tool Selection in toolbar
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(toolbar, text="Tool:").pack(side=tk.LEFT, padx=2)

        r_brush = ttk.Radiobutton(toolbar, text="🖌️ Brush", variable=self.tool_mode, value="brush")
        r_brush.pack(side=tk.LEFT, padx=2)
        r_lasso = ttk.Radiobutton(toolbar, text="➰ Lasso / Poly", variable=self.tool_mode, value="lasso")
        r_lasso.pack(side=tk.LEFT, padx=2)
        r_eraser = ttk.Radiobutton(toolbar, text="🧹 Eraser", variable=self.tool_mode, value="eraser")
        r_eraser.pack(side=tk.LEFT, padx=2)

        ttk.Label(toolbar, text="Radius:").pack(side=tk.LEFT, padx=(8, 2))
        spin_radius = ttk.Spinbox(toolbar, from_=1, to=50, textvariable=self.brush_radius, width=4)
        spin_radius.pack(side=tk.LEFT, padx=2)

        btn_undo = ttk.Button(toolbar, text="↶ Undo", command=self.undo_mask, width=6)
        btn_undo.pack(side=tk.LEFT, padx=4)
        btn_clear = ttk.Button(toolbar, text="✖ Clear Mask", command=self.clear_current_mask, width=11)
        btn_clear.pack(side=tk.LEFT, padx=4)
        btn_clear_all = ttk.Button(toolbar, text="🗑️ Clear All", command=self.clear_all_masks, width=10)
        btn_clear_all.pack(side=tk.LEFT, padx=4)

        # Save button in toolbar
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        btn_save = ttk.Button(toolbar, text="💾 Save Masks", command=self.on_save_masks)
        btn_save.pack(side=tk.LEFT, padx=4)

        # Main Workspace (Split into Left: Controls, Right: Canvas)
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left Control Panel
        left_panel = ttk.Frame(main_paned, width=280, padding=8)
        main_paned.add(left_panel, weight=0)

        # Right View Panel
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=1)

        self._create_control_panel(left_panel)
        self._create_canvas_panel(right_panel)

        # Status Bar
        self.status_bar = ttk.Label(
            self,
            text="Ready. Open a folder or start drawing an initial mask.",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(6, 4)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_control_panel(self, parent: ttk.Frame):
        # 1. Dataset & Slice Info
        lbl_info_group = ttk.LabelFrame(parent, text="Dataset & Slice", padding=8)
        lbl_info_group.pack(fill=tk.X, pady=4)

        self.lbl_slice_info = ttk.Label(lbl_info_group, text="Slice: 0 / 0", font=("Segoe UI", 10, "bold"))
        self.lbl_slice_info.pack(anchor=tk.W, pady=2)
        self.lbl_filename_info = ttk.Label(lbl_info_group, text="File: -", foreground="gray")
        self.lbl_filename_info.pack(anchor=tk.W, pady=2)
        self.lbl_mask_stats = ttk.Label(lbl_info_group, text="Mask: None", foreground="#2a52be")
        self.lbl_mask_stats.pack(anchor=tk.W, pady=2)

        # 2. Algorithm Parameters Frame
        algo_group = ttk.LabelFrame(parent, text="Algorithm Parameters", padding=8)
        algo_group.pack(fill=tk.X, pady=4)

        ttk.Label(algo_group, text="Alpha (Moving Avg, 0-1):").pack(anchor=tk.W)
        scale_alpha = ttk.Scale(algo_group, from_=0.0, to=1.0, variable=self.alpha_var, orient=tk.HORIZONTAL)
        scale_alpha.pack(fill=tk.X, pady=2)
        lbl_alpha_val = ttk.Label(algo_group, textvariable=self.alpha_var)
        lbl_alpha_val.pack(anchor=tk.E)

        ttk.Label(algo_group, text="Intensity Tolerance (k * std):").pack(anchor=tk.W)
        spin_k = ttk.Spinbox(algo_group, from_=0.5, to=5.0, increment=0.1, textvariable=self.k_std_var, width=6)
        spin_k.pack(anchor=tk.W, pady=2)

        ttk.Label(algo_group, text="Max Iterations:").pack(anchor=tk.W)
        spin_iter = ttk.Spinbox(algo_group, from_=1, to=200, textvariable=self.max_iter_var, width=6)
        spin_iter.pack(anchor=tk.W, pady=2)

        ttk.Label(algo_group, text="Seed Points (m, Binormal):").pack(anchor=tk.W)
        spin_m = ttk.Spinbox(algo_group, from_=1, to=100, textvariable=self.m_seeds_var, width=6)
        spin_m.pack(anchor=tk.W, pady=2)

        # 3. Execution Control Frame
        exec_group = ttk.LabelFrame(parent, text="Segmentation Execution", padding=8)
        exec_group.pack(fill=tk.X, pady=4)

        range_frame = ttk.Frame(exec_group)
        range_frame.pack(fill=tk.X, pady=2)
        ttk.Label(range_frame, text="From:").pack(side=tk.LEFT)
        spin_start = ttk.Spinbox(
            range_frame,
            from_=0,
            to=9999,
            textvariable=self.start_slice_var,
            width=5,
            command=self._on_start_spin_clicked
        )
        spin_start.pack(side=tk.LEFT, padx=4)
        ttk.Label(range_frame, text="To:").pack(side=tk.LEFT, padx=(6, 0))
        self.spin_end = ttk.Spinbox(range_frame, from_=0, to=9999, textvariable=self.end_slice_var, width=5)
        self.spin_end.pack(side=tk.LEFT, padx=4)

        self.btn_run = ttk.Button(
            exec_group,
            text="▶ Run Segmentation",
            command=self.start_segmentation,
            style="Accent.TButton"
        )
        self.btn_run.pack(fill=tk.X, pady=6)

        self.btn_stop = ttk.Button(
            exec_group,
            text="⏹ Stop",
            command=self.stop_segmentation,
            state=tk.DISABLED
        )
        self.btn_stop.pack(fill=tk.X, pady=2)

        # Resume button (prominent when paused for manual segment)
        self.btn_resume = ttk.Button(
            exec_group,
            text="✔ Resume (from manual mask)",
            command=self.resume_segmentation,
            state=tk.DISABLED
        )
        self.btn_resume.pack(fill=tk.X, pady=4)

        # Progress bar
        self.progress_bar = ttk.Progressbar(exec_group, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=4)

        # 4. Display Settings Frame
        display_group = ttk.LabelFrame(parent, text="Display & Overlay", padding=8)
        display_group.pack(fill=tk.X, pady=4)

        chk_overlay = ttk.Checkbutton(
            display_group,
            text="Show Mask Overlay",
            variable=self.show_overlay,
            command=self.render_canvas
        )
        chk_overlay.pack(anchor=tk.W, pady=2)

        ttk.Label(display_group, text="Overlay Opacity:").pack(anchor=tk.W)
        scale_opac = ttk.Scale(
            display_group,
            from_=0.1,
            to=1.0,
            variable=self.overlay_alpha,
            orient=tk.HORIZONTAL,
            command=lambda v: self.render_canvas()
        )
        scale_opac.pack(fill=tk.X, pady=2)

        # Zoom buttons
        zoom_frame = ttk.Frame(display_group)
        zoom_frame.pack(fill=tk.X, pady=4)
        ttk.Button(zoom_frame, text="🔍 Zoom In", command=self.zoom_in, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔍 Zoom Out", command=self.zoom_out, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="↺ Reset", command=self.zoom_reset, width=7).pack(side=tk.LEFT, padx=2)

    def _create_canvas_panel(self, parent: ttk.Frame):
        # Canvas for image viewing and drawing
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Scrubbing Bar
        scrub_frame = ttk.Frame(parent, padding=(4, 6))
        scrub_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_prev = ttk.Button(scrub_frame, text="◀ Prev", command=self.prev_slice, width=7)
        self.btn_prev.pack(side=tk.LEFT, padx=4)

        self.scrub_slider = ttk.Scale(
            scrub_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.on_scrub_slider
        )
        self.scrub_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.btn_next = ttk.Button(scrub_frame, text="Next ▶", command=self.next_slice, width=7)
        self.btn_next.pack(side=tk.LEFT, padx=4)

    def _bind_events(self):
        # Keyboard shortcuts
        self.bind("<Left>", lambda e: self.prev_slice())
        self.bind("<Right>", lambda e: self.next_slice())
        self.bind("<Control-z>", lambda e: self.undo_mask())
        self.bind("<Control-Z>", lambda e: self.undo_mask())

        # Mouse interaction on canvas
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # Right-click panning
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_pan_end)

        # Mouse wheel (Scrubbing / Zooming)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", lambda e: self.render_canvas())

    # =========================================================================
    # Dataset Loading & Navigation
    # =========================================================================

    def on_open_folder(self):
        folder = filedialog.askdirectory(
            title="Select Image Stack Folder",
            initialdir=os.path.abspath("data") if os.path.isdir("data") else os.getcwd()
        )
        if folder:
            self.load_dataset(folder)

    def load_dataset(self, folder_path: str):
        try:
            stack, filenames, folder_name = load_image_stack(folder_path)
        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load image stack:\n{e}")
            return

        self.stack = stack
        self.filenames = filenames
        self.folder_name = folder_name
        self.num_slices = stack.shape[0]
        self.current_slice_idx = 0
        self.masks.clear()
        self.mask_history.clear()

        # Update controls
        self.lbl_folder.config(text=f"📂 {folder_name} ({self.num_slices} slices)")
        self._syncing_slice = True
        self.scrub_slider.config(from_=0, to=self.num_slices - 1)
        self.scrub_slider.set(0)
        self.start_slice_var.set(0)
        self.end_slice_var.set(self.num_slices - 1)
        self.spin_end.config(to=self.num_slices - 1)
        self._syncing_slice = False

        self.zoom_reset()
        self.update_slice_view()
        self.set_status(f"Loaded {self.num_slices} slices from '{folder_name}'.")

    def _on_start_slice_changed(self, *args):
        """Update current displayed slice when user alters start_slice_var."""
        if getattr(self, "_syncing_slice", False):
            return
        try:
            val = self.start_slice_var.get()
        except (tk.TclError, ValueError):
            return
        if self.stack is not None and 0 <= val < self.num_slices:
            if val != self.current_slice_idx:
                self._syncing_slice = True
                self.current_slice_idx = val
                self.scrub_slider.set(val)
                self.update_slice_view()
                self._syncing_slice = False

    def _on_start_spin_clicked(self):
        self.after_idle(self._on_start_slice_changed)

    def update_slice_view(self):
        if self.stack is None:
            return

        idx = self.current_slice_idx
        fname = self.filenames[idx] if idx < len(self.filenames) else f"Slice_{idx}"
        self.lbl_slice_info.config(text=f"Slice: {idx} / {self.num_slices - 1}")
        self.lbl_filename_info.config(text=f"File: {fname}")

        # Update mask statistics label
        if idx in self.masks and np.any(self.masks[idx] > 0):
            stats = characterize_mask(self.stack[idx], self.masks[idx])
            self.lbl_mask_stats.config(
                text=f"Mask: {stats['area']} px | C: ({stats['yc']},{stats['xc']}) | μ={stats['mean']:.1f}, σ={stats['std']:.1f}",
                foreground="#00802b"
            )
        else:
            self.lbl_mask_stats.config(text="Mask: None (Draw to create)", foreground="#888888")

        self.render_canvas()

    def prev_slice(self):
        if self.stack is not None and self.current_slice_idx > 0:
            self.current_slice_idx -= 1
            self.scrub_slider.set(self.current_slice_idx)
            if not self.is_running and not self.is_paused_for_manual:
                if not getattr(self, "_syncing_slice", False):
                    self._syncing_slice = True
                    self.start_slice_var.set(self.current_slice_idx)
                    self._syncing_slice = False
            self.update_slice_view()

    def next_slice(self):
        if self.stack is not None and self.current_slice_idx < self.num_slices - 1:
            self.current_slice_idx += 1
            self.scrub_slider.set(self.current_slice_idx)
            if not self.is_running and not self.is_paused_for_manual:
                if not getattr(self, "_syncing_slice", False):
                    self._syncing_slice = True
                    self.start_slice_var.set(self.current_slice_idx)
                    self._syncing_slice = False
            self.update_slice_view()

    def on_scrub_slider(self, val):
        if self.stack is None:
            return
        new_idx = int(float(val))
        if new_idx != self.current_slice_idx and 0 <= new_idx < self.num_slices:
            self.current_slice_idx = new_idx
            if not self.is_running and not self.is_paused_for_manual:
                if not getattr(self, "_syncing_slice", False):
                    self._syncing_slice = True
                    self.start_slice_var.set(new_idx)
                    self._syncing_slice = False
            self.update_slice_view()

    # =========================================================================
    # Zoom, Pan, Coordinate Transformations
    # =========================================================================

    def zoom_in(self):
        self.zoom_scale = min(self.zoom_scale * 1.25, 10.0)
        self.render_canvas()

    def zoom_out(self):
        self.zoom_scale = max(self.zoom_scale / 1.25, 0.2)
        self.render_canvas()

    def zoom_reset(self):
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.render_canvas()

    def on_mouse_wheel(self, event):
        if self.stack is None:
            return
        # Ctrl + MouseWheel -> Zoom
        if event.state & 0x0004:
            if event.delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        # MouseWheel -> Scrub Slices
        else:
            if event.delta > 0:
                self.prev_slice()
            else:
                self.next_slice()

    def on_pan_start(self, event):
        self.is_panning = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_pan_drag(self, event):
        if self.is_panning:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            self.pan_x += dx
            self.pan_y += dy
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            self.render_canvas()

    def on_pan_end(self, event):
        self.is_panning = False

    def canvas_to_img_coords(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        """Map canvas coordinates to image (y, x) coordinates."""
        if self.stack is None:
            return None
        H, W = self.stack.shape[1], self.stack.shape[2]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # Canvas center offset
        disp_w = int(W * self.zoom_scale)
        disp_h = int(H * self.zoom_scale)
        origin_x = (cw - disp_w) // 2 + self.pan_x
        origin_y = (ch - disp_h) // 2 + self.pan_y

        ix = int((cx - origin_x) / self.zoom_scale)
        iy = int((cy - origin_y) / self.zoom_scale)

        if 0 <= ix < W and 0 <= iy < H:
            return iy, ix
        return None

    # =========================================================================
    # Rendering & Mask Overlay
    # =========================================================================

    def render_canvas(self):
        if self.stack is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="No dataset loaded. Click 'Open Stack' to begin.",
                fill="#888888",
                font=("Segoe UI", 12)
            )
            return

        slice_data = self.stack[self.current_slice_idx]
        H, W = slice_data.shape

        # Normalize 8-bit / 16-bit to uint8 RGB for display
        if slice_data.dtype != np.uint8:
            s_min, s_max = float(slice_data.min()), float(slice_data.max())
            if s_max > s_min:
                normalized = np.clip((slice_data - s_min) / (s_max - s_min) * 255.0, 0, 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(slice_data, dtype=np.uint8)
        else:
            normalized = slice_data

        rgb = np.stack([normalized] * 3, axis=-1)

        # Apply Mask Overlay if enabled
        mask = self.masks.get(self.current_slice_idx, None)
        if self.show_overlay.get() and mask is not None and np.any(mask > 0):
            alpha = self.overlay_alpha.get()
            # Green translucent overlay: RGB (40, 220, 60)
            overlay_color = np.array([40, 220, 60], dtype=np.float32)
            mask_bool = (mask > 0)
            blended = rgb[mask_bool].astype(np.float32) * (1.0 - alpha) + overlay_color * alpha
            rgb[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)

        # Convert to PIL and resize according to zoom
        pil_img = Image.fromarray(rgb)
        disp_w = max(1, int(W * self.zoom_scale))
        disp_h = max(1, int(H * self.zoom_scale))

        resample_mode = Image.Resampling.NEAREST if self.zoom_scale >= 2.0 else Image.Resampling.BILINEAR
        pil_resized = pil_img.resize((disp_w, disp_h), resample=resample_mode)
        self.photo_img = ImageTk.PhotoImage(pil_resized)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        origin_x = (cw - disp_w) // 2 + self.pan_x
        origin_y = (ch - disp_h) // 2 + self.pan_y

        self.canvas.delete("all")
        self.canvas.create_image(origin_x, origin_y, anchor=tk.NW, image=self.photo_img)

        # Draw centroid marker if present
        if mask is not None and np.any(mask > 0):
            ys, xs = np.where(mask > 0)
            yc, xc = int(np.mean(ys)), int(np.mean(xs))
            cx = origin_x + int(xc * self.zoom_scale)
            cy = origin_y + int(yc * self.zoom_scale)
            r = 4
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#ff3333", width=2)
            self.canvas.create_line(cx - r - 3, cy, cx + r + 3, cy, fill="#ff3333", width=1)
            self.canvas.create_line(cx, cy - r - 3, cx, cy + r + 3, fill="#ff3333", width=1)

        # Draw lasso temporary line if in progress
        if self.lasso_points and len(self.lasso_points) > 1:
            flat_pts = []
            for ly, lx in self.lasso_points:
                px = origin_x + int(lx * self.zoom_scale)
                py = origin_y + int(ly * self.zoom_scale)
                flat_pts.extend([px, py])
            self.canvas.create_line(flat_pts, fill="#00ffcc", width=2)

    # =========================================================================
    # Interactive Mask Drawing Tools
    # =========================================================================

    def _get_or_create_mask(self) -> np.ndarray:
        if self.current_slice_idx not in self.masks:
            H, W = self.stack.shape[1], self.stack.shape[2]
            self.masks[self.current_slice_idx] = np.zeros((H, W), dtype=np.uint8)
        return self.masks[self.current_slice_idx]

    def _save_history(self):
        idx = self.current_slice_idx
        if idx not in self.mask_history:
            self.mask_history[idx] = []
        current = self.masks.get(idx, None)
        if current is not None:
            self.mask_history[idx].append(current.copy())
            if len(self.mask_history[idx]) > 20:
                self.mask_history[idx].pop(0)

    def undo_mask(self):
        idx = self.current_slice_idx
        if idx in self.mask_history and self.mask_history[idx]:
            prev = self.mask_history[idx].pop()
            self.masks[idx] = prev
            self.update_slice_view()
            self.set_status(f"Undid last stroke on slice {idx}.")

    def clear_current_mask(self):
        idx = self.current_slice_idx
        if idx in self.masks:
            self._save_history()
            H, W = self.stack.shape[1], self.stack.shape[2]
            self.masks[idx] = np.zeros((H, W), dtype=np.uint8)
            self.update_slice_view()
            self.set_status(f"Cleared mask on slice {idx}.")

    def clear_all_masks(self):
        """Clear all generated masks across the entire image stack with user confirmation."""
        active_count = len([k for k, m in self.masks.items() if np.any(m > 0)])
        if active_count == 0:
            messagebox.showinfo("No Masks", "There are currently no masks to clear.")
            return

        if messagebox.askyesno(
            "Clear All Masks",
            f"Are you sure you want to clear all {active_count} mask(s) across the entire stack?\n\n"
            "This will reset all masks to empty."
        ):
            self.masks.clear()
            self.mask_history.clear()
            self.update_slice_view()
            self.set_status(f"Cleared all {active_count} mask(s) across the stack.")

    def on_canvas_press(self, event):
        if self.stack is None:
            return
        coords = self.canvas_to_img_coords(event.x, event.y)
        if coords is None:
            return

        mode = self.tool_mode.get()
        if mode in ("brush", "eraser"):
            self._save_history()
            self._apply_brush(coords[0], coords[1], is_erase=(mode == "eraser"))
        elif mode == "lasso":
            self._save_history()
            self.lasso_points = [coords]
            self.render_canvas()

    def on_canvas_drag(self, event):
        if self.stack is None:
            return
        coords = self.canvas_to_img_coords(event.x, event.y)
        if coords is None:
            return

        mode = self.tool_mode.get()
        if mode in ("brush", "eraser"):
            self._apply_brush(coords[0], coords[1], is_erase=(mode == "eraser"))
        elif mode == "lasso":
            self.lasso_points.append(coords)
            self.render_canvas()

    def on_canvas_release(self, event):
        if self.stack is None:
            return
        mode = self.tool_mode.get()
        if mode == "lasso" and len(self.lasso_points) >= 3:
            self._fill_lasso()
            self.lasso_points = []
            self.update_slice_view()

    def _apply_brush(self, iy: int, ix: int, is_erase: bool = False):
        mask = self._get_or_create_mask()
        H, W = mask.shape
        r = self.brush_radius.get()
        val = 0 if is_erase else 1

        y_min = max(0, iy - r)
        y_max = min(H, iy + r + 1)
        x_min = max(0, ix - r)
        x_max = min(W, ix + r + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (yy - iy) ** 2 + (xx - ix) ** 2
        circle_mask = dist_sq <= (r ** 2)

        mask[y_min:y_max, x_min:x_max][circle_mask] = val
        self.update_slice_view()

    def _fill_lasso(self):
        mask = self._get_or_create_mask()
        H, W = mask.shape
        # Create a PIL image to draw the filled polygon
        poly_img = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(poly_img)
        # Convert (iy, ix) to (x, y) for PIL
        pts = [(x, y) for (y, x) in self.lasso_points]
        draw.polygon(pts, outline=1, fill=1)
        lasso_arr = np.array(poly_img, dtype=np.uint8)

        # Merge with current mask
        mask[lasso_arr == 1] = 1
        # Fill holes in the newly added region
        self.masks[self.current_slice_idx] = fill_mask_holes(mask)

    # =========================================================================
    # Algorithm Execution & Empty Mask Handling
    # =========================================================================

    def start_segmentation(self):
        if self.stack is None:
            messagebox.showwarning("No Dataset", "Please load an image stack first.")
            return

        start_idx = self.start_slice_var.get()
        end_idx = self.end_slice_var.get()

        if not (0 <= start_idx < self.num_slices):
            messagebox.showwarning("Invalid Range", f"Start slice {start_idx} is out of bounds.")
            return

        if not (start_idx <= end_idx < self.num_slices):
            messagebox.showwarning("Invalid Range", f"End slice {end_idx} must be >= start slice.")
            return

        # Check that starting slice has an initial mask
        if start_idx not in self.masks or np.count_nonzero(self.masks[start_idx]) == 0:
            self.current_slice_idx = start_idx
            self.scrub_slider.set(start_idx)
            self.update_slice_view()
            messagebox.showinfo(
                "Initial Mask Required",
                f"Please draw the initial mask by hand on starting slice {start_idx} using the brush or lasso tool."
            )
            return

        # Step 1: Characterize starting mask (mean, std, centroid yc, xc)
        initial_stats = characterize_mask(self.stack[start_idx], self.masks[start_idx])
        if not initial_stats["valid"]:
            messagebox.showerror("Invalid Mask", "Starting mask contains no foreground pixels.")
            return

        self.active_yc = initial_stats["yc"]
        self.active_xc = initial_stats["xc"]
        self.active_mean = initial_stats["mean"]
        self.active_std = initial_stats["std"]
        self.active_spatial_std_y = initial_stats.get("spatial_std_y", 2.0)
        self.active_spatial_std_x = initial_stats.get("spatial_std_x", 2.0)

        # Configure state for propagation
        self.is_running = True
        self.is_paused_for_manual = False
        self.running_current_slice = start_idx + 1

        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)

        self.progress_bar.config(maximum=end_idx - start_idx, value=0)
        self.set_status(f"Started segmentation from slice {start_idx} to {end_idx}...")
        self.after(50, self._segmentation_step)

    def _segmentation_step(self):
        if not self.is_running:
            return

        end_idx = self.end_slice_var.get()
        start_idx = self.start_slice_var.get()
        slice_idx = self.running_current_slice

        if slice_idx > end_idx:
            # Reached end successfully
            self.is_running = False
            self.btn_run.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.progress_bar.config(value=end_idx - start_idx)
            self.set_status(f"Segmentation completed successfully through slice {end_idx}.")
            if messagebox.askyesno("Completed", f"Segmentation finished through slice {end_idx}!\nSave generated masks now?"):
                self.on_save_masks()
            return

        # Execute Pluto adaptive algorithm on current slice
        current_slice = self.stack[slice_idx]
        alpha = self.alpha_var.get()
        k_std = self.k_std_var.get()
        max_iter = self.max_iter_var.get()
        m_seeds = max(1, self.m_seeds_var.get())

        res = propagate_slice(
            current_slice=current_slice,
            prev_yc=self.active_yc,
            prev_xc=self.active_xc,
            mean=self.active_mean,
            std=self.active_std,
            alpha=alpha,
            n_iterations_max=max_iter,
            k_std=k_std,
            m_seeds=m_seeds,
            spatial_std_y=self.active_spatial_std_y,
            spatial_std_x=self.active_spatial_std_x,
        )

        # Check for empty / failed mask
        if res["is_empty"]:
            self._handle_empty_mask(slice_idx, res["reason"])
            return

        # Valid mask found: update state
        self.masks[slice_idx] = res["mask"]
        self.active_yc = res["yc"]
        self.active_xc = res["xc"]
        self.active_mean = res["mean"]
        self.active_std = res["std"]
        self.active_spatial_std_y = res.get("spatial_std_y", 2.0)
        self.active_spatial_std_x = res.get("spatial_std_x", 2.0)

        # Update view
        self.current_slice_idx = slice_idx
        self.scrub_slider.set(slice_idx)
        self.update_slice_view()
        self.progress_bar.config(value=slice_idx - start_idx)

        self.set_status(
            f"Segmented slice {slice_idx}/{end_idx} | C: ({res['yc']},{res['xc']}) | "
            f"Area: {res['area']} px | μ={res['mean']:.1f}, σ={res['std']:.1f}"
        )

        # Schedule next slice step
        self.running_current_slice += 1
        self.after(20, self._segmentation_step)

    def _handle_empty_mask(self, slice_idx: int, reason: str):
        """
        Handle empty mask:
        - Pause execution immediately.
        - Ask user to choose between:
          1. Segment current slice by hand.
          2. Go back {n} images and segment that image, where n is selected by the user.
          3. End the program there.
        """
        self.is_running = False
        self.current_slice_idx = slice_idx
        self.scrub_slider.set(slice_idx)
        self.update_slice_view()

        fname = self.filenames[slice_idx] if slice_idx < len(self.filenames) else f"Slice {slice_idx}"
        max_back = max(1, slice_idx)

        dialog = tk.Toplevel(self)
        dialog.title("Empty Mask Encountered")
        dialog.geometry("540x380")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=f"⚠️ Empty mask encountered at Slice {slice_idx} ({fname})!",
            font=("Segoe UI", 10, "bold"),
            foreground="#cc0000"
        ).pack(anchor=tk.W, pady=(0, 4))

        ttk.Label(
            frame,
            text=f"Reason: {reason}\n\nPlease choose how to proceed:",
            wraplength=500
        ).pack(anchor=tk.W, pady=(0, 8))

        choice_var = tk.StringVar(value="stop")
        back_n_var = tk.IntVar(value=min(1, max_back))

        def choose(choice):
            choice_var.set(choice)
            dialog.destroy()

        # --- Option 1: Segment current slice by hand ---
        opt1_frame = ttk.LabelFrame(frame, text="Option 1: Segment Current Image by Hand", padding=8)
        opt1_frame.pack(fill=tk.X, pady=4)
        btn_manual = ttk.Button(
            opt1_frame,
            text=f"✏️ Segment Slice {slice_idx} by Hand",
            command=lambda: choose("manual_current"),
            style="Accent.TButton"
        )
        btn_manual.pack(fill=tk.X)

        # --- Option 2: Go back {n} images and segment that image ---
        opt2_frame = ttk.LabelFrame(frame, text="Option 2: Go Back {n} Images & Segment", padding=8)
        opt2_frame.pack(fill=tk.X, pady=4)

        back_controls = ttk.Frame(opt2_frame)
        back_controls.pack(fill=tk.X, pady=2)

        ttk.Label(back_controls, text="Go back:").pack(side=tk.LEFT)
        spin_n = ttk.Spinbox(
            back_controls,
            from_=1,
            to=max(1, max_back),
            textvariable=back_n_var,
            width=5
        )
        spin_n.pack(side=tk.LEFT, padx=4)
        ttk.Label(back_controls, text="image(s)").pack(side=tk.LEFT, padx=(0, 8))

        init_target = max(0, slice_idx - back_n_var.get())
        init_tfname = self.filenames[init_target] if init_target < len(self.filenames) else ""
        lbl_target_info = ttk.Label(
            back_controls,
            text=f"➔ Target: Slice {init_target} ({init_tfname})",
            foreground="#0066cc",
            font=("Segoe UI", 9, "bold")
        )
        lbl_target_info.pack(side=tk.LEFT, padx=4)

        def update_target_lbl(*args):
            try:
                n_val = back_n_var.get()
                t_slice = max(0, slice_idx - n_val)
                t_fname = self.filenames[t_slice] if t_slice < len(self.filenames) else ""
                lbl_target_info.config(text=f"➔ Target: Slice {t_slice} ({t_fname})")
            except (tk.TclError, ValueError):
                pass

        back_n_var.trace_add("write", update_target_lbl)

        btn_go_back = ttk.Button(
            opt2_frame,
            text="⏪ Go Back & Segment That Image",
            command=lambda: choose("go_back")
        )
        btn_go_back.pack(fill=tk.X, pady=(4, 0))

        if slice_idx == 0:
            spin_n.config(state=tk.DISABLED)
            btn_go_back.config(state=tk.DISABLED)

        # --- Option 3: End program here ---
        opt3_frame = ttk.LabelFrame(frame, text="Option 3: End Program", padding=8)
        opt3_frame.pack(fill=tk.X, pady=4)
        btn_stop = ttk.Button(
            opt3_frame,
            text="⏹ End Program There (Stop Execution)",
            command=lambda: choose("stop")
        )
        btn_stop.pack(fill=tk.X)

        self.wait_window(dialog)

        action = choice_var.get()
        if action == "manual_current":
            self.is_paused_for_manual = True
            self.btn_run.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_resume.config(state=tk.NORMAL)
            self.set_status(f"PAUSED: Please draw mask for Slice {slice_idx}, then click '✔ Resume'.")
            messagebox.showinfo(
                "Segment by Hand",
                f"Draw the mask for Slice {slice_idx} using the brush or lasso.\n"
                f"When finished, click the green '✔ Resume' button to continue automatic propagation."
            )
        elif action == "go_back":
            try:
                n_val = int(back_n_var.get())
            except (ValueError, tk.TclError):
                n_val = 1
            n_val = max(1, min(n_val, max_back))
            target_slice = max(0, slice_idx - n_val)

            # Invalidate any masks produced after target_slice
            for s in list(self.masks.keys()):
                if s > target_slice:
                    del self.masks[s]

            self.current_slice_idx = target_slice
            self.scrub_slider.set(target_slice)
            self.update_slice_view()

            self.is_paused_for_manual = True
            self.btn_run.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_resume.config(state=tk.NORMAL)
            self.set_status(
                f"PAUSED: Jumped back {n_val} image(s) to Slice {target_slice}. Please edit/redraw mask, then click '✔ Resume'."
            )
            messagebox.showinfo(
                "Segment by Hand",
                f"Moved back {n_val} image(s) to Slice {target_slice}.\n\n"
                f"Please edit or redraw the mask for Slice {target_slice}.\n"
                f"When finished, click '✔ Resume' to restart automatic propagation from Slice {target_slice}."
            )
        else:
            # End program there
            self.stop_segmentation()
            self.masks.pop(slice_idx, None)
            self.set_status(f"Stopped at slice {slice_idx}.")
            if messagebox.askyesno("Save Masks", f"Execution ended at slice {slice_idx}.\nSave masks generated so far ({len(self.masks)} masks)?"):
                self.on_save_masks()

    def resume_segmentation(self):
        """Resume automatic propagation after user has drawn a mask by hand."""
        if not self.is_paused_for_manual:
            return

        slice_idx = self.current_slice_idx
        mask = self.masks.get(slice_idx, None)

        if mask is None or np.count_nonzero(mask) == 0:
            messagebox.showwarning(
                "Mask Required",
                f"Please draw a mask on Slice {slice_idx} before clicking Resume."
            )
            return

        # Re-characterize the newly hand-drawn mask
        stats = characterize_mask(self.stack[slice_idx], mask)
        if not stats["valid"]:
            messagebox.showerror("Error", "Drawn mask contains no foreground pixels.")
            return

        self.active_yc = stats["yc"]
        self.active_xc = stats["xc"]
        self.active_mean = stats["mean"]
        self.active_std = stats["std"]
        self.active_spatial_std_y = stats.get("spatial_std_y", 2.0)
        self.active_spatial_std_x = stats.get("spatial_std_x", 2.0)

        # Discard any stale masks ahead of slice_idx
        for s in list(self.masks.keys()):
            if s > slice_idx:
                del self.masks[s]

        # Resume propagation to the next slice
        self.is_paused_for_manual = False
        self.is_running = True
        self.running_current_slice = slice_idx + 1

        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)

        self.set_status(f"Resumed segmentation from Slice {slice_idx}...")
        self.after(50, self._segmentation_step)

    def stop_segmentation(self):
        self.is_running = False
        self.is_paused_for_manual = False
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.DISABLED)
        self.set_status("Segmentation stopped.")

    # =========================================================================
    # Saving Masks
    # =========================================================================

    def on_save_masks(self):
        if not self.masks or not any(np.any(m > 0) for m in self.masks.values()):
            messagebox.showwarning("No Masks", "No masks have been generated yet to save.")
            return

        folder = self.folder_name if self.folder_name else "dataset"
        try:
            saved_paths = save_masks(
                masks_dict=self.masks,
                folder_name=folder,
                filenames=self.filenames,
                base_dir="masks"
            )
            out_dir = os.path.dirname(saved_paths[0]) if saved_paths else os.path.join("masks", folder)
            messagebox.showinfo(
                "Masks Saved",
                f"Successfully saved {len(saved_paths)} binary PNG masks to:\n{os.path.abspath(out_dir)}"
            )
            self.set_status(f"Saved {len(saved_paths)} masks to ./masks/{folder}/")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save masks:\n{e}")

    # =========================================================================
    # Helpers
    # =========================================================================

    def set_status(self, msg: str):
        self.status_bar.config(text=msg)

    def show_about(self):
        messagebox.showinfo(
            "About Pluto",
            "Pluto - Adaptive 3D Image Segmentation\n\n"
            "An interactive desktop application for semi-automatic canalicular "
            "segmentation in 3D microscopy and micro-CT image stacks.\n\n"
            "Based on the adaptive region-growing and moving-average logic of pluto.py."
        )


def main():
    app = PlutoApp()
    app.mainloop()


if __name__ == "__main__":
    main()

