# Pluto: Adaptive 3D Image Segmentation

An interactive desktop application for semi-automatic 3D image stack segmentation (e.g. canalicular structures in micro-CT and confocal microscopy), wrapping the custom adaptive segmentation logic from `pluto.py`.

---

## Features

- **Data Loading**: Loads 2D image stacks (TIFF, PNG, etc.) from `./data/{folder_name}` sorted numerically by filename.
- **Interactive UI**:
  - Desktop GUI built with Tkinter and Pillow.
  - Scrub through slices with the slider, Left/Right arrow keys, or mouse wheel.
  - Zoom and Pan navigation (Zoom In/Out buttons, or right-click drag to pan).
  - Manual mask drawing tools: **Brush**, **Lasso / Polygon**, and **Eraser** with adjustable radius.
  - Real-time translucent mask overlay with adjustable opacity.
- **Adaptive Segmentation Propagation**:
  - Propagates segmentation across consecutive slices using the `pluto.py` algorithm:
    - Initial mask statistical profile ($\mu_0, \sigma_0$, centroid $(y_c, x_c)$).
    - Iterative dilation bounded by $[\mu - k\sigma, \mu + k\sigma]$.
    - Exponential moving average updates ($\alpha$) for intensity mean and standard deviation.
    - Morphological hole filling.
- **Empty Mask Handling**:
  - Detects if a mask vanishes or fails to expand from its seed.
  - Pauses execution and prompts the user:
    - **Stop Execution**: Terminates propagation and preserves generated masks.
    - **Segment by Hand**: Allows the user to draw the mask on that slice, then click **Resume** to continue propagation automatically.
- **Mask Export**:
  - Saves all masks strictly as **binary PNG files** (`0` background, `255` foreground) into `./masks/{folder_name}`.

---

## Project Structure

```text
Pluto/
  ├── requirements.txt      # Project dependencies
  ├── main.py               # Interactive desktop application
  ├── Utils/
  │   ├── __init__.py
  │   └── utils.py          # Core I/O, segmentation, and saving logic
  ├── data/                 # Input image stacks (e.g. data/cropped_3_3_1)
  ├── masks/                # Output binary PNG masks (e.g. masks/cropped_3_3_1)
  ├── tests/
  │   └── test_pluto.py     # Unit and integration test suite
  └── pluto.py              # Original reference script
```

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Quickstart & Usage

### 1. Launch the Desktop Application

```bash
python main.py
```

### 2. Workflow

1. **Load Stack**: Click **"Open Stack"** (or use the preloaded `cropped_3_3_1` stack).
2. **Select Starting Slice**: Use the bottom slider, arrow keys, or mouse wheel to navigate to the slice you wish to start on (e.g., slice 0).
3. **Draw Initial Mask**:
   - Choose **Brush** or **Lasso / Poly** from the toolbar.
   - Draw over the target canalicular structure.
   - Adjust the radius or use **Undo** / **Eraser** if needed.
4. **Run Segmentation**:
   - Set the range (default is from the current slice to the end of the stack).
   - Adjust hyperparameters if desired:
     - `Alpha`: Exponential moving average weight (default: `0.8`).
     - `k * std`: Multiplier for standard deviation bounds (default: `2.0`).
     - `Max Iterations`: Max dilation steps (default: `40`).
   - Click **"Run Segmentation"**.
5. **Handling Pauses / Empty Masks**:
   - If an empty mask or constriction is encountered, a modal dialog appears.
   - Choose **"Segment by Hand"** to draw the mask on that slice, then click **"Resume"**.
   - Or choose **"Stop Execution"** to finalize.
6. **Save Masks**:
   - Click **"Save Masks"** (or accept the automatic prompt when finished).
   - All masks will be written to `./masks/{folder_name}/*.png` as binary PNG files.

---

## Running Tests

Run the test suite to verify data loading, segmentation logic, and binary saving:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

