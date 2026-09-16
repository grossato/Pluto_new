"""
Utils module for Pluto Adaptive 3D Image Segmentation.

Contains functions for:
- Loading image stacks with natural numerical ordering.
- Characterizing masks (mean, std, centroid).
- Propagating masks across consecutive slices using the Pluto algorithm.
- Saving generated binary masks strictly as binary PNG files.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes


def natural_sort_key(s: str) -> List[Union[int, str]]:
    """
    Sort strings containing embedded numbers in natural human order.
    Example: 'file1.tif', 'file2.tif', 'file10.tif'
    """
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", s)
    ]


def load_image_stack(
    folder_path: str,
) -> Tuple[np.ndarray, List[str], str]:
    """
    Load a stack of 2D images (typically TIFF files) from a directory,
    sorted numerically by filename.

    :param folder_path: Path to the directory containing image slices.
    :return: (stack, filenames, folder_name)
             - stack: 3D numpy array of shape (N, H, W)
             - filenames: List of base filenames in sorted order
             - folder_name: Name of the folder
    :raises FileNotFoundError: If folder does not exist or has no valid images.
    :raises ValueError: If image dimensions are inconsistent.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: '{folder_path}'")

    valid_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    all_entries = os.listdir(folder_path)
    image_files = [
        f for f in all_entries
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not image_files:
        raise FileNotFoundError(
            f"No valid image files ({', '.join(valid_extensions)}) found in '{folder_path}'"
        )

    # Sort numerically by filename
    image_files.sort(key=natural_sort_key)

    slices = []
    base_shape = None

    for fname in image_files:
        fpath = os.path.join(folder_path, fname)
        with Image.open(fpath) as img:
            # Handle potential multi-channel or unusual formats
            if img.mode in ("RGB", "RGBA"):
                img_gray = img.convert("L")
                arr = np.array(img_gray)
            else:
                arr = np.array(img)

        if base_shape is None:
            base_shape = arr.shape
        elif arr.shape != base_shape:
            raise ValueError(
                f"Image '{fname}' has shape {arr.shape}, expected {base_shape}."
            )

        slices.append(arr)

    stack = np.stack(slices, axis=0)
    folder_name = os.path.basename(os.path.normpath(folder_path))
    return stack, image_files, folder_name


def characterize_mask(
    slice_img: np.ndarray,
    binary_mask: np.ndarray,
) -> Dict[str, Union[int, float, bool, None]]:
    """
    Characterize a binary mask on a given 2D slice:
    Computes centroid (yc, xc), mean intensity, and standard deviation.

    :param slice_img: 2D numpy array representing the grayscale image.
    :param binary_mask: 2D numpy array with 1 for foreground, 0 for background.
    :return: Dictionary containing 'yc', 'xc', 'mean', 'std', 'area', 'valid'.
    """
    foreground_indices = np.where(binary_mask > 0)
    num_pixels = len(foreground_indices[0])

    if num_pixels == 0:
        return {
            "yc": None,
            "xc": None,
            "mean": None,
            "std": None,
            "area": 0,
            "valid": False,
        }

    ys, xs = foreground_indices
    yc = int(np.round(np.mean(ys)))
    xc = int(np.round(np.mean(xs)))

    spatial_std_y = max(1.0, float(np.std(ys))) if len(ys) > 1 else 2.0
    spatial_std_x = max(1.0, float(np.std(xs))) if len(xs) > 1 else 2.0

    foreground_values = slice_img[foreground_indices]
    mean = float(np.mean(foreground_values))
    std = float(np.std(foreground_values))

    return {
        "yc": yc,
        "xc": xc,
        "mean": mean,
        "std": std,
        "spatial_std_y": spatial_std_y,
        "spatial_std_x": spatial_std_x,
        "area": int(num_pixels),
        "valid": True,
    }


def fill_mask_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill interior holes in a binary mask using morphological hole filling.

    :param binary_mask: 2D binary numpy array.
    :return: 2D binary numpy array with holes filled (uint8).
    """
    return binary_fill_holes(binary_mask > 0).astype(np.uint8)


def propagate_slice(
    current_slice: np.ndarray,
    prev_yc: int,
    prev_xc: int,
    mean: float,
    std: float,
    alpha: float = 0.8,
    n_iterations_max: int = 40,
    k_std: float = 2.0,
    m_seeds: int = 5,
    spatial_std_y: float = 2.0,
    spatial_std_x: float = 2.0,
) -> Dict[str, Union[np.ndarray, int, float, bool, str, None]]:
    """
    Apply the Pluto adaptive segmentation algorithm to propagate a mask to the current slice:
    - Seeds on the slice are m points surrounding the center of mass (prev_yc, prev_xc)
      extracted at random from a binormal distribution centered at the center of mass.
    - Iteratively dilates the candidate mask and excludes pixels outside [mean - k*std, mean + k*std].
    - Stops if area stops growing or max iterations reached.
    - If empty or failed to grow, returns is_empty=True.
    - Otherwise, updates centroid, exponential moving average of mean and std, and fills holes.

    :param current_slice: 2D image array of current slice.
    :param prev_yc: Y coordinate of center of mass of previous mask.
    :param prev_xc: X coordinate of center of mass of previous mask.
    :param mean: Moving average mean intensity.
    :param std: Moving average standard deviation of intensity.
    :param alpha: Moving average factor (default 0.8).
    :param n_iterations_max: Max growth iterations (default 40).
    :param k_std: Multiplier for standard deviation interval (default 2.0).
    :param m_seeds: Number of seed points surrounding the center of mass (default 5).
    :param spatial_std_y: Vertical spatial spread of the binormal distribution.
    :param spatial_std_x: Horizontal spatial spread of the binormal distribution.
    :return: Dictionary containing 'mask', 'yc', 'xc', 'mean', 'std', 'spatial_std_y', 'spatial_std_x', 'area', 'is_empty', 'reason'.
    """
    H, W = current_slice.shape
    empty_mask = np.zeros((H, W), dtype=np.uint8)

    # Check bounds of center of mass
    if prev_yc is None or prev_xc is None or not (0 <= prev_yc < H and 0 <= prev_xc < W):
        return {
            "mask": empty_mask,
            "yc": prev_yc,
            "xc": prev_xc,
            "mean": mean,
            "std": std,
            "spatial_std_y": spatial_std_y,
            "spatial_std_x": spatial_std_x,
            "area": 0,
            "is_empty": True,
            "reason": f"Center of mass ({prev_yc}, {prev_xc}) is out of bounds for image shape {current_slice.shape}.",
        }

    # Initial seed points: m points surrounding center of mass from binormal distribution
    mask_slice = np.zeros((H, W), dtype=np.uint8)
    mask_slice[prev_yc, prev_xc] = 1

    if m_seeds > 1:
        # Sample m-1 additional points from 2D binormal distribution centered at (prev_yc, prev_xc)
        sampled_ys = np.random.normal(loc=prev_yc, scale=max(1.0, spatial_std_y), size=m_seeds - 1)
        sampled_xs = np.random.normal(loc=prev_xc, scale=max(1.0, spatial_std_x), size=m_seeds - 1)
        valid_ys = np.clip(np.round(sampled_ys).astype(int), 0, H - 1)
        valid_xs = np.clip(np.round(sampled_xs).astype(int), 0, W - 1)
        mask_slice[valid_ys, valid_xs] = 1

    initial_seed_count = int(np.count_nonzero(mask_slice))
    candidate_region_area = initial_seed_count

    low_bound = mean - k_std * std
    high_bound = mean + k_std * std

    # Iterative dilation and threshold exclusion
    for _ in range(n_iterations_max):
        tmp_pre_mask = binary_dilation(mask_slice, iterations=1)
        # Identify pixels within candidate region that fall outside intensity bounds
        points_to_exclude = np.logical_or(
            current_slice[tmp_pre_mask] < low_bound,
            current_slice[tmp_pre_mask] > high_bound,
        )
        region_ys, region_xs = np.where(tmp_pre_mask == 1)
        tmp_pre_mask[region_ys[points_to_exclude], region_xs[points_to_exclude]] = 0

        new_area = int(np.count_nonzero(tmp_pre_mask))
        if new_area > candidate_region_area:
            mask_slice = tmp_pre_mask.astype(np.uint8)
            candidate_region_area = new_area
        else:
            break

    # Empty mask check: if mask vanished or failed to grow beyond seeds
    final_nonzero = int(np.count_nonzero(mask_slice))
    if final_nonzero <= max(1, initial_seed_count):
        return {
            "mask": empty_mask,
            "yc": prev_yc,
            "xc": prev_xc,
            "mean": mean,
            "std": std,
            "spatial_std_y": spatial_std_y,
            "spatial_std_x": spatial_std_x,
            "area": 0,
            "is_empty": True,
            "reason": (
                f"Candidate region failed to grow from {initial_seed_count} binormal seed(s) around center of mass ({prev_yc}, {prev_xc}) "
                f"under intensity bounds [{low_bound:.2f}, {high_bound:.2f}]."
            ),
        }

    # Extract new center of mass and spatial spreads
    ys, xs = np.where(mask_slice == 1)
    new_yc = int(np.round(np.mean(ys)))
    new_xc = int(np.round(np.mean(xs)))
    new_spatial_std_y = max(1.0, float(np.std(ys))) if len(ys) > 1 else spatial_std_y
    new_spatial_std_x = max(1.0, float(np.std(xs))) if len(xs) > 1 else spatial_std_x

    # Update mean and std with exponential moving average
    foreground_vals = current_slice[mask_slice == 1]
    curr_mean = float(np.mean(foreground_vals))
    curr_std = float(np.std(foreground_vals))

    new_mean = alpha * mean + (1.0 - alpha) * curr_mean
    new_std = alpha * std + (1.0 - alpha) * curr_std

    # Fill holes in the mask
    mask_slice = binary_fill_holes(mask_slice).astype(np.uint8)
    final_area = int(np.count_nonzero(mask_slice))

    return {
        "mask": mask_slice,
        "yc": new_yc,
        "xc": new_xc,
        "mean": new_mean,
        "std": new_std,
        "spatial_std_y": new_spatial_std_y,
        "spatial_std_x": new_spatial_std_x,
        "area": final_area,
        "is_empty": False,
        "reason": None,
    }


def save_masks(
    masks_dict: Dict[int, np.ndarray],
    folder_name: str,
    filenames: Optional[List[str]] = None,
    base_dir: str = "masks",
) -> List[str]:
    """
    Save generated masks strictly as binary PNG files into `./masks/{folder_name}`.

    :param masks_dict: Dictionary mapping slice_index -> 2D binary numpy array.
    :param folder_name: Name of the dataset subfolder (e.g. 'cropped_3_3_1').
    :param filenames: Optional list of original filenames to preserve matching names.
    :param base_dir: Base directory for saving masks (default 'masks').
    :return: List of written file paths.
    """
    out_dir = os.path.join(base_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    saved_paths = []
    for idx in sorted(masks_dict.keys()):
        mask = masks_dict[idx]
        # Strictly binary PNG: 0 for background, 255 for foreground
        binary_png = ((mask > 0).astype(np.uint8) * 255)

        if filenames and 0 <= idx < len(filenames):
            base_name = os.path.splitext(filenames[idx])[0] + ".png"
        else:
            base_name = f"mask_{idx:04d}.png"

        save_path = os.path.join(out_dir, base_name)
        img = Image.fromarray(binary_png, mode="L")
        img.save(save_path, format="PNG")
        saved_paths.append(save_path)

    return saved_paths

