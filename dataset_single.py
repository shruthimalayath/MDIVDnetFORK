from torch.utils.data import Dataset
import glob
import os
import cv2
import torch
import numpy as np
import random
import re
import tifffile

"""
Clean-only thermal dataset loader for 16-bit clean image sequences.

This dataset preserves the raw 16-bit intensity values and applies the same
random temporal window selection and spatial cropping used in the training flow.
"""


def _frame_numeric_key(basename: str) -> int:
    """Extract the first integer in filename for numeric sorting (fallback 0)."""
    m = re.search(r"(\d+)", basename)
    return int(m.group(1)) if m else 0


def _read_thermal_image(path, gray_mode=True):
    if path.lower().endswith(('.tif', '.tiff')):
        img = tifffile.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise IOError(f"Could not read image: {path}")

    if img.ndim == 2:
        img = img[..., None]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[..., None]
    elif img.ndim == 3 and img.shape[2] == 3:
        if gray_mode:
            img = img[..., 0:1]
    else:
        raise ValueError(f"Unsupported image shape {img.shape} for {path}")

    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    return np.ascontiguousarray(img)

class CleanThermalDataset(Dataset):
    def __init__(
        self,
        clean_root: str,
        patch_size: int = 96,
        temp_patch_size: int = 5,
        epoch_size: int = 256000,
        preload: bool = True,
        progress: bool = False,
        gray_mode: bool = True,
    ):
        self.clean_root = clean_root
        self.patch_size = patch_size
        self.temp_patch_size = temp_patch_size
        self.epoch_size = epoch_size
        self.gray_mode = gray_mode

        # only keep directory entries (same assumption as original code)
        entries = sorted(os.listdir(self.clean_root))
        self.sequences = [e for e in entries if os.path.isdir(os.path.join(self.clean_root, e))]

        self.clean_cache = {}

        if preload:
            self._preload_all(progress=progress)

        self.sequences = [s for s in self.sequences if s in self.clean_cache]
        if len(self.sequences) == 0:
            raise RuntimeError("No valid sequences found after preload. Check clean directory and filenames.")

    def __len__(self):
        return int(self.epoch_size)

    def _preload_all(self, progress: bool = False):
        seq_iter = self.sequences
        if progress:
            try:
                from tqdm import tqdm
                seq_iter = tqdm(self.sequences, desc="Preloading sequences")
            except Exception:
                seq_iter = self.sequences

        skipped_no_files = []
        skipped_read_error = []
        loaded = []

        for seq in seq_iter:
            clean_dir = os.path.join(self.clean_root, seq)
            clean_files_all = sorted(glob.glob(os.path.join(clean_dir, "*.tif")))

            if len(clean_files_all) == 0:
                skipped_no_files.append(seq)
                print(f"[Preload] SKIP (no files): {seq}")
                continue

            clean_files_all.sort(key=_frame_numeric_key)

            try:
                clean_frames = [_read_thermal_image(p, gray_mode=self.gray_mode) for p in clean_files_all]
            except Exception as e:
                skipped_read_error.append((seq, str(e)))
                print(f"[Preload] SKIP (read error): {seq} -> {e}")
                continue

            try:
                clean_arr = np.stack(clean_frames, axis=0)
            except Exception as e:
                skipped_read_error.append((seq, str(e)))
                print(f"[Preload] SKIP (stack error): {seq} -> {e}")
                continue

            self.clean_cache[seq] = clean_arr
            loaded.append(seq)
            print(f"[Preload] LOADED: {seq}  frames={clean_arr.shape[0]} dtype={clean_arr.dtype}")

        print(f"[Preload] Summary: loaded={len(loaded)}, skipped_no_files={len(skipped_no_files)}, skipped_errors={len(skipped_read_error)}")
        if skipped_no_files:
            print("[Preload] sequences with no files:", skipped_no_files)
        if skipped_read_error:
            print("[Preload] sequences with read/stack errors:", skipped_read_error)

    def __getitem__(self, idx):
        if len(self.sequences) == 0:
            raise RuntimeError("No sequences available in CleanThermalDataset")

        seq = random.choice(self.sequences)
        clean_full = self.clean_cache.get(seq)
        if clean_full is None:
            raise ValueError(f"Sequence {seq} not loaded in cache")

        F_all = clean_full.shape[0]
        half = self.temp_patch_size // 2

        if F_all <= self.temp_patch_size:
            center = F_all // 2
        else:
            center = random.randint(half, F_all - 1 - half)

        indices = list(range(center - half, center + half + 1))
        indices = [min(max(i, 0), F_all - 1) for i in indices]

        clean_seq = clean_full[indices]

        H, W = clean_seq.shape[1], clean_seq.shape[2]
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"Input patch ({H}x{W}) is smaller than requested crop size {self.patch_size}")

        x = random.randint(0, W - self.patch_size)
        y = random.randint(0, H - self.patch_size)

        clean_crop = clean_seq[:, y : y + self.patch_size, x : x + self.patch_size]
        if clean_crop.ndim == 3:
            clean_crop = clean_crop[:, None, :, :]
        elif clean_crop.ndim == 4:
            clean_crop = clean_crop.transpose(0, 3, 1, 2)

        clean_t = torch.from_numpy(clean_crop.astype(np.float32))

        return clean_t