"""
Clean-only validation dataset for thermal sequences.
"""

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import tifffile

class CleanValDataset(Dataset):
    def __init__(self, clean_root, max_num_fr=15, gray_mode=True):
        self.clean_root = clean_root
        self.max_num_fr = max_num_fr
        self.gray_mode = gray_mode
        self.sequences = [
            e for e in sorted(os.listdir(clean_root))
            if os.path.isdir(os.path.join(clean_root, e))
        ]

    def __len__(self):
        return len(self.sequences)

    def _read_frame(self, path):
        if path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise IOError(f"Could not read image: {path}")

        if img.ndim == 2:
            img = img[:, :, None]
        elif img.ndim == 3 and img.shape[2] == 1:
            pass
        elif img.ndim == 3 and img.shape[2] == 3:
            if self.gray_mode:
                img = img[..., 0:1]
            else:
                img = img.astype(np.float32)
        else:
            raise ValueError(f"Unsupported image shape {img.shape} for {path}")

        if not self.gray_mode and img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        return np.ascontiguousarray(img)

    def __getitem__(self, idx):
        seq_name = self.sequences[idx]
        clean_dir = os.path.join(self.clean_root, seq_name)
        clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.tif")))[:self.max_num_fr]

        if len(clean_files) == 0:
            raise ValueError(f"Sequence {seq_name} contains no TIFF files")

        clean_frames = [self._read_frame(p) for p in clean_files]
        clean_seq = np.stack(clean_frames, axis=0)
        clean_seq = clean_seq[:, None, :, :]
        clean_seq = torch.from_numpy(clean_seq.astype(np.float32)) / 65535.0

        return clean_seq, seq_name