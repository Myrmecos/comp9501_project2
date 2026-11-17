"""Shared utilities and TAPOR datasets (small + full split)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fill_thermal_zeros(arr: np.ndarray) -> np.ndarray:
    """Neighbor-average fill for MLX90640 subpage zeros (TAPOR-style)."""
    mat = arr.astype(np.float32).copy()
    zeros = mat <= 0
    if not zeros.any():
        return mat
    padded = np.pad(mat, 1, mode="edge")
    for _ in range(3):
        still = mat <= 0
        if not still.any():
            break
        top = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        stack = np.stack([top, down, left, right], axis=0)
        valid = stack > 0
        denom = valid.sum(axis=0).clip(min=1)
        filled = (stack * valid).sum(axis=0) / denom
        mat = np.where(still, filled, mat)
        padded = np.pad(mat, 1, mode="edge")
    return mat


def resize_depth_map(depth: np.ndarray, out_h: int = 24, out_w: int = 32) -> np.ndarray:
    """Resize RealSense depth (H,W) to thermal grid; convert mm-like ints to meters."""
    t = torch.from_numpy(depth.astype(np.float32))[None, None]
    if depth.dtype == np.uint16 or t.max() > 20:
        t = t / 1000.0
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def adaptive_hand_mask(
    thermal: np.ndarray,
    ambient: float,
    top_frac: float = 0.25,
    min_delta: float = 6.0,
    min_cover: float = 0.08,
    max_cover: float = 0.45,
) -> np.ndarray:
    """Warm-hand mask via ambient delta + adaptive percentile."""
    rel = thermal.astype(np.float32) - float(ambient)
    thr = max(float(np.percentile(rel, (1.0 - top_frac) * 100.0)), float(min_delta))
    mask = (rel >= thr).astype(np.float32)
    cover = float(mask.mean())
    if cover < min_cover:
        thr = float(np.percentile(rel, (1.0 - max(top_frac, 0.2)) * 100.0))
        mask = (rel >= thr).astype(np.float32)
        cover = float(mask.mean())
    if cover > max_cover:
        thr = float(np.percentile(rel, (1.0 - max_cover) * 100.0))
        mask = (rel >= max(thr, min_delta * 0.5)).astype(np.float32)
    if mask.sum() < 1:
        mask[rel >= rel.max() - 1e-3] = 1.0
    return mask.astype(np.float32)


def hand_focused_depth_map(
    depth: np.ndarray,
    mask: np.ndarray,
    hand_depth: float,
    max_dev: float = 0.12,
) -> np.ndarray:
    """Depth target on hand pixels only."""
    hd = float(hand_depth)
    out = np.zeros_like(depth, dtype=np.float32)
    on = mask > 0.5
    if not on.any():
        return out
    local = depth[on]
    use_rs = np.abs(local - hd) <= max_dev
    filled = np.where(use_rs, local, hd)
    out[on] = filled.astype(np.float32)
    return out


def make_gaussian_heatmaps(joints_2d: np.ndarray, heatmap_size=(256, 256), sigma: float = 5.0) -> np.ndarray:
    """joints_2d: (21,2) normalized [0,1] → (21,H,W) heatmaps."""
    h, w = heatmap_size[1], heatmap_size[0]  # TAPOR uses (W,H) naming; we store (H,W)
    # Use square size consistently as (H,W)
    hh, ww = int(heatmap_size[0]), int(heatmap_size[1]) if len(heatmap_size) > 1 else int(heatmap_size[0])
    # Official generate_heatmaps uses heatmap_size as (width, height) for linspace
    width, height = int(heatmap_size[0]), int(heatmap_size[1])
    heatmaps = np.zeros((21, height, width), dtype=np.float32)
    x_axis = np.linspace(0, width, width)
    y_axis = np.linspace(0, height, height)
    X, Y = np.meshgrid(x_axis, y_axis)
    for j in range(21):
        x, y = float(joints_2d[j, 0]), float(joints_2d[j, 1])
        if x < 0 or x > 1:
            x = 0.0
        if y < 0 or y > 1:
            y = 0.0
        x_scaled = x * width
        y_scaled = y * height
        gaussian = np.exp(-((X - x_scaled) ** 2 + (Y - y_scaled) ** 2) / (2 * sigma ** 2))
        m = float(gaussian.max())
        heatmaps[j] = (gaussian / m).astype(np.float32) if m > 0 else gaussian.astype(np.float32)
    return heatmaps


class TaporSmallDataset(Dataset):
    """Frame-level dataset from small_split.json (file list) or full_split.json ({rel,frame})."""

    def __init__(
        self,
        data_root: Path,
        split_json: Path,
        split: str = "train",
        height: int = 24,
        width: int = 32,
        skip_start_frames: Optional[int] = None,
        max_frames_per_file: Optional[int] = 80,
        num_history: int = 0,
        mask_top_frac: float = 0.25,
        mask_min_delta: float = 6.0,
        require_heatmap: bool = False,
        heatmap_size: int = 256,
    ):
        self.data_root = Path(data_root)
        payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
        self.skip = int(
            skip_start_frames if skip_start_frames is not None else payload.get("skip_start_frames", 20)
        )
        self.height = height
        self.width = width
        self.num_history = int(num_history)
        self.mask_top_frac = float(mask_top_frac)
        self.mask_min_delta = float(mask_min_delta)
        self.require_heatmap = bool(require_heatmap)
        self.heatmap_size = int(heatmap_size)
        self.samples: List[Tuple[Path, int]] = []
        self._cache: Dict[str, dict] = {}

        entries = payload[split]
        if not entries:
            raise ValueError(f"Empty split {split} in {split_json}")

        # full_split: list of {rel, frame}; small_split: list of relative pickle paths
        if isinstance(entries[0], dict):
            for item in entries:
                self.samples.append((self.data_root / item["rel"], int(item["frame"])))
        else:
            for rel in entries:
                path = self.data_root / rel
                with path.open("rb") as f:
                    data = pickle.load(f)
                n = len(data["thermal_array_temperature_maps"])
                valid_idxs = []
                for i in range(self.skip, n):
                    j = data["label_3d_joints"][i]
                    if j is None:
                        continue
                    arr = np.asarray(j)
                    if arr.shape == (21, 3) and np.isfinite(arr).all():
                        valid_idxs.append(i)
                if max_frames_per_file is not None and len(valid_idxs) > max_frames_per_file:
                    sel = np.linspace(0, len(valid_idxs) - 1, num=max_frames_per_file, dtype=int)
                    valid_idxs = [valid_idxs[i] for i in sel]
                for i in valid_idxs:
                    self.samples.append((path, i))

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> dict:
        key = str(path)
        if key not in self._cache:
            with path.open("rb") as f:
                self._cache[key] = pickle.load(f)
        return self._cache[key]

    def _thermal_at(self, data: dict, frame: int) -> np.ndarray:
        thermal = fill_thermal_zeros(
            np.asarray(data["thermal_array_temperature_maps"][frame], dtype=np.float32)
        )
        if thermal.shape != (self.height, self.width):
            t = torch.from_numpy(thermal)[None, None]
            t = F.interpolate(t, size=(self.height, self.width), mode="bilinear", align_corners=False)
            thermal = t[0, 0].numpy()
        return thermal.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        path, frame = self.samples[idx]
        data = self._load(path)
        thermal = self._thermal_at(data, frame)
        ambient = float(np.asarray(data["thermal_array_ambient_temperatures"][frame]).reshape(-1)[0])
        joints = np.asarray(data["label_3d_joints"][frame], dtype=np.float32)
        depth_raw = resize_depth_map(np.asarray(data["label_depth_maps"][frame]), self.height, self.width)
        hand_depth = float(np.asarray(data["label_hand_depth"][frame]).reshape(-1)[0])
        mask = adaptive_hand_mask(
            thermal, ambient, top_frac=self.mask_top_frac, min_delta=self.mask_min_delta
        )
        depth = hand_focused_depth_map(depth_raw, mask, hand_depth)

        # hand side from filename tokens: L → [1,0], else right [0,1]
        parts = path.name.replace(".pkl", "").split("_")
        hand_side = (
            np.array([1.0, 0.0], dtype=np.float32)
            if "L" in parts
            else np.array([0.0, 1.0], dtype=np.float32)
        )

        j2d_raw = data["label_2d_joints"][frame]
        if j2d_raw is None or len(j2d_raw) == 0:
            joints_2d = np.zeros((21, 2), dtype=np.float32)
        else:
            arr = np.asarray(j2d_raw, dtype=np.float32)
            joints_2d = arr[:, :2] if arr.ndim == 2 else np.zeros((21, 2), dtype=np.float32)

        if self.num_history > 0:
            frames = []
            for t in range(frame - self.num_history, frame + 1):
                tt = max(self.skip, min(t, len(data["thermal_array_temperature_maps"]) - 1))
                frames.append(self._thermal_at(data, tt))
            thermal_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(frames, axis=0)))
        else:
            thermal_tensor = torch.from_numpy(np.ascontiguousarray(thermal, dtype=np.float32)).unsqueeze(0)

        sample = {
            "thermal": thermal_tensor,
            "ambient": torch.tensor([ambient], dtype=torch.float32),
            "joints": torch.from_numpy(np.ascontiguousarray(joints, dtype=np.float32)),
            "joints_2d": torch.from_numpy(np.ascontiguousarray(joints_2d, dtype=np.float32)),
            "depth": torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32)).unsqueeze(0),
            "depth_raw": torch.from_numpy(np.ascontiguousarray(depth_raw, dtype=np.float32)).unsqueeze(0),
            "hand_depth": torch.tensor([hand_depth], dtype=torch.float32),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0),
            "hand_side": torch.from_numpy(hand_side),
        }
        if self.require_heatmap:
            hm = make_gaussian_heatmaps(
                joints_2d, heatmap_size=(self.heatmap_size, self.heatmap_size), sigma=5.0
            )
            sample["heatmap"] = torch.from_numpy(np.ascontiguousarray(hm))
        return sample


def collate_batch(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


GESTURE_LABEL_FROM_NAME = {
    "gesture0": 0,
    "gesture1": 1,
    "gesture2": 2,
    "gesture3": 3,
    "gesture4": 4,
    "gesture5": 5,
    "gesture6": 6,
}


def parse_gesture_label(path: Path) -> int:
    for part in path.stem.split("_"):
        if part in GESTURE_LABEL_FROM_NAME:
            return GESTURE_LABEL_FROM_NAME[part]
    raise ValueError(f"No gesture label in filename: {path.name}")


class GestureDataset(Dataset):
    """Frame-level TAPOR gesture dataset (label from filename gesture0–gesture6)."""

    def __init__(
        self,
        data_root: Path,
        split_json: Path,
        split: str = "train",
        height: int = 24,
        width: int = 32,
        num_history: int = 0,
        include_gt_joints: bool = True,
    ):
        self.data_root = Path(data_root)
        payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
        self.height = int(height)
        self.width = int(width)
        self.num_history = int(num_history)
        self.include_gt_joints = bool(include_gt_joints)
        self.skip = int(payload.get("skip_start_frames", 20))
        self.samples: List[Tuple[Path, int, int]] = []
        self._cache: Dict[str, dict] = {}
        for item in payload[split]:
            path = self.data_root / item["rel"]
            self.samples.append((path, int(item["frame"]), int(item["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> dict:
        key = str(path)
        if key not in self._cache:
            with path.open("rb") as f:
                self._cache[key] = pickle.load(f)
        return self._cache[key]

    def _thermal_at(self, data: dict, frame: int) -> np.ndarray:
        thermal = fill_thermal_zeros(
            np.asarray(data["thermal_array_temperature_maps"][frame], dtype=np.float32)
        )
        if thermal.shape != (self.height, self.width):
            t = torch.from_numpy(thermal)[None, None]
            t = F.interpolate(t, size=(self.height, self.width), mode="bilinear", align_corners=False)
            thermal = t[0, 0].numpy()
        return thermal.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        path, frame, label = self.samples[idx]
        data = self._load(path)
        n = len(data["thermal_array_temperature_maps"])
        ambient = float(np.asarray(data["thermal_array_ambient_temperatures"][frame]).reshape(-1)[0])
        parts = path.name.replace(".pkl", "").split("_")
        hand_side = (
            np.array([1.0, 0.0], dtype=np.float32)
            if "L" in parts
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        if self.num_history > 0:
            frames = []
            for t in range(frame - self.num_history, frame + 1):
                tt = max(self.skip, min(t, n - 1))
                frames.append(self._thermal_at(data, tt))
            thermal_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(frames, axis=0)))
        else:
            thermal = self._thermal_at(data, frame)
            thermal_tensor = torch.from_numpy(np.ascontiguousarray(thermal)).unsqueeze(0)

        sample = {
            "thermal": thermal_tensor,
            "ambient": torch.tensor([ambient], dtype=torch.float32),
            "hand_side": torch.from_numpy(hand_side),
            "label": torch.tensor(label, dtype=torch.long),
        }
        if self.include_gt_joints:
            j = data["label_3d_joints"][frame]
            if j is None:
                joints = np.zeros((21, 3), dtype=np.float32)
            else:
                joints = np.asarray(j, dtype=np.float32)
                if joints.shape != (21, 3) or not np.isfinite(joints).all():
                    joints = np.zeros((21, 3), dtype=np.float32)
            sample["joints"] = torch.from_numpy(np.ascontiguousarray(joints))
        return sample


def collate_gesture(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
