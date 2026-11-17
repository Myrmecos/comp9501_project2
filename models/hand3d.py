"""Hand3d baseline (ColorHandPose3D / baseline3d) as used in TAPOR."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_CHP_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "TAPOR" / "models" / "colorhandpose3d"
_PKG = "chp3d"


def _load_color_hand_pose3d():
    if f"{_PKG}.model.ColorHandPose3D" in sys.modules:
        return sys.modules[f"{_PKG}.model.ColorHandPose3D"].ColorHandPose3D
    root = _CHP_ROOT
    sys.modules[_PKG] = types.ModuleType(_PKG)
    sys.modules[_PKG].__path__ = [str(root)]
    mod = types.ModuleType(f"{_PKG}.model")
    mod.__path__ = [str(root / "model")]
    sys.modules[f"{_PKG}.model"] = mod
    utils = types.ModuleType(f"{_PKG}.model.utils")
    utils.__path__ = [str(root / "model" / "utils")]
    sys.modules[f"{_PKG}.model.utils"] = utils
    importlib.import_module(f"{_PKG}.model.utils.general")
    importlib.import_module(f"{_PKG}.model.utils.transforms")
    importlib.import_module(f"{_PKG}.model.PoseNet")
    importlib.import_module(f"{_PKG}.model.PosePrior")
    return importlib.import_module(f"{_PKG}.model.ColorHandPose3D").ColorHandPose3D


class Hand3D(nn.Module):
    """TAPOR paper Hand3d / baseline3d: upsample → ColorHandPose3D."""

    has_physics = False
    needs_hand_side = True
    needs_heatmap = True

    def __init__(self, resolution: int = 256, **_kwargs):
        super().__init__()
        ColorHandPose3D = _load_color_hand_pose3d()
        self.resolution = int(resolution)
        self.core = ColorHandPose3D(in_channels=1, crop_size=self.resolution)

    def forward(self, thermal, ambient=None, hand_side=None):
        x = thermal
        if x.dim() == 4 and x.shape[1] != 1:
            x = x[:, -1:].contiguous()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if hand_side is None:
            hand_side = torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)
            hand_side[:, 1] = 1.0
        joints, heatmap = self.core(x, hand_side)
        return {
            "joints": joints,
            "heatmap": heatmap,
            "depth": None,
            "t_recon": None,
            "mask_logits": None,
        }
