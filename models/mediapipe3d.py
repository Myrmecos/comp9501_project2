"""MediaPipe3d baseline (BlazeHandLandmark) adapted to thermal, as in TAPOR."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_MEDIAPIPE_DIR = Path(__file__).resolve().parents[1] / "third_party" / "TAPOR" / "models" / "mediapipe"


def _load_blaze_hand():
    """Load BlazeHandLandmark without colliding with our `models` package."""
    if "tapor_mediapipe_blazebase" not in sys.modules:
        spec_base = importlib.util.spec_from_file_location(
            "tapor_mediapipe_blazebase", _MEDIAPIPE_DIR / "blazebase.py"
        )
        mod_base = importlib.util.module_from_spec(spec_base)
        sys.modules["tapor_mediapipe_blazebase"] = mod_base
        assert spec_base.loader is not None
        spec_base.loader.exec_module(mod_base)

        # blazehand_landmark imports `.blazebase` — register package shim
        pkg = type(sys)("tapor_mp_pkg")
        pkg.__path__ = [str(_MEDIAPIPE_DIR)]
        sys.modules["tapor_mp_pkg"] = pkg
        sys.modules["tapor_mp_pkg.blazebase"] = mod_base

        spec = importlib.util.spec_from_file_location(
            "tapor_mp_pkg.blazehand_landmark",
            _MEDIAPIPE_DIR / "blazehand_landmark.py",
            submodule_search_locations=[str(_MEDIAPIPE_DIR)],
        )
        # Patch relative import by writing a thin loader
        source = (_MEDIAPIPE_DIR / "blazehand_landmark.py").read_text(encoding="utf-8")
        source = source.replace("from .blazebase import", "from tapor_mediapipe_blazebase import")
        source = source.replace("from torchinfo import summary", "summary = None  # optional")
        mod = type(sys)("tapor_mp_pkg.blazehand_landmark")
        sys.modules["tapor_mp_pkg.blazehand_landmark"] = mod
        exec(compile(source, str(_MEDIAPIPE_DIR / "blazehand_landmark.py"), "exec"), mod.__dict__)
    return sys.modules["tapor_mp_pkg.blazehand_landmark"].BlazeHandLandmark


class MediaPipe3D(nn.Module):
    """Official TAPOR MediaPipe baseline: upsample 32×24 → 256×256, 1-channel."""

    has_physics = False

    def __init__(self, resolution: int = 256, **_kwargs):
        super().__init__()
        BlazeHandLandmark = _load_blaze_hand()
        self.resolution = int(resolution)
        self.core = BlazeHandLandmark(resolution=self.resolution, in_channels=1)

    def forward(self, thermal, ambient=None):
        x = thermal
        if x.dim() == 4 and x.shape[1] != 1:
            x = x[:, -1:].contiguous()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        landmarks, _hand_flag, _handed = self.core(x)
        return {"joints": landmarks, "depth": None, "t_recon": None, "mask_logits": None}
