"""TAPOR-style gesture head: predicted 21×3 joints → 7 classes."""
from __future__ import annotations

import torch
import torch.nn as nn


GESTURE_NAMES = {
    0: "thumbs_up",
    1: "raised_fist",
    2: "crossed_fingers",
    3: "victory",
    4: "OK",
    5: "raised_hand",
    6: "no_hand",
}


class GestureClassifier(nn.Module):
    """Matches third_party/TAPOR/GR_traintest.py GestureClassifier (pose-only)."""

    def __init__(self, expanding_vector: int = 2, num_classes: int = 7, plug_thermal_map: bool = False, map_hw=(24, 32)):
        super().__init__()
        self.plug_thermal_map = bool(plug_thermal_map)
        self.num_classes = int(num_classes)
        h, w = int(map_hw[0]), int(map_hw[1])
        ev = int(expanding_vector)
        if self.plug_thermal_map:
            self.map2hidden = nn.Sequential(
                nn.Flatten(),
                nn.Linear(h * w, h * w * ev),
                nn.ReLU(inplace=True),
            )
            self.pose2hidden = nn.Sequential(
                nn.Flatten(),
                nn.Linear(21 * 3, 21 * 3 * ev),
                nn.ReLU(inplace=True),
            )
            self.hiddens2gesture = nn.Linear(h * w * ev + 21 * 3 * ev, num_classes)
        else:
            self.pose2gesture = nn.Sequential(
                nn.Flatten(),
                nn.Linear(21 * 3, 21 * 3 * ev),
                nn.ReLU(inplace=True),
                nn.Linear(21 * 3 * ev, num_classes),
            )

    def forward(self, pose: torch.Tensor, thermal_map: torch.Tensor | None = None) -> torch.Tensor:
        # pose: B,21,3
        if self.plug_thermal_map:
            assert thermal_map is not None
            t = thermal_map
            if t.dim() == 4:
                t = t[:, -1] if t.shape[1] != 1 else t[:, 0]
            th = self.map2hidden(t)
            ph = self.pose2hidden(pose)
            return self.hiddens2gesture(torch.cat([th, ph], dim=1))
        return self.pose2gesture(pose)
