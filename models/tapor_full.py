"""Full-size official TAPOR (base1) wrapped for our training loop."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

from .physics_utils import ambient_normalize

_TAPOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "TAPOR"
_PKG = "tapor_upstream"


def _ensure_upstream_package() -> None:
    if _PKG in sys.modules and hasattr(sys.modules[_PKG], "__path__"):
        return
    root = _TAPOR_ROOT
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(root)]
    sys.modules[_PKG] = pkg

    models_pkg = types.ModuleType(f"{_PKG}.models")
    models_pkg.__path__ = [str(root / "models")]
    sys.modules[f"{_PKG}.models"] = models_pkg

    tapor_pkg = types.ModuleType(f"{_PKG}.models.tapor")
    tapor_pkg.__path__ = [str(root / "models" / "tapor")]
    sys.modules[f"{_PKG}.models.tapor"] = tapor_pkg


def _load_official():
    _ensure_upstream_package()
    cfg_path = _TAPOR_ROOT / "tapor_model_config.py"
    spec = importlib.util.spec_from_file_location(f"{_PKG}.tapor_model_config", cfg_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{_PKG}.tapor_model_config"] = cfg_mod
    assert spec.loader is not None
    spec.loader.exec_module(cfg_mod)

    importlib.import_module(f"{_PKG}.models.tapor.MobileNetEncoder")
    importlib.import_module(f"{_PKG}.models.tapor.FusionBlocks")
    importlib.import_module(f"{_PKG}.models.tapor.transformer")
    tapor_model = importlib.import_module(f"{_PKG}.models.tapor.tapor_model")
    return tapor_model.Tapor, cfg_mod.base1_setting


class TaporFull(nn.Module):
    """Official TAPOR base1 with optional temporal history."""

    has_physics = False

    def __init__(
        self,
        batch_size: int = 8,
        device: str | torch.device | None = None,
        num_history: int = 4,
        **_kwargs,
    ):
        super().__init__()
        OfficialTapor, base1 = _load_official()
        cfg = copy.deepcopy(base1)
        # Official: num_history = fragment_size - 1; input length = num_history + 1
        self.num_history = int(num_history)
        cfg["temporal_keypoints_fusion_param"]["num_history"] = self.num_history
        self.batch_size = int(batch_size)
        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)
        self._init_device = device
        self.core = OfficialTapor(
            cfg["spatial_encoder_param"],
            cfg["keypoints_encoder_param"],
            cfg["cross_keypoints_fusion_param"],
            cfg["temporal_keypoints_fusion_param"],
            cfg["handpose_encoder_param"],
            input_width=32,
            input_height=24,
            batch_size=self.batch_size,
            train=True,
            device=device,
        )

    def _sync_pe_device(self, device: torch.device) -> None:
        he = self.core.handpose_encoder
        pe = getattr(he, "position_embedding", None)
        if isinstance(pe, torch.Tensor) and pe.device != device:
            he.position_embedding = pe.to(device)

    def forward(self, thermal: torch.Tensor, ambient=None):
        # Accept (B,1,H,W), (B,T,H,W), or (B,T,1,H,W)
        x = thermal
        if x.dim() == 5:
            x = x.squeeze(2)
        if x.dim() == 4 and x.shape[1] == 1 and self.num_history == 0:
            x = x  # B,1,H,W
        elif x.dim() == 4 and x.shape[1] == 1 and self.num_history > 0:
            # single frame given — repeat to fragment length
            x = x[:, 0]  # B,H,W
            x = x.unsqueeze(1).expand(-1, self.num_history + 1, -1, -1).contiguous()
        elif x.dim() == 3:
            x = x.unsqueeze(1)
            if self.num_history > 0:
                x = x.expand(-1, self.num_history + 1, -1, -1).contiguous()

        if ambient is not None:
            x = ambient_normalize(x, ambient)

        expect_t = self.num_history + 1
        if x.shape[1] != expect_t:
            if x.shape[1] == 1:
                x = x.expand(-1, expect_t, -1, -1).contiguous()
            elif x.shape[1] > expect_t:
                x = x[:, -expect_t:]
            else:
                pad = expect_t - x.shape[1]
                x = torch.cat([x[:, :1].expand(-1, pad, -1, -1), x], dim=1)

        b = x.shape[0]
        device = x.device
        self._sync_pe_device(device)

        if b != self.batch_size:
            if b > self.batch_size:
                x = x[: self.batch_size]
                b = self.batch_size
            else:
                pad_n = self.batch_size - b
                x = torch.cat([x, x[:1].expand(pad_n, -1, -1, -1)], dim=0)

        handpose, *_ = self.core(x)
        joints = handpose[:b]
        return {"joints": joints, "depth": None, "t_recon": None, "mask_logits": None}
