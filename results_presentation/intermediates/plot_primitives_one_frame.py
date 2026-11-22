#!/usr/bin/env python3
"""Plot thermal / RGB / depth / emissivity / surface temp / reflectance for one frame.

Depth, emissivity, surface temperature, and reflectance come from Hybrid A
intermediate outputs. Released TAPOR pickles have no RealSense RGB, so the RGB
panel is a false-color rendering of the thermal map.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import forward_model, prepare_batch
from models import build_model
from utils import HAND_BONES, adaptive_hand_mask, fill_thermal_zeros, resize_depth_map

OUT_DIR = Path(__file__).resolve().parent
PKL = ROOT / "data" / "tapor" / "Testset" / "P1_U2_R_LightCond3_X_1.pkl"
FRAME = 60
CKPT = ROOT / "weights" / "hybridA_full" / "best.pt"
MODEL_NAME = "A"
UPSAMPLE = 12


def upsample(arr: np.ndarray, scale: int = UPSAMPLE) -> np.ndarray:
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def masked(arr: np.ndarray, mask: np.ndarray, fill=np.nan) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    out[mask < 0.5] = fill
    return out


def thermal_false_color_rgb(th: np.ndarray, amb: float) -> np.ndarray:
    vmin = max(float(np.percentile(th, 5)), amb - 1)
    vmax = max(float(np.percentile(th, 99)), amb + 3)
    if vmax - vmin < 4:
        vmax = vmin + 4
    norm = np.clip((th - vmin) / (vmax - vmin), 0.0, 1.0)
    return cm.turbo(norm)[..., :3].astype(np.float32)


def load_frame(path: Path, frame: int):
    with path.open("rb") as f:
        data = pickle.load(f)
    th = fill_thermal_zeros(np.asarray(data["thermal_array_temperature_maps"][frame], dtype=np.float32))
    amb = float(np.asarray(data["thermal_array_ambient_temperatures"][frame]).reshape(-1)[0])
    depth_gt = resize_depth_map(np.asarray(data["label_depth_maps"][frame]), 24, 32)
    hand_depth = float(np.asarray(data["label_hand_depth"][frame]).reshape(-1)[0])
    joints = np.asarray(data["label_3d_joints"][frame], dtype=np.float32)
    j2 = data["label_2d_joints"][frame]
    joints_2d = (
        np.asarray(j2, dtype=np.float32)[:, :2]
        if j2 is not None and len(j2)
        else np.zeros((21, 2), np.float32)
    )
    mask = adaptive_hand_mask(th, amb)
    parts = path.name.replace(".pkl", "").split("_")
    hand_side = np.array([1.0, 0.0], np.float32) if "L" in parts else np.array([0.0, 1.0], np.float32)
    sample = {
        "thermal": torch.from_numpy(th[None].copy()),
        "ambient": torch.tensor([amb], dtype=torch.float32),
        "joints": torch.from_numpy(joints),
        "joints_2d": torch.from_numpy(joints_2d),
        "depth": torch.from_numpy(depth_gt[None].copy()),
        "hand_depth": torch.tensor([hand_depth], dtype=torch.float32),
        "mask": torch.from_numpy(mask[None].copy()),
        "hand_side": torch.from_numpy(hand_side),
    }
    meta = {
        "thermal": th,
        "ambient": amb,
        "depth_gt": depth_gt,
        "mask": mask,
        "joints_2d": joints_2d,
        "path": path,
        "frame": frame,
        "hand_depth": hand_depth,
    }
    return sample, meta


def run_model(sample: dict) -> dict:
    device = torch.device("cpu")
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = build_model(
        MODEL_NAME,
        detach_physics_for_pose=True,
        spatial_ch=int(cfg.get("spatial_ch", 128)),
        kp_dim=int(cfg.get("kp_dim", 96)),
        inverse_base=int(cfg.get("inverse_base", 64)),
    ).float().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    batch = prepare_batch({k: v.unsqueeze(0) for k, v in sample.items()}, device)
    with torch.no_grad():
        out = forward_model(model, batch)
    return {
        "depth": out["depth"][0, 0].cpu().numpy(),
        "emissivity": out["emissivity"][0, 0].cpu().numpy(),
        "surface_temp": out["surface_temp"][0, 0].cpu().numpy(),
        "reflection": out["reflection"][0, 0].cpu().numpy(),
        "mask_pred": torch.sigmoid(out["mask_logits"][0, 0]).cpu().numpy()
        if out.get("mask_logits") is not None
        else None,
    }


def overlay_2d(ax, joints_2d: np.ndarray, shape_hw) -> None:
    if not np.isfinite(joints_2d).all() or np.abs(joints_2d).sum() < 1e-3:
        return
    h, w = shape_hw[:2]
    u = joints_2d[:, 0] * (w - 1)
    v = joints_2d[:, 1] * (h - 1)
    for a, b in HAND_BONES:
        ax.plot([u[a], u[b]], [v[a], v[b]], color="white", lw=0.9, alpha=0.9)
    ax.scatter(u, v, s=7, c="white", zorder=3)


def add_colorbar(fig, ax, im, label: str = "") -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.04)
    cb = fig.colorbar(im, cax=cax)
    if label:
        cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def hand_limits(arr: np.ndarray, mask: np.ndarray, q=(5, 95)):
    vals = arr[mask > 0.5]
    if vals.size < 5:
        vals = arr.ravel()
    vmin, vmax = np.percentile(vals, q)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    return float(vmin), float(vmax)


def main() -> None:
    sample, meta = load_frame(PKL, FRAME)
    print(
        f"loaded {meta['path'].name} frame={meta['frame']} "
        f"amb={meta['ambient']:.2f} hand_depth={meta['hand_depth']:.3f} m"
    )
    prim = run_model(sample)

    # Prefer model mask when available; fall back to adaptive thermal mask.
    if prim["mask_pred"] is not None:
        show_mask = (0.7 * prim["mask_pred"] + 0.3 * meta["mask"]) > 0.45
    else:
        show_mask = meta["mask"] > 0.5
    show_mask = show_mask.astype(np.float32)

    th = meta["thermal"]
    amb = meta["ambient"]
    rgb = thermal_false_color_rgb(th, amb)

    depth_m = masked(prim["depth"], show_mask)
    depth_gt_m = masked(meta["depth_gt"], show_mask)
    eps_m = masked(prim["emissivity"], show_mask)
    tt_m = masked(prim["surface_temp"], show_mask)
    refl_m = masked(prim["reflection"], show_mask)
    mask_vis = show_mask.astype(np.float32)
    # Soft predicted mask if available (full field, not binary).
    mask_pred = prim["mask_pred"] if prim["mask_pred"] is not None else show_mask

    np.savez_compressed(
        OUT_DIR / "frame_primitives.npz",
        thermal=th,
        ambient=np.array([amb]),
        rgb_falsecolor=rgb,
        depth_pred=prim["depth"],
        depth_gt=meta["depth_gt"],
        emissivity=prim["emissivity"],
        surface_temp=prim["surface_temp"],
        reflection=prim["reflection"],
        mask=show_mask,
        mask_pred=mask_pred,
        source=np.array(str(meta["path"].relative_to(ROOT))),
        frame=np.array([FRAME]),
        model=np.array(f"{MODEL_NAME}:{CKPT.relative_to(ROOT)}"),
    )

    cmap_masked = cm.viridis.copy()
    cmap_masked.set_bad("#ececec")

    panels = [
        (
            "Thermal input (°C)",
            upsample(th),
            "inferno",
            (max(float(np.percentile(th, 2)), amb - 2), max(float(np.percentile(th, 99)), amb + 2)),
            "°C",
            True,
            False,
        ),
        (
            "False-color RGB\n(from thermal; RGB not in pickles)",
            upsample(rgb),
            None,
            None,
            None,
            True,
            True,
        ),
        (
            "Predicted depth (m)",
            upsample(depth_m),
            cmap_masked,
            hand_limits(prim["depth"], show_mask),
            "m",
            False,
            False,
        ),
        (
            "Emissivity ε",
            upsample(eps_m),
            cm.magma.copy(),
            hand_limits(prim["emissivity"], show_mask),
            None,
            False,
            False,
        ),
        (
            "Surface temperature Tt (°C)",
            upsample(tt_m),
            cm.plasma.copy(),
            hand_limits(prim["surface_temp"], show_mask),
            "°C",
            False,
            False,
        ),
        (
            "Reflectance R",
            upsample(refl_m),
            cm.cividis.copy(),
            hand_limits(prim["reflection"], show_mask),
            None,
            False,
            False,
        ),
    ]
    for _, _, cmap, _, _, _, _ in panels:
        if cmap is not None and hasattr(cmap, "set_bad"):
            cmap.set_bad("#ececec")

    def to_rgb_u8(img, cmap=None, lims=None, is_rgb=False, bad=(236, 236, 236)):
        """Rasterize to HxWx3 uint8 with no chrome."""
        if is_rgb:
            rgb = np.clip(img, 0, 1)
            return (rgb * 255).astype(np.uint8)
        vmin, vmax = lims
        data = np.ma.masked_invalid(img.astype(np.float32))
        norm = (data.filled(vmin) - vmin) / max(vmax - vmin, 1e-8)
        norm = np.clip(norm, 0, 1)
        cmap_obj = plt.colormaps[cmap] if isinstance(cmap, str) else cmap
        rgba = cmap_obj(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        if np.ma.is_masked(data):
            rgb[np.ma.getmaskarray(data)] = np.asarray(bad, dtype=np.uint8)
        return rgb

    # One PNG per modality: image pixels only (no title / colorbar / border).
    singles = [
        ("thermal.png", panels[0][1], panels[0][2], panels[0][3], False),
        ("rgb.png", panels[1][1], None, None, True),
        ("predicted_depth.png", panels[2][1], panels[2][2], panels[2][3], False),
        (
            "depth.png",
            upsample(depth_gt_m),
            cmap_masked,
            hand_limits(meta["depth_gt"], show_mask),
            False,
        ),
        ("emissivity.png", panels[3][1], panels[3][2], panels[3][3], False),
        ("surface_temperature.png", panels[4][1], panels[4][2], panels[4][3], False),
        ("reflectance.png", panels[5][1], panels[5][2], panels[5][3], False),
        (
            "reflection.png",
            panels[5][1],
            panels[5][2],
            panels[5][3],
            False,
        ),
        (
            "mask.png",
            upsample(mask_pred),
            "gray",
            (0.0, 1.0),
            False,
        ),
    ]
    from PIL import Image

    for fname, img, cmap, lims, is_rgb in singles:
        rgb = to_rgb_u8(img, cmap=cmap, lims=lims, is_rgb=is_rgb)
        out = OUT_DIR / fname
        Image.fromarray(rgb, mode="RGB").save(out)
        print("saved", out, rgb.shape[:2])


if __name__ == "__main__":
    main()
