#!/usr/bin/env python3
"""Plot 8 TAPOR-dataset hand poses in a 2x4 grid (thermal + top-down skeleton).

Style follows TAPOR Fig. 11, but every panel is drawn from our pickles with
matplotlib — no crops from the paper PDF/PNG.
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import HAND_BONES, fill_thermal_zeros

OUT = Path(__file__).resolve().parent / "figures" / "08_dataset_poses_2x4.png"
THERMAL_HW = (24, 32)  # (H, W)

TARGET = [
    ("point", "Point"),
    ("v_sign", "V-sign"),
    ("three", "Three"),
    ("four", "Four"),
    ("open", "Open hand"),
    ("thumbs_up", "Thumbs up"),
    ("ok", "OK"),
    ("fist", "Fist"),
]


def tip_lengths(j: np.ndarray) -> np.ndarray:
    return np.linalg.norm(j[[4, 8, 12, 16, 20]] - j[0], axis=1)


def classify(j: np.ndarray, thr: float = 0.105) -> str | None:
    tips = tip_lengths(j)
    ext = tips > thr
    thumb, idx, mid, ring, pinky = [bool(x) for x in ext]
    n = int(ext[1:].sum())
    tip_dist_ti = float(np.linalg.norm(j[4] - j[8]))
    if tip_dist_ti < 0.035 and mid and (ring or pinky):
        return "ok"
    if n == 0 and not thumb:
        return "fist"
    if n == 0 and thumb:
        return "thumbs_up"
    if n == 1 and idx:
        return "point"
    if n == 2 and idx and mid:
        return "v_sign"
    if n == 3 and idx and mid and ring:
        return "three"
    if n == 4 and not thumb:
        return "four"
    if n == 4 and thumb:
        return "open"
    if n >= 3 and thumb and float(tips.mean()) > 0.12:
        return "open"
    return None


def has_horizontal_streak(th: np.ndarray, amb: float) -> bool:
    """Reject frames with bright full-width horizontal bands."""
    peak = float(np.percentile(th, 99) - amb)
    if peak < 1e-3:
        return True
    thr = amb + 0.55 * peak
    warm = th > thr
    row_frac = warm.mean(axis=1)
    row_mean = (th - amb).mean(axis=1)
    for r in range(th.shape[0]):
        if row_frac[r] < 0.45:
            continue
        prev = row_mean[r - 1] if r > 0 else 0.0
        nxt = row_mean[r + 1] if r + 1 < len(row_mean) else 0.0
        if row_mean[r] > 1.35 * max(prev, nxt, 0.05):
            return True
        if row_frac[r] > 0.70:
            return True
    if float(row_frac.max()) > 0.60 and float(warm.mean(axis=0).max()) < 0.35:
        return True
    return False


def finger_peak_score(th: np.ndarray, amb: float) -> float:
    """How clearly separated finger tips look in the thermal map."""
    warm = th > (amb + 3.0)
    rows = np.where(warm.any(axis=1))[0]
    if len(rows) < 3:
        return 0.0
    top = rows[: max(2, len(rows) // 4)]
    profile = warm[top].mean(axis=0)
    # Count local maxima along the top band.
    peaks = 0
    for x in range(1, len(profile) - 1):
        if profile[x] > profile[x - 1] and profile[x] > profile[x + 1] and profile[x] > 0.15:
            peaks += 1
    # Prefer a connected hand mass that is not a single blob filling everything.
    cover = float(warm.mean())
    return float(peaks) - 2.0 * abs(cover - 0.22)


def thermal_ok(th: np.ndarray, amb: float):
    th = fill_thermal_zeros(th.astype(np.float32))
    if not np.isfinite(th).all():
        return False, None, None
    if th.min() < -20 or th.max() > 60:
        return False, None, None
    p99 = float(np.percentile(th, 99))
    if float(th.max() - p99) > 3.0:
        return False, None, None
    contrast = float(p99 - amb)
    if contrast < 5 or contrast > 25:
        return False, None, None
    warm = float((th > amb + 3).mean())
    if warm < 0.05 or warm > 0.55:
        return False, None, None
    if has_horizontal_streak(th, amb):
        return False, None, None
    return True, th, contrast


def project_topdown(joints: np.ndarray) -> np.ndarray:
    """Palm-facing 2D coords with fingers pointing +Y."""
    j = joints.astype(np.float64)
    wrist = j[0]
    n = np.cross(j[5] - wrist, j[17] - wrist)
    n_norm = np.linalg.norm(n)
    n = np.array([0.0, 0.0, 1.0]) if n_norm < 1e-8 else n / n_norm
    tips = j[[8, 12, 16]].mean(axis=0) - wrist
    up = tips - np.dot(tips, n) * n
    un = np.linalg.norm(up)
    if un < 1e-8:
        up = j[9] - wrist
        up = up - np.dot(up, n) * n
        un = np.linalg.norm(up)
    up = up / (un + 1e-12)
    right = np.cross(up, n)
    right = right / (np.linalg.norm(right) + 1e-12)
    if np.dot(right, j[5] - wrist) < 0:
        right = -right
    return np.stack([np.dot(j - wrist, right), np.dot(j - wrist, up)], axis=1)


def thermal_to_rgb(th: np.ndarray, amb: float, out_hw: tuple[int, int]) -> np.ndarray:
    """Upsample thermal colormap to a fixed pixel size."""
    from matplotlib import cm

    h, w = out_hw
    vmin = max(float(np.percentile(th, 5)), amb - 1)
    vmax = max(float(np.percentile(th, 99)), amb + 3)
    if vmax - vmin < 4:
        vmax = vmin + 4
    norm = np.clip((th - vmin) / (vmax - vmin), 0, 1)
    rgb = cm.viridis(norm)[..., :3]
    # Nearest-neighbor upsample to match skeleton canvas.
    ys = (np.linspace(0, th.shape[0] - 1, h)).astype(int)
    xs = (np.linspace(0, th.shape[1] - 1, w)).astype(int)
    return rgb[ys][:, xs]


def skeleton_to_rgb(
    xy: np.ndarray,
    out_hw: tuple[int, int],
    bone_color="#2ca02c",
    joint_color="#d62728",
) -> np.ndarray:
    """Rasterize skeleton to the same pixel width/height as the thermal panel."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    h, w = out_hw
    fig = Figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("#f3f3f3")
    palm_idx = [0, 5, 9, 13, 17]
    palm = xy[palm_idx]
    ax.fill(palm[:, 0], palm[:, 1], color=bone_color, alpha=0.12, zorder=1)
    ax.plot(
        np.r_[palm[:, 0], palm[0, 0]],
        np.r_[palm[:, 1], palm[0, 1]],
        color=bone_color,
        lw=1.8,
        zorder=2,
    )
    for a, b in HAND_BONES:
        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            color=bone_color,
            lw=2.0,
            solid_capstyle="round",
            zorder=3,
        )
    ax.scatter(xy[:, 0], xy[:, 1], s=18, c=joint_color, zorder=4, linewidths=0)
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    hw = max(xmax - xmin, 1e-6)
    hh = max(ymax - ymin, 1e-6)
    # Keep bone proportions; fill the same 32x24 panel as the thermal map.
    panel_aspect = out_hw[1] / out_hw[0]  # width / height
    span_x = hw * 1.12
    span_y = span_x / panel_aspect
    if span_y < hh * 1.12:
        span_y = hh * 1.12
        span_x = span_y * panel_aspect
    ax.set_xlim(cx - 0.5 * span_x, cx + 0.5 * span_x)
    ax.set_ylim(cy - 0.5 * span_y, cy + 0.5 * span_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    if rgb.shape[0] != h or rgb.shape[1] != w:
        ys = (np.linspace(0, rgb.shape[0] - 1, h)).astype(int)
        xs = (np.linspace(0, rgb.shape[1] - 1, w)).astype(int)
        rgb = rgb[ys][:, xs]
    return rgb


def collect_candidates(data_root: Path) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    search_files = (
        sorted((data_root / "Trainset").glob("*.pkl"))[::2]
        + sorted((data_root / "Testset").glob("P1_U*.pkl"))[:120]
        + sorted((data_root / "gesture").glob("*.pkl"))
    )
    for path in search_files:
        if any(tok in path.name for tok in ("Cover", "BG", "body", "display", "LightCond")):
            continue
        with path.open("rb") as f:
            data = pickle.load(f)
        n = len(data["thermal_array_temperature_maps"])
        step = 4 if "gesture" in path.parts else 6
        for i in range(20, n, step):
            jraw = data["label_3d_joints"][i]
            if jraw is None:
                continue
            j = np.asarray(jraw, dtype=np.float32)
            if j.shape != (21, 3) or not np.isfinite(j).all():
                continue
            amb = float(np.asarray(data["thermal_array_ambient_temperatures"][i]).reshape(-1)[0])
            ok, th, contrast = thermal_ok(
                np.asarray(data["thermal_array_temperature_maps"][i]), amb
            )
            if not ok:
                continue
            lab = classify(j)
            if lab is None:
                continue
            tips = tip_lengths(j)
            shape = float(tips.mean())
            peaks = finger_peak_score(th, amb)
            if lab == "fist":
                shape_score = 40.0 * (0.13 - shape)
            elif lab == "open":
                # Need several visible finger tips in the thermal map.
                shape_score = 8.0 * shape + 6.0 * peaks
            elif lab == "v_sign":
                shape_score = 8.0 * float(tips[1] + tips[2] - tips[3] - tips[4]) + 5.0 * peaks
            elif lab == "thumbs_up":
                shape_score = 6.0 * float(tips[0] - tips[1:].mean())
                if "gesture1" in path.name:
                    shape_score += 4.0
            elif lab == "ok" and "gesture5" in path.name:
                shape_score = 4.0 * float(tips.std()) + 4.0
            elif lab == "point":
                shape_score = 6.0 * float(tips[1] - tips[2:].mean()) + 2.0 * peaks
            else:
                shape_score = 4.0 * float(tips.std()) + 2.0 * peaks
            score = contrast + shape_score
            buckets[lab].append(
                {
                    "score": score,
                    "path": path,
                    "frame": i,
                    "joints": j,
                    "thermal": th,
                    "ambient": amb,
                    "contrast": contrast,
                    "peaks": peaks,
                }
            )
    return buckets


# Preferred frames with clear thermal finger structure (filename, frame).
PREFERRED = {
    "v_sign": [
        ("P1_U1_R_gesture3_0.pkl", 716),
        ("P1_U1_R_gesture3_0.pkl", 740),
        ("P1_U1_R_gesture3_0.pkl", 516),
        ("P1_U2_R_gesture3_0.pkl", 452),
    ],
    "open": [
        ("P1_U2_R_LightCond3_X_1.pkl", 60),
        ("P1_U2_R_LightCond3_X_1.pkl", 210),
        ("P1_U1_R_LightCond1_X_1.pkl", 230),
        ("P1_U2_L_gesture4_0.pkl", 120),
    ],
}


def load_frame(data_root: Path, filename: str, frame: int) -> dict | None:
    matches = list(data_root.rglob(filename))
    if not matches:
        return None
    path = matches[0]
    with path.open("rb") as f:
        data = pickle.load(f)
    if frame >= len(data["thermal_array_temperature_maps"]):
        return None
    jraw = data["label_3d_joints"][frame]
    if jraw is None:
        return None
    j = np.asarray(jraw, dtype=np.float32)
    if j.shape != (21, 3) or not np.isfinite(j).all():
        return None
    amb = float(np.asarray(data["thermal_array_ambient_temperatures"][frame]).reshape(-1)[0])
    th = fill_thermal_zeros(np.asarray(data["thermal_array_temperature_maps"][frame], dtype=np.float32))
    contrast = float(np.percentile(th, 99) - amb)
    return {
        "score": 0.0,
        "path": path,
        "frame": frame,
        "joints": j,
        "thermal": th,
        "ambient": amb,
        "contrast": contrast,
        "peaks": finger_peak_score(th, amb),
    }


def _find_pref(opts: list[dict], prefs: list[tuple[str, int]], data_root: Path) -> dict | None:
    by_key = {(o["path"].name, int(o["frame"])): o for o in opts}
    for name, frame in prefs:
        if (name, frame) in by_key:
            return by_key[(name, frame)]
        loaded = load_frame(data_root, name, frame)
        if loaded is not None:
            return loaded
    for name, frame in prefs:
        same = [o for o in opts if o["path"].name == name]
        if same:
            return min(same, key=lambda o: abs(int(o["frame"]) - frame))
    return None


def select_panels(buckets: dict[str, list[dict]], data_root: Path) -> list[tuple[str, dict]]:
    selected = []
    used_files: set[str] = set()
    for key, title in TARGET:
        opts = sorted(buckets.get(key, []), key=lambda x: -x["score"])
        if key == "open":
            opts = [o for o in opts if "gesture5" not in o["path"].name] or opts
        if not opts and key not in PREFERRED:
            raise RuntimeError(f"No dataset frames found for class {key}")
        pick = _find_pref(opts, PREFERRED.get(key, []), data_root) if key in PREFERRED else None
        if pick is None:
            for o in opts:
                if str(o["path"]) not in used_files:
                    pick = o
                    break
        if pick is None:
            pick = opts[0]
        used_files.add(str(pick["path"]))
        selected.append((title, pick))
        print(
            f"{title:12s} {pick['path'].name:40s} "
            f"f={pick['frame']:4d} contrast={pick['contrast']:.1f} peaks={pick['peaks']:.1f}"
        )
    return selected


def plot(selected: list[tuple[str, dict]], out: Path) -> None:
    # Compact 2x4. Thermal and skeleton are rasterized to the same pixel size, then
    # stacked, so their displayed widths are identical.
    scale = 12
    panel_hw = (THERMAL_HW[0] * scale, THERMAL_HW[1] * scale)
    gap = np.ones((max(4, scale // 3), panel_hw[1], 3), dtype=np.float32)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.85), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, wspace=0.015, hspace=0.03)
    for ax, (title, sample) in zip(axes.ravel(), selected):
        th_rgb = thermal_to_rgb(sample["thermal"], sample["ambient"], panel_hw)
        sk_rgb = skeleton_to_rgb(project_topdown(sample["joints"]), panel_hw)
        stack = np.concatenate([th_rgb, gap, sk_rgb], axis=0)
        ax.imshow(stack, interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=9.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888")

    fig.suptitle(
        "Hand poses from the TAPOR dataset  ·  thermal → 3D joints (2×4)",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    print("saved", out)


def main() -> None:
    data_root = ROOT / "data" / "tapor"
    buckets = collect_candidates(data_root)
    print("bucket sizes:", {k: len(v) for k, v in sorted(buckets.items())})
    selected = select_panels(buckets, data_root)
    plot(selected, OUT)


if __name__ == "__main__":
    main()
