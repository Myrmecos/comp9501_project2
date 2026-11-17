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


def thermal_ok(th: np.ndarray, amb: float):
    th = fill_thermal_zeros(th.astype(np.float32))
    if not np.isfinite(th).all():
        return False, None, None
    if th.min() < -20 or th.max() > 60:
        return False, None, None
    contrast = float(th.max() - amb)
    if contrast < 5 or contrast > 40:
        return False, None, None
    warm = float((th > amb + 3).mean())
    if warm < 0.05 or warm > 0.85:
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


def draw_skeleton(ax, xy: np.ndarray, bone_color="#2ca02c", joint_color="#d62728") -> None:
    palm_idx = [0, 5, 9, 13, 17]
    palm = xy[palm_idx]
    ax.fill(palm[:, 0], palm[:, 1], color=bone_color, alpha=0.12, zorder=1)
    ax.plot(
        np.r_[palm[:, 0], palm[0, 0]],
        np.r_[palm[:, 1], palm[0, 1]],
        color=bone_color,
        lw=1.6,
        zorder=2,
    )
    for a, b in HAND_BONES:
        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            color=bone_color,
            lw=1.8,
            solid_capstyle="round",
            zorder=3,
        )
    ax.scatter(xy[:, 0], xy[:, 1], s=14, c=joint_color, zorder=4, linewidths=0)
    ax.set_aspect("equal")
    pad = 0.02
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f5f5f5")
    for spine in ax.spines.values():
        spine.set_visible(False)


def collect_candidates(data_root: Path) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    search_files = (
        sorted((data_root / "Trainset").glob("*.pkl"))[::2]
        + sorted((data_root / "Testset").glob("P1_U*.pkl"))[:80]
        + sorted((data_root / "gesture").glob("*.pkl"))
    )
    for path in search_files:
        with path.open("rb") as f:
            data = pickle.load(f)
        n = len(data["thermal_array_temperature_maps"])
        step = 6 if "gesture" in path.parts else 8
        # Prefer clean scenes over cover/interference pickles when scoring.
        # Skip cover / background-interference pickles for the presentation figure.
        if any(tok in path.name for tok in ("Cover", "BG", "body", "display", "LightCond")):
            continue
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
            # Fist: reward short tips; open: reward long tips.
            shape = float(tips.mean())
            if lab == "fist":
                # Strongly prefer the most curled fingers.
                shape_score = 40.0 * (0.13 - shape)
                if "gesture2" in path.name:
                    shape_score += 8.0
            elif lab == "open":
                shape_score = 8.0 * shape
            elif lab == "thumbs_up":
                # Prefer gesture1 files when available.
                shape_score = 6.0 * float(tips[0] - tips[1:].mean())
                if "gesture1" in path.name:
                    shape_score += 6.0
            elif lab == "ok" and "gesture5" in path.name:
                shape_score = 4.0 * float(tips.std()) + 6.0
            else:
                shape_score = 4.0 * float(tips.std())
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
                }
            )
    return buckets


def select_panels(buckets: dict[str, list[dict]]) -> list[tuple[str, dict]]:
    selected = []
    used_files: set[str] = set()
    for key, title in TARGET:
        opts = sorted(buckets.get(key, []), key=lambda x: -x["score"])
        if not opts:
            raise RuntimeError(f"No dataset frames found for class {key}")
        pick = None
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
            f"f={pick['frame']:4d} contrast={pick['contrast']:.1f}"
        )
    return selected


def plot(selected: list[tuple[str, dict]], out: Path) -> None:
    fig = plt.figure(figsize=(11.2, 6.2))
    outer = fig.add_gridspec(2, 4, wspace=0.18, hspace=0.28)
    for idx, (title, sample) in enumerate(selected):
        r, c = divmod(idx, 4)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.08)
        ax_t = fig.add_subplot(inner[0])
        ax_p = fig.add_subplot(inner[1])

        th = sample["thermal"]
        amb = sample["ambient"]
        vmin = max(float(np.percentile(th, 2)), amb - 2)
        vmax = max(float(np.percentile(th, 99)), amb + 1)
        ax_t.imshow(th, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for spine in ax_t.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#888")
        ax_t.set_title(title, fontsize=11, pad=4)

        draw_skeleton(ax_p, project_topdown(sample["joints"]))

    fig.suptitle(
        "Hand poses from the TAPOR dataset  ·  thermal → 3D joints (2×4)",
        fontsize=13,
        y=0.98,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def main() -> None:
    buckets = collect_candidates(ROOT / "data" / "tapor")
    print("bucket sizes:", {k: len(v) for k, v in sorted(buckets.items())})
    selected = select_panels(buckets)
    plot(selected, OUT)


if __name__ == "__main__":
    main()
