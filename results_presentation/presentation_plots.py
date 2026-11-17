# %% [markdown]
# # Presentation figures
#
# Official TAPOR test set (39,710 frames) for the bar charts, training logs for the curves, and every 20th test frame for PCK, root-aligned error, and distance bins.
#
# Depth-only (no forward model) and label-fraction runs are not in the logs, so those two ablations are not plotted.

# %%
import json
import os
import re
import sys
import warnings
from itertools import groupby
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)

def find_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "log" / "official_test_results.json").exists() and (p / "models").is_dir():
            return p
    raise FileNotFoundError("Could not find the repo root (log/official_test_results.json)")

ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

FIG = ROOT / "results+presentation" / "figures"
CACHE = ROOT / "results+presentation" / "cache"
FIG.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
CAPTURED = []

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Stable order: published baselines, then physics hybrids.
MODELS = [
    ("hand3d", "Hand3d", "hand3d_full"),
    ("mediapipe", "MediaPipe3d", "mediapipe_full"),
    ("nanotapor", "NanoTapor", "nanotapor_full"),
    ("vanilla", "TAPOR", "vanilla_full"),
    ("A", "Hybrid A", "hybridA_full"),
    ("B", "Hybrid B", "hybridB_full"),
    ("C", "Hybrid C", "hybridC_full"),
    ("D", "Hybrid D", "hybridD_full"),
]
COLORS = {
    "hand3d": "#8c6d4f",
    "mediapipe": "#4c78a8",
    "nanotapor": "#54a24b",
    "vanilla": "#f58518",
    "A": "#e45756",
    "B": "#72b7b2",
    "C": "#b279a2",
    "D": "#17becf",
}
HYBRIDS = ["A", "B", "C", "D"]
# TAPOR paper Table / Fig. 12c (full test).
PAPER_MPJPE = {"hand3d": 4.1137, "mediapipe": 3.3196, "nanotapor": 3.2683, "vanilla": 2.2636}
PAPER_PCK4 = {"hand3d": 0.5750, "mediapipe": 0.7048, "nanotapor": 0.7238, "vanilla": 0.9041}
PAPER_PARAMS = {"hand3d": 18.9612, "mediapipe": 2.0134, "nanotapor": 0.1565, "vanilla": 0.6910}
STRIDE = 20
CACHE_PATH = CACHE / "per_frame_stride20.npz"
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
LOG_RE = re.compile(
    r"epoch=(\d+)\s+train_loss=(\S+)\s+val_mpjpe_cm=(\S+)\s+"
    r"hand_depth_cm=(\S+)\s+depth_cm=(\S+)\s+recon=(\S+)"
)

def show(fig, name: str) -> None:
    path = FIG / name
    fig.savefig(path, bbox_inches="tight")
    CAPTURED.append((name, path.read_bytes()))
    print(f"saved {path.relative_to(ROOT)}")
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except Exception:
        pass
    plt.close(fig)

def load_official():
    rows = json.loads((ROOT / "log" / "official_test_results.json").read_text())
    by_key = {r["model"]: r for r in rows if r.get("status") == "ok"}
    return [by_key[k] for k, _, _ in MODELS]

def parse_log(exp: str):
    text = (ROOT / "log" / f"{exp}.log").read_text(errors="replace")
    rows = []
    for m in LOG_RE.finditer(text):
        ep, loss, mpjpe, hand, depth, recon = m.groups()
        rows.append({
            "epoch": int(ep),
            "train_loss": float(loss),
            "mpjpe_cm": float(mpjpe),
            "hand_depth_cm": float(hand),
            "depth_cm": float(depth),
            "recon": float(recon),
        })
    rows.sort(key=lambda r: r["epoch"])
    # Keep the last line if a resume rewrote an epoch.
    dedup = {}
    for r in rows:
        dedup[r["epoch"]] = r
    return [dedup[k] for k in sorted(dedup)]

def roll_mean(y, w=5):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return np.arange(1, len(y) + 1), y
    kernel = np.ones(w) / w
    smooth = np.convolve(y, kernel, mode="valid")
    return np.arange(w, len(y) + 1), smooth

print("root", ROOT)

# %% [markdown]
# ## 1. Official-test pose error
#
# Bars are the mean MPJPE over all 39,710 official test frames. Diamonds are the numbers reported in the TAPOR paper for the same four baselines.

# %%
official = load_official()
names = [label for _, label, _ in MODELS]
keys = [k for k, _, _ in MODELS]
mpjpe = np.array([r["test_mpjpe_cm"] for r in official])
colors = [COLORS[k] for k in keys]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")

# (a) our full-test MPJPE
ax = axes[0]
x = np.arange(len(names))
bars = ax.bar(x, mpjpe, color=colors, width=0.72, zorder=2)
for b, v in zip(bars, mpjpe):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
paper_x, paper_y = [], []
for i, k in enumerate(keys):
    if k in PAPER_MPJPE:
        paper_x.append(i)
        paper_y.append(PAPER_MPJPE[k])
ax.scatter(paper_x, paper_y, marker="D", s=28, c="black", zorder=3, label="TAPOR paper")
ax.set_xticks(x, names, rotation=28, ha="right")
ax.set_ylabel("MPJPE (cm)")
ax.set_ylim(0, max(mpjpe.max(), max(PAPER_MPJPE.values())) * 1.18)
ax.set_title("Official test MPJPE")
ax.legend(frameon=False, fontsize=9)
ax.set_axisbelow(True)

# (b) baselines vs the paper, grouped
ax = axes[1]
base_idx = [i for i, k in enumerate(keys) if k in PAPER_MPJPE]
bx = np.arange(len(base_idx))
w = 0.36
ours_v = [mpjpe[i] for i in base_idx]
paper_v = [PAPER_MPJPE[keys[i]] for i in base_idx]
b1 = ax.bar(bx - w / 2, ours_v, width=w, color=[colors[i] for i in base_idx], zorder=2, label="Ours")
b2 = ax.bar(bx + w / 2, paper_v, width=w, color="#d0d0d0", edgecolor="#666", linewidth=0.6, zorder=2, label="TAPOR paper")
for b, v in list(zip(b1, ours_v)) + list(zip(b2, paper_v)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(bx, [names[i] for i in base_idx])
ax.set_ylabel("MPJPE (cm)")
ax.set_ylim(0, max(ours_v + paper_v) * 1.22)
ax.set_title("Baselines vs paper")
ax.legend(frameon=False, fontsize=9)
ax.set_axisbelow(True)

show(fig, "01_test_mpjpe.png")

# %% [markdown]
# ## 2. Hand depth and thermal reconstruction
#
# Pose-only baselines do not emit a depth map. Reconstruction MAE is the masked absolute error between the input temperature and the differentiable forward model, in °C.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), layout="constrained")
h_keys = HYBRIDS
h_names = [label for k, label, _ in MODELS if k in h_keys]
h_rows = [r for r, (k, _, _) in zip(official, MODELS) if k in h_keys]
h_colors = [COLORS[k] for k in h_keys]
xx = np.arange(len(h_keys))
w = 0.36

ax = axes[0]
hand = np.array([r["test_hand_depth_mae_cm"] for r in h_rows])
depth = np.array([r["test_depth_mae_cm"] for r in h_rows])
b1 = ax.bar(xx - w / 2, hand, width=w, color=h_colors, zorder=2, label="Hand-depth MAE")
b2 = ax.bar(xx + w / 2, depth, width=w, color=h_colors, alpha=0.45, edgecolor=h_colors, zorder=2, label="Depth-map MAE")
for b, v in list(zip(b1, hand)) + list(zip(b2, depth)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xx, h_names)
ax.set_ylabel("MAE (cm)")
ax.set_ylim(0, max(hand.max(), depth.max()) * 1.22)
ax.set_title("Depth error on the official test set")
ax.legend(frameon=False, fontsize=9)
ax.set_axisbelow(True)

ax = axes[1]
recon = np.array([r["test_recon_mae"] for r in h_rows])
bars = ax.bar(xx, recon, color=h_colors, width=0.62, zorder=2)
for b, v in zip(bars, recon):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xx, h_names)
ax.set_ylabel("Masked recon MAE (°C)")
ax.set_ylim(0, recon.max() * 1.25)
ax.set_title("Forward-model reconstruction")
ax.set_axisbelow(True)

show(fig, "02_depth_recon.png")

# %% [markdown]
# ## 3. Error against model size
#
# Filled markers are our checkpoints. Hollow markers are the TAPOR paper's parameter counts and MPJPE for the four baselines.

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
for r, (k, label, _) in zip(official, MODELS):
    ax.scatter(r["params_m"], r["test_mpjpe_cm"], s=70, color=COLORS[k], zorder=3)
    ax.annotate(label, (r["params_m"], r["test_mpjpe_cm"]), textcoords="offset points", xytext=(6, 5), fontsize=8)
for k, (pcount, err) in {kk: (PAPER_PARAMS[kk], PAPER_MPJPE[kk]) for kk in PAPER_MPJPE}.items():
    ax.scatter(pcount, err, s=70, facecolors="none", edgecolors=COLORS[k], linewidths=1.6, zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Parameters (millions, log scale)")
ax.set_ylabel("MPJPE (cm)")
ax.set_title("Test error vs model size")
legend_elems = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#444", markeredgecolor="#444", markersize=8, label="Ours"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#444", markersize=8, label="TAPOR paper"),
]
ax.legend(handles=legend_elems, frameon=False)
ax.set_axisbelow(True)
show(fig, "03_params_vs_mpjpe.png")

# %% [markdown]
# ## 4. Per-frame errors
#
# PCK and distance bins need every joint, which the official summary does not store. This cell runs the eight checkpoints on every 20th official test frame and caches the errors. Later cells only read the cache.

# %%
import torch
from torch.utils.data import DataLoader

from main import forward_model, prepare_batch
from models import build_model
from utils import HAND_BONES as _BONES
from utils import TaporSmallDataset, collate_batch

assert _BONES == HAND_BONES

def build_loaded(key, exp, device):
    ckpt_path = ROOT / "weights" / exp / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    num_history = int(ckpt.get("num_history") or 0) if key == "vanilla" else 0
    model = build_model(
        key,
        batch_size=8,
        device=device,
        num_history=num_history,
        detach_physics_for_pose=True,
        spatial_ch=int(cfg.get("spatial_ch", 128)),
        kp_dim=int(cfg.get("kp_dim", 96)),
        inverse_base=int(cfg.get("inverse_base", 64)),
    ).float().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, num_history

def open_test_dataset(num_history: int) -> TaporSmallDataset:
    return TaporSmallDataset(
        data_root=ROOT / "data" / "tapor",
        split_json=ROOT / "data" / "tapor" / "official_test_split.json",
        split="test",
        num_history=num_history,
        max_frames_per_file=None,
        require_heatmap=False,
    )

def per_frame_errors(out, batch):
    joints = out["joints"]
    gt = batch["joints"]
    n = gt.shape[0]
    joints = joints[:n]
    jerr = torch.norm(joints - gt, dim=-1) * 100.0  # (B, 21) cm
    pred_r = joints - joints[:, :1]
    gt_r = gt - gt[:, :1]
    mpjrpe = torch.norm(pred_r - gt_r, dim=-1).mean(dim=-1) * 100.0
    hand = torch.full((n,), float("nan"), device=joints.device)
    depth = torch.full((n,), float("nan"), device=joints.device)
    if out.get("depth") is not None:
        pred_d = out["depth"][:n]
        mask = batch["mask"][:n]
        gt_d = batch["depth"][:n]
        denom = mask.sum(dim=(2, 3)).clamp(min=1.0)
        depth = ((pred_d - gt_d).abs() * mask).sum(dim=(2, 3)) / denom
        depth = depth.view(n) * 100.0
        pred_mean = (pred_d * mask).sum(dim=(2, 3)) / denom
        hand = (pred_mean.view(n) - batch["hand_depth"][:n].view(n)).abs() * 100.0
    return jerr.detach().cpu().numpy(), mpjrpe.detach().cpu().numpy(), hand.detach().cpu().numpy(), depth.detach().cpu().numpy()

def compute_cache(device):
    # History length only changes the TAPOR input. The frame index is shared.
    vanilla_hist = int(torch.load(ROOT / "weights" / "vanilla_full" / "best.pt", map_location="cpu", weights_only=False).get("num_history") or 0)
    ds_single = open_test_dataset(0)
    ds_hist = open_test_dataset(vanilla_hist)
    models = {}
    for key, _, exp in MODELS:
        print(f"load {exp}", flush=True)
        models[key], _ = build_loaded(key, exp, device)

    indices = list(range(0, len(ds_single), STRIDE))
    buckets = {
        k: {"joint": [], "mpjrpe": [], "hand": [], "depth": []}
        for k, _, _ in MODELS
    }
    kept_index, kept_frame, kept_hd, kept_rel = [], [], [], []

    def path_key(i):
        return ds_single.samples[i][0]

    n_files = 0
    for path, group in groupby(indices, key=path_key):
        idxs = list(group)
        n_files += 1
        for start in range(0, len(idxs), 8):
            chunk = idxs[start:start + 8]
            batch_s = prepare_batch(collate_batch([ds_single[i] for i in chunk]), device)
            batch_h = prepare_batch(collate_batch([ds_hist[i] for i in chunk]), device)
            with torch.no_grad():
                for key, _, _ in MODELS:
                    batch = batch_h if key == "vanilla" else batch_s
                    out = forward_model(models[key], batch)
                    jerr, mpjrpe, hand, depth = per_frame_errors(out, batch)
                    buckets[key]["joint"].append(jerr)
                    buckets[key]["mpjrpe"].append(mpjrpe)
                    buckets[key]["hand"].append(hand)
                    buckets[key]["depth"].append(depth)
            hd = batch_s["hand_depth"].detach().cpu().numpy().reshape(-1) * 100.0
            kept_hd.extend(hd.tolist())
            for i in chunk:
                p, frame = ds_single.samples[i]
                kept_index.append(i)
                kept_frame.append(frame)
                kept_rel.append(str(p.relative_to(ROOT / "data" / "tapor")))
        ds_single._cache.clear()
        ds_hist._cache.clear()
        if n_files % 25 == 0:
            print(f"  files {n_files}", flush=True)

    payload = {
        "stride": np.array([STRIDE]),
        "index": np.asarray(kept_index, dtype=np.int32),
        "frame": np.asarray(kept_frame, dtype=np.int32),
        "hand_depth_cm": np.asarray(kept_hd, dtype=np.float32),
        "rel": np.asarray(kept_rel),
    }
    for key, _, _ in MODELS:
        payload[f"joint_{key}"] = np.concatenate(buckets[key]["joint"]).astype(np.float32)
        payload[f"mpjrpe_{key}"] = np.concatenate(buckets[key]["mpjrpe"]).astype(np.float32)
        payload[f"hand_{key}"] = np.concatenate(buckets[key]["hand"]).astype(np.float32)
        payload[f"depth_{key}"] = np.concatenate(buckets[key]["depth"]).astype(np.float32)
    np.savez_compressed(CACHE_PATH, **payload)
    print(f"cached {len(kept_index)} frames -> {CACHE_PATH.relative_to(ROOT)}", flush=True)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
rebuild = not CACHE_PATH.exists()
if CACHE_PATH.exists():
    probe = np.load(CACHE_PATH, allow_pickle=True)
    if int(probe["stride"][0]) != STRIDE or "joint_A" not in probe.files:
        rebuild = True
    probe.close()
if rebuild:
    compute_cache(device)
else:
    print(f"using cache {CACHE_PATH.relative_to(ROOT)}")

per_frame = np.load(CACHE_PATH, allow_pickle=True)
print(f"per-frame N={len(per_frame['index'])}")
print("hand depth cm  percentiles", np.percentile(per_frame["hand_depth_cm"], [0, 5, 25, 50, 75, 95, 100]).round(1))
print(f"{'model':<14} {'sub MPJPE':>10} {'full MPJPE':>10} {'MPJRPE':>8} {'PCK@4':>8}")
for (key, label, _), row in zip(MODELS, official):
    j = per_frame[f"joint_{key}"]
    sub = float(j.mean())
    pck4 = float((j <= 4.0).mean())
    rel = float(per_frame[f"mpjrpe_{key}"].mean())
    print(f"{label:<14} {sub:10.3f} {row['test_mpjpe_cm']:10.3f} {rel:8.3f} {pck4:8.3f}")

# %% [markdown]
# ## 5. PCK curve
#
# PCK is the fraction of the 21 joints within a threshold, on the stride-20 subset. The curve is the same definition as TAPOR Fig. 12b. Open circles at 4 cm are the paper's PCK@4 cm.

# %%
thresholds = np.linspace(0, 10, 201)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), layout="constrained")

ax = axes[0]
pck4 = {}
for key, label, _ in MODELS:
    err = per_frame[f"joint_{key}"].reshape(-1)
    curve = np.array([(err <= t).mean() for t in thresholds])
    pck4[key] = float((err <= 4.0).mean())
    ax.plot(thresholds, curve, color=COLORS[key], lw=1.8, label=f"{label}  {pck4[key]:.2f}")
for key, y in PAPER_PCK4.items():
    ax.scatter([4.0], [y], s=36, facecolors="none", edgecolors=COLORS[key], linewidths=1.5, zorder=4)
ax.axvline(4.0, color="#888", lw=0.8, ls=":")
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Joint error threshold (cm)")
ax.set_ylabel("PCK")
ax.set_title(f"PCK curve  (every {STRIDE}th test frame, N={len(per_frame['index'])})")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_axisbelow(True)

ax = axes[1]
x = np.arange(len(MODELS))
vals = [pck4[k] for k, _, _ in MODELS]
bars = ax.bar(x, vals, color=[COLORS[k] for k, _, _ in MODELS], width=0.72, zorder=2)
for i, (key, _, _) in enumerate(MODELS):
    if key in PAPER_PCK4:
        ax.scatter([i], [PAPER_PCK4[key]], marker="D", s=28, c="black", zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.scatter([], [], marker="D", s=28, c="black", label="TAPOR paper PCK@4 cm")
ax.set_xticks(x, [label for _, label, _ in MODELS], rotation=28, ha="right")
ax.set_ylabel("PCK@4 cm")
ax.set_ylim(0, 1.08)
ax.set_title("PCK at 4 cm")
ax.legend(frameon=False, fontsize=9)
ax.set_axisbelow(True)
show(fig, "04_pck.png")

# %% [markdown]
# ## 6. Error against hand distance
#
# Frames are binned by the ground-truth scalar hand depth. The left panel is pose error (TAPOR Fig. 16). The right panel is the physics models' hand-depth MAE.

# %%
hd = per_frame["hand_depth_cm"]
edges = [0, 15, 25, 35, 45, 80]
bin_masks, bin_labels = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (hd >= lo) & (hd < hi)
    if int(m.sum()) < 30:
        continue
    bin_masks.append(m)
    bin_labels.append(f"{lo}–{hi}\n(n={int(m.sum())})")

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
xs = np.arange(len(bin_labels))

ax = axes[0]
for key, label, _ in MODELS:
    means = []
    for m in bin_masks:
        means.append(float(per_frame[f"joint_{key}"][m].mean()))
    ax.plot(xs, means, marker="o", ms=4.5, lw=1.6, color=COLORS[key], label=label)
ax.set_xticks(xs, bin_labels)
ax.set_xlabel("Ground-truth hand depth (cm)")
ax.set_ylabel("MPJPE (cm)")
ax.set_title("Pose error vs distance")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_axisbelow(True)

ax = axes[1]
for key, label, _ in MODELS:
    if key not in HYBRIDS:
        continue
    means = [float(np.nanmean(per_frame[f"hand_{key}"][m])) for m in bin_masks]
    ax.plot(xs, means, marker="o", ms=4.5, lw=1.6, color=COLORS[key], label=label)
ax.set_xticks(xs, bin_labels)
ax.set_xlabel("Ground-truth hand depth (cm)")
ax.set_ylabel("Hand-depth MAE (cm)")
ax.set_title("Depth error vs distance")
ax.legend(frameon=False, fontsize=8)
ax.set_axisbelow(True)
show(fig, "05_error_vs_distance.png")

# %% [markdown]
# ## 7. Training curves
#
# Faint lines are the raw validation metric. Solid lines are a 5-epoch moving average. The dotted line on the depth panel is the switch from the physics-first phase to the pose phase (epoch 12).

# %%
histories = {exp: parse_log(exp) for _, _, exp in MODELS}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
ax = axes[0]
for key, label, exp in MODELS:
    rows = histories[exp]
    epochs = np.array([r["epoch"] for r in rows])
    y = np.array([r["mpjpe_cm"] for r in rows])
    ax.plot(epochs, y, color=COLORS[key], alpha=0.22, lw=1.0)
    sx, sy = roll_mean(y, 5)
    # roll_mean returns positions 5..N for a contiguous epoch-1 start; map back if epochs are 1..N
    if len(epochs) == len(y) and np.all(np.diff(epochs) == 1):
        sx = epochs[4:] if len(epochs) >= 5 else epochs
    ax.plot(sx, sy, color=COLORS[key], lw=1.8, label=label)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation MPJPE (cm)")
ax.set_ylim(0, 9)
ax.set_title("Validation pose error")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_axisbelow(True)

ax = axes[1]
for key, label, exp in MODELS:
    if key not in HYBRIDS:
        continue
    rows = histories[exp]
    epochs = np.array([r["epoch"] for r in rows])
    y = np.array([r["hand_depth_cm"] for r in rows])
    ax.plot(epochs, y, color=COLORS[key], alpha=0.22, lw=1.0)
    if len(epochs) >= 5 and np.all(np.diff(epochs) == 1):
        sx = epochs[4:]
        _, sy = roll_mean(y, 5)
    else:
        sx, sy = epochs, y
    ax.plot(sx, sy, color=COLORS[key], lw=1.8, label=label)
ax.axvline(12, color="#888", lw=0.9, ls=":")
ymax = ax.get_ylim()[1]
ax.text(13, 8.6, "phase 2", color="#666", fontsize=8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation hand-depth MAE (cm)")
ax.set_ylim(0, 10)
ax.set_title("Validation hand depth")
ax.legend(frameon=False, fontsize=8)
ax.set_axisbelow(True)
show(fig, "06_training_curves.png")

# %% [markdown]
# ## 8. Qualitative examples
#
# Two official-test frames for Hybrid A: the lowest pose error in the 15–45 cm band, and the largest hand-depth error. Depth maps are shown on the hand mask. Skeletons are root-aligned so the finger shape stays visible; the titles give the absolute MPJPE.

# %%
def masked_depth_cm(depth, mask):
    show = np.ma.masked_where(mask < 0.5, np.asarray(depth) * 100.0)
    return show

def draw_skeleton(ax, joints_m, color, label):
    j = (np.asarray(joints_m) - np.asarray(joints_m)[0]) * 100.0
    ax.scatter(j[:, 0], j[:, 1], j[:, 2], s=10, c=color, depthshade=False)
    first = True
    for a, b in HAND_BONES:
        ax.plot(
            [j[a, 0], j[b, 0]], [j[a, 1], j[b, 1]], [j[a, 2], j[b, 2]],
            color=color, lw=1.4, label=label if first else None,
        )
        first = False

mpjpe_A = per_frame["joint_A"].mean(axis=1)
hd_cm = per_frame["hand_depth_cm"]
hand_A = per_frame["hand_A"]
band = (hd_cm >= 15) & (hd_cm < 45) & np.isfinite(mpjpe_A)
best_i = int(np.flatnonzero(band)[np.argmin(mpjpe_A[band])])
finite_hand = np.flatnonzero(np.isfinite(hand_A))
worst_i = int(finite_hand[np.argmax(hand_A[finite_hand])])
if worst_i == best_i:
    order = finite_hand[np.argsort(hand_A[finite_hand])]
    worst_i = int(order[-2])
picks = [("Lowest pose error", best_i), ("Largest depth error", worst_i)]
print("qualitative local rows", picks, "dataset index", per_frame["index"][best_i], per_frame["index"][worst_i])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_A, _ = build_loaded("A", "hybridA_full", device)
model_T, hist = build_loaded("vanilla", "vanilla_full", device)
ds_single = open_test_dataset(0)
ds_hist = open_test_dataset(hist)

examples = []
for title, local in picks:
    ds_i = int(per_frame["index"][local])
    batch_s = prepare_batch(collate_batch([ds_single[ds_i]]), device)
    batch_h = prepare_batch(collate_batch([ds_hist[ds_i]]), device)
    with torch.no_grad():
        out_A = forward_model(model_A, batch_s)
        out_T = forward_model(model_T, batch_h)
    thermal = batch_s["thermal_abs"][0, 0].detach().cpu().numpy()
    ambient = float(batch_s["ambient"][0].detach().cpu())
    mask = batch_s["mask"][0, 0].detach().cpu().numpy()
    gt_depth = batch_s["depth"][0, 0].detach().cpu().numpy()
    pred_depth = out_A["depth"][0, 0].detach().cpu().numpy()
    recon = out_A["t_recon"][0, 0].detach().cpu().numpy()
    j2 = batch_s["joints_2d"][0].detach().cpu().numpy()
    examples.append({
        "title": title,
        "thermal": thermal,
        "ambient": ambient,
        "mask": mask,
        "gt_depth": gt_depth,
        "pred_depth": pred_depth,
        "recon": recon,
        "joints_2d": j2,
        "gt_joints": batch_s["joints"][0].detach().cpu().numpy(),
        "pred_A": out_A["joints"][0].detach().cpu().numpy(),
        "pred_T": out_T["joints"][0].detach().cpu().numpy(),
        "mpjpe_A": float(mpjpe_A[local]),
        "mpjpe_T": float(per_frame["joint_vanilla"][local].mean()),
        "hand_err": float(hand_A[local]),
        "hand_depth": float(hd_cm[local]),
        "rel": str(per_frame["rel"][local]),
        "frame": int(per_frame["frame"][local]),
    })
ds_single._cache.clear()
ds_hist._cache.clear()
del model_A, model_T

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(13.6, 6.2))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.15], wspace=0.28, hspace=0.32)
axes2d = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]
axes3d = [fig.add_subplot(gs[r, 4], projection="3d") for r in range(2)]
im_d = im_t = None
for row, ex in enumerate(examples):
    thermal = ex["thermal"]
    tmin = float(np.percentile(thermal, 1))
    tmax = float(np.percentile(thermal, 99.5))
    ax = axes2d[row][0]
    ax.imshow(thermal, cmap="inferno", vmin=tmin, vmax=tmax)
    j2 = ex["joints_2d"]
    if np.isfinite(j2).all() and np.abs(j2).sum() > 1e-3:
        u = j2[:, 0] * (thermal.shape[1] - 1)
        v = j2[:, 1] * (thermal.shape[0] - 1)
        for a, b in HAND_BONES:
            ax.plot([u[a], u[b]], [v[a], v[b]], color="white", lw=0.7)
        ax.scatter(u, v, s=8, c="white", zorder=3)
    ax.axis("off")
    ax.set_title("Input + GT joints" if row == 0 else "")
    ax.set_ylabel(
        f"{ex['title']}\n{ex['hand_depth']:.0f} cm away",
        fontsize=9,
    )

    pred_show = masked_depth_cm(ex["pred_depth"], ex["mask"])
    gt_show = masked_depth_cm(ex["gt_depth"], ex["mask"])
    stacked = np.concatenate([
        pred_show.compressed() if pred_show.count() else np.array([0.0]),
        gt_show.compressed() if gt_show.count() else np.array([0.0]),
    ])
    vmin, vmax = np.percentile(stacked, [2, 98])
    if vmin == vmax:
        vmax = vmin + 1.0
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f2f2f2")
    ax = axes2d[row][1]
    im_d = ax.imshow(pred_show, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title("Pred depth" if row == 0 else "")
    ax = axes2d[row][2]
    ax.imshow(gt_show, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title("GT depth" if row == 0 else "")

    ax = axes2d[row][3]
    im_t = ax.imshow(ex["recon"], cmap="inferno", vmin=tmin, vmax=tmax)
    ax.axis("off")
    ax.set_title("Reconstructed T" if row == 0 else "")

    ax = axes3d[row]
    draw_skeleton(ax, ex["gt_joints"], "#222222", "GT")
    draw_skeleton(ax, ex["pred_T"], COLORS["vanilla"], "TAPOR")
    draw_skeleton(ax, ex["pred_A"], COLORS["A"], "Hybrid A")
    ax.set_title(
        f"A {ex['mpjpe_A']:.1f} cm   TAPOR {ex['mpjpe_T']:.1f} cm\ndepth err {ex['hand_err']:.1f} cm",
        fontsize=9,
    )
    ax.set_xlabel("x (cm)", fontsize=7)
    ax.set_ylabel("y (cm)", fontsize=7)
    ax.set_zlabel("z (cm)", fontsize=7)
    ax.tick_params(labelsize=6)
    if row == 0:
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    print(
        f"{ex['title']}: {ex['rel']} frame {ex['frame']}  "
        f"MPJPE A={ex['mpjpe_A']:.2f} TAPOR={ex['mpjpe_T']:.2f} depth={ex['hand_err']:.2f}"
    )

fig.colorbar(im_d, ax=[axes2d[0][1], axes2d[0][2], axes2d[1][1], axes2d[1][2]], fraction=0.03, pad=0.02, label="Depth (cm)")
fig.colorbar(im_t, ax=[axes2d[0][3], axes2d[1][3]], fraction=0.05, pad=0.02, label="°C")
show(fig, "07_qualitative.png")
