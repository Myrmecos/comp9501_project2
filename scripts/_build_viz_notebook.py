#!/usr/bin/env python3
"""Generate visualize_dataset.ipynb (one-shot helper)."""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

nb = new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

cells.append(
    new_markdown_cell(
        """# TAPOR Dataset Visualization

Depth-first exploration of the three TAPOR Drive releases for the COMP9501 proposal:
**Physics-Guided Hand Depth Estimation from Low-Resolution Thermal Arrays**.

| Archive | Path | Role |
|---------|------|------|
| Training Dataset | `data/tapor/Trainset/` | Pose / depth train |
| Test Dataset | `data/tapor/Testset/` | Pose / depth eval |
| Gesture Recognition | `data/tapor/gesture/` | Downstream gesture eval |

Run with `PYTHONPATH=.python_pkgs:$PYTHONPATH` if using the local package tree.
"""
    )
)

cells.append(
    new_code_cell(
        """import json
import pickle
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path('data/tapor')
ASSETS = Path('data_overview/assets')
ASSETS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0f1115',
    'axes.facecolor': '#171a21',
    'axes.edgecolor': '#3a3f4b',
    'axes.labelcolor': '#e8e6e3',
    'text.color': '#e8e6e3',
    'xtick.color': '#b7b3ad',
    'ytick.color': '#b7b3ad',
    'grid.color': '#2a2f3a',
    'savefig.facecolor': '#0f1115',
    'font.size': 11,
})

THERMAL_CMAP = 'inferno'
DEPTH_CMAP = 'viridis'

REQUIRED = [
    'thermal_array_temperature_maps',
    'thermal_array_ambient_temperatures',
    'label_depth_maps',
    'label_2d_joints',
    'label_3d_joints',
    'label_hand_depth',
]

HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def list_pkls(directory):
    return sorted(p for p in Path(directory).glob('*.pkl') if not p.name.startswith('._'))

archives = {
    'Trainset': ROOT / 'Trainset',
    'Testset': ROOT / 'Testset',
    'gesture': ROOT / 'gesture',
}
for name, path in archives.items():
    print(f'{name}: {len(list_pkls(path))} pickles @ {path}')
"""
    )
)

cells.append(new_markdown_cell("## 1. Inventory"))

cells.append(
    new_code_cell(
        """summary_path = ROOT / 'inspection_summary.json'
inspection = json.loads(summary_path.read_text()) if summary_path.exists() else []

rows = []
for name, path in archives.items():
    files = list_pkls(path)
    sample = load_pkl(files[0])
    rows.append({
        'archive': name,
        'n_files': len(files),
        'keys': sorted(sample.keys()),
        'thermal0': list(np.asarray(sample['thermal_array_temperature_maps'][0]).shape),
        'depth0': list(np.asarray(sample['label_depth_maps'][0]).shape),
        'frames0': len(sample['thermal_array_temperature_maps']),
    })

print('=== Archive inventory ===')
for r in rows:
    print(f"{r['archive']}: {r['n_files']} files | frames[0]={r['frames0']} | thermal={r['thermal0']} | depth={r['depth0']}")
    print('  keys:', r['keys'])

print('\\nPose vs gesture verdict:')
print('- Trainset / Testset: official pose protocol with depth + joints (primary for proposal depth/KP).')
print('- gesture/: same modality keys; filenames encode gesture0–gesture6 for GR_traintest.py.')
for block in inspection:
    print(f"\\n[{block['archive']}] frames {block.get('frame_count_summary')} | hand_depth {block.get('hand_depth_summary')}")
"""
    )
)

cells.append(new_markdown_cell("## 2. Thermal gallery"))

cells.append(
    new_code_cell(
        """def pick_samples(directory, k=6, seed=0):
    files = list_pkls(directory)
    rng = random.Random(seed)
    return [files[i] for i in rng.sample(range(len(files)), min(k, len(files)))]

def show_thermal_gallery(directory, title, out_name, k=6):
    files = pick_samples(directory, k=k)
    cols = 3
    rows = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis('off')
    for i, path in enumerate(files):
        data = load_pkl(path)
        th = data['thermal_array_temperature_maps']
        idx = len(th) // 2
        arr = np.asarray(th[idx], dtype=np.float64)
        amb = float(np.asarray(data['thermal_array_ambient_temperatures'][idx]))
        ax = axes[i // cols, i % cols]
        im = ax.imshow(arr, cmap=THERMAL_CMAP, vmin=max(arr.min(), amb - 5), vmax=max(arr.max(), amb + 1))
        ax.set_title(f'{path.stem}\\namb={amb:.1f}°C', fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, color='#f0a050', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(ASSETS / out_name, dpi=140, bbox_inches='tight')
    plt.show()

show_thermal_gallery(archives['Trainset'], 'Trainset — thermal maps (24×32)', 'thermal_gallery_train.png')
show_thermal_gallery(archives['gesture'], 'Gesture — thermal maps', 'thermal_gallery_gesture.png', k=6)
"""
    )
)

cells.append(new_markdown_cell("## 3. Depth-first panels"))

cells.append(
    new_code_cell(
        """def collect_hand_depths(directory, max_files=40, frames_per_file=40, seed=1):
    files = list_pkls(directory)
    rng = random.Random(seed)
    chosen = files if len(files) <= max_files else [files[i] for i in rng.sample(range(len(files)), max_files)]
    vals = []
    for path in chosen:
        data = load_pkl(path)
        hd = data['label_hand_depth']
        n = len(hd)
        idxs = np.linspace(0, n - 1, num=min(frames_per_file, n), dtype=int)
        for i in idxs:
            v = float(np.asarray(hd[i]).reshape(-1)[0])
            if np.isfinite(v) and v > 0:
                vals.append(v)
    return np.asarray(vals)

depth_train = collect_hand_depths(archives['Trainset'])
depth_test = collect_hand_depths(archives['Testset'], max_files=60)
depth_gest = collect_hand_depths(archives['gesture'], max_files=28)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, vals, title in zip(axes, [depth_train, depth_test, depth_gest], ['Trainset', 'Testset', 'gesture']):
    ax.hist(vals, bins=40, color='#f0a050', edgecolor='#0f1115', alpha=0.9)
    ax.axvline(0.4, color='#7ec8e3', ls='--', lw=1, label='40 cm')
    ax.set_title(title)
    ax.set_xlabel('label_hand_depth (m)')
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('count')
axes[0].legend(fontsize=8)
fig.suptitle('Hand depth distributions', color='#f0a050', fontsize=14)
fig.tight_layout()
fig.savefig(ASSETS / 'hand_depth_hist.png', dpi=140, bbox_inches='tight')
plt.show()

print('Train depth median/mean:', float(np.median(depth_train)), float(np.mean(depth_train)))
print('Fraction <=0.40 m train/test:', float((depth_train<=0.4).mean()), float((depth_test<=0.4).mean()))
"""
    )
)

cells.append(
    new_code_cell(
        """train_files = list_pkls(archives['Trainset'])
data = load_pkl(train_files[10])
idx = 120
thermal = np.asarray(data['thermal_array_temperature_maps'][idx], dtype=np.float64)
depth = np.asarray(data['label_depth_maps'][idx], dtype=np.float64)
depth_plot = depth.copy().astype(np.float64)
depth_plot[depth_plot <= 0] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
im0 = axes[0].imshow(thermal, cmap=THERMAL_CMAP)
axes[0].set_title(f"Thermal 24×32 | amb={float(data['thermal_array_ambient_temperatures'][idx]):.1f}°C")
fig.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(depth_plot, cmap=DEPTH_CMAP)
axes[1].set_title(f"Depth map 120×160 | hand_d={float(np.asarray(data['label_hand_depth'][idx])):.3f} m")
fig.colorbar(im1, ax=axes[1], fraction=0.046)
for ax in axes:
    ax.axis('off')
fig.suptitle(train_files[10].name, color='#f0a050')
fig.tight_layout()
fig.savefig(ASSETS / 'thermal_vs_depth.png', dpi=140, bbox_inches='tight')
plt.show()
"""
    )
)

cells.append(new_markdown_cell("## 4. Thermo–depth coupling"))

cells.append(
    new_code_cell(
        """def sample_thermo_depth_pairs(directory, max_files=25, frames_per=30, seed=2):
    files = list_pkls(directory)
    rng = random.Random(seed)
    chosen = files if len(files) <= max_files else [files[i] for i in rng.sample(range(len(files)), max_files)]
    xs, ys = [], []
    for path in chosen:
        data = load_pkl(path)
        th = data['thermal_array_temperature_maps']
        hd = data['label_hand_depth']
        n = len(th)
        idxs = np.linspace(20, n - 1, num=min(frames_per, max(1, n - 20)), dtype=int)
        for i in idxs:
            tmap = np.asarray(th[i], dtype=np.float64)
            warm = tmap[tmap > np.percentile(tmap, 85)]
            if warm.size == 0:
                continue
            d = float(np.asarray(hd[i]).reshape(-1)[0])
            if np.isfinite(d) and 0.05 < d < 1.2:
                xs.append(float(warm.mean()))
                ys.append(d)
    return np.asarray(xs), np.asarray(ys)

tx, ty = sample_thermo_depth_pairs(archives['Trainset'])
fig, ax = plt.subplots(figsize=(6.5, 4.5))
hb = ax.hexbin(tx, ty, gridsize=35, cmap='inferno', mincnt=1)
ax.set_xlabel('Mean warm-pixel temperature (°C)')
ax.set_ylabel('label_hand_depth (m)')
ax.set_title('Thermo–depth coupling (Trainset)')
fig.colorbar(hb, ax=ax, label='count')
fig.tight_layout()
fig.savefig(ASSETS / 'thermo_depth_hexbin.png', dpi=140, bbox_inches='tight')
plt.show()
print('Pearson r (approx):', float(np.corrcoef(tx, ty)[0, 1]) if len(tx) > 2 else None)
"""
    )
)

cells.append(new_markdown_cell("## 5. Joints (2D overlay + 3D skeleton)"))

cells.append(
    new_code_cell(
        """def find_labeled_frame(data):
    joints = data['label_3d_joints']
    joints2d = data['label_2d_joints']
    for i in range(len(joints)):
        j3 = joints[i]
        j2 = joints2d[i]
        if j3 is None or j2 is None:
            continue
        a3 = np.asarray(j3)
        a2 = np.asarray(j2)
        if a3.shape == (21, 3) and a2.shape == (21, 3) and np.isfinite(a3).all() and np.isfinite(a2).all():
            return i, a2, a3
    return None, None, None

labeled = None
for path in list_pkls(archives['Trainset']):
    data = load_pkl(path)
    idx, j2, j3 = find_labeled_frame(data)
    if idx is not None:
        labeled = (path, data, idx, j2, j3)
        break

assert labeled is not None, 'No valid 3D joints found in Trainset sample'
path, data, idx, j2, j3 = labeled
thermal = np.asarray(data['thermal_array_temperature_maps'][idx], dtype=np.float64)

fig = plt.figure(figsize=(12, 4.5))
ax0 = fig.add_subplot(1, 2, 1)
ax0.imshow(thermal, cmap=THERMAL_CMAP)
u = j2[:, 0] * (thermal.shape[1] - 1)
v = j2[:, 1] * (thermal.shape[0] - 1)
ax0.scatter(u, v, c='#7ec8e3', s=18, zorder=3)
for a, b in HAND_BONES:
    ax0.plot([u[a], u[b]], [v[a], v[b]], color='#f0a050', lw=1.2)
ax0.set_title(f'2D joints on thermal\\n{path.name} @ frame {idx}')
ax0.axis('off')

ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.scatter(j3[:, 0], j3[:, 1], j3[:, 2], c='#7ec8e3', s=20)
for a, b in HAND_BONES:
    ax1.plot([j3[a, 0], j3[b, 0]], [j3[a, 1], j3[b, 1]], [j3[a, 2], j3[b, 2]], color='#f0a050', lw=1.2)
ax1.set_title('3D skeleton (Leap)')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
fig.tight_layout()
fig.savefig(ASSETS / 'joints_overlay.png', dpi=140, bbox_inches='tight')
plt.show()
"""
    )
)

cells.append(
    new_code_cell(
        """def bone_lengths(j3):
    return np.array([np.linalg.norm(j3[a] - j3[b]) for a, b in HAND_BONES])

def joint_validity_stats(directory, max_files=40, seed=3):
    files = list_pkls(directory)
    rng = random.Random(seed)
    chosen = files if len(files) <= max_files else [files[i] for i in rng.sample(range(len(files)), max_files)]
    valid = empty = malformed = 0
    bones = []
    for path in chosen:
        data = load_pkl(path)
        for lab in data['label_3d_joints']:
            if lab is None:
                empty += 1
                continue
            arr = np.asarray(lab)
            if arr.size == 0:
                empty += 1
            elif arr.shape == (21, 3) and np.isfinite(arr).all():
                valid += 1
                bones.append(bone_lengths(arr))
            else:
                malformed += 1
    return valid, empty, malformed, np.asarray(bones) if bones else np.zeros((0, len(HAND_BONES)))

v, e, m, bones = joint_validity_stats(archives['Trainset'])
print(f'Trainset sampled joint frames: valid={v}, empty={e}, malformed={m}, valid_rate={v/(v+e+m):.3f}')

fig, ax = plt.subplots(figsize=(8, 3.8))
if bones.size:
    ax.boxplot(
        bones,
        showfliers=False,
        boxprops=dict(color='#f0a050'),
        medianprops=dict(color='#7ec8e3'),
        whiskerprops=dict(color='#b7b3ad'),
        capprops=dict(color='#b7b3ad'),
    )
ax.set_xlabel('bone index')
ax.set_ylabel('length')
ax.set_title('3D bone-length distributions (Trainset sample)')
fig.tight_layout()
fig.savefig(ASSETS / 'bone_lengths.png', dpi=140, bbox_inches='tight')
plt.show()
"""
    )
)

cells.append(new_markdown_cell("## 6. Gesture coverage"))

cells.append(
    new_code_cell(
        """gesture_files = list_pkls(archives['gesture'])
gcounts = Counter()
for p in gesture_files:
    for part in p.stem.split('_'):
        if part.startswith('gesture'):
            gcounts[part] += 1
print('Gesture file counts:', dict(sorted(gcounts.items())))

classes = sorted(gcounts)
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.ravel()
for ax in axes:
    ax.axis('off')
for i, g in enumerate(classes):
    path = next(p for p in gesture_files if g in p.stem)
    data = load_pkl(path)
    idx = 500
    arr = np.asarray(data['thermal_array_temperature_maps'][idx], dtype=np.float64)
    axes[i].imshow(arr, cmap=THERMAL_CMAP)
    axes[i].set_title(g)
    axes[i].axis('off')
fig.suptitle('Gesture exemplars (mid-sequence thermal)', color='#f0a050', fontsize=14)
fig.tight_layout()
fig.savefig(ASSETS / 'gesture_exemplars.png', dpi=140, bbox_inches='tight')
plt.show()
"""
    )
)

cells.append(new_markdown_cell("## 7. Spatial / temporal continuity"))

cells.append(
    new_code_cell(
        """occ = np.zeros((24, 32), dtype=np.float64)
files = pick_samples(archives['Trainset'], k=20, seed=5)
deltas = []
for path in files:
    data = load_pkl(path)
    th = data['thermal_array_temperature_maps']
    for i in range(20, min(len(th) - 1, 200)):
        a = np.asarray(th[i], dtype=np.float64)
        b = np.asarray(th[i + 1], dtype=np.float64)
        deltas.append(float(np.mean(np.abs(a - b))))
    for j2 in data['label_2d_joints'][20:200]:
        if j2 is None:
            continue
        arr = np.asarray(j2)
        if arr.shape != (21, 3) or not np.isfinite(arr).all():
            continue
        u = np.clip((arr[:, 0] * 31).astype(int), 0, 31)
        v = np.clip((arr[:, 1] * 23).astype(int), 0, 23)
        occ[v, u] += 1

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
im = axes[0].imshow(occ, cmap='inferno')
axes[0].set_title('2D joint occupancy (Train sample)')
axes[0].axis('off')
fig.colorbar(im, ax=axes[0], fraction=0.046)
axes[1].hist(deltas, bins=40, color='#7ec8e3', edgecolor='#0f1115')
axes[1].set_title('Adjacent-frame |ΔT| mean')
axes[1].set_xlabel('°C')
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / 'spatial_temporal.png', dpi=140, bbox_inches='tight')
plt.show()
print('mean |ΔT|:', float(np.mean(deltas)))
"""
    )
)

cells.append(new_markdown_cell("## 8. Proposal readiness summary"))

cells.append(
    new_code_cell(
        """print('''
PROPOSAL READINESS
==================
Primary task: physics-guided hand depth from 32×24 thermal.

Available supervision NOW (all three archives share keys):
  [YES] thermal 24×32 + ambient  -> forward recon target
  [YES] label_depth_maps 120×160 -> dense depth GT (needs resize/align to 24×32)
  [YES] label_hand_depth scalar  -> coarse depth / range check
  [YES] label_3d_joints (21,3)   -> downstream KP (some frames empty)
  [YES] gesture archive labels in filenames (gesture0–6) -> gesture head eval

Missing GT (as expected for TAP3D-style SSL):
  [NO]  emissivity / surface-temp / reflection maps
        -> must rely on differentiable forward thermal reconstruction loss

Recommended next steps:
  1. Align/resize depth maps to thermal grid (or project joints→depth).
  2. Implement inverse multi-primitive U-Net + forward recon on Trainset.
  3. Ablate vs end-to-end thermal→depth without physics recon.
  4. Attach lightweight KP + gesture heads; eval on Testset + gesture/.
''')

readiness = {
    'train_files': len(list_pkls(archives['Trainset'])),
    'test_files': len(list_pkls(archives['Testset'])),
    'gesture_files': len(list_pkls(archives['gesture'])),
    'thermal_shape': [24, 32],
    'depth_map_shape': [120, 160],
    'modalities': REQUIRED,
    'physics_gt_missing': ['emissivity', 'surface_temperature', 'reflection'],
    'assets': sorted(p.name for p in ASSETS.glob('*.png')),
}
(ASSETS / 'readiness.json').write_text(json.dumps(readiness, indent=2))
print('Wrote', ASSETS / 'readiness.json')
print('Assets:', readiness['assets'])
"""
    )
)

nb.cells = cells
out = Path("visualize_dataset.ipynb")
with out.open("w") as f:
    nbf.write(nb, f)
print(f"Wrote {out} cells={len(nb.cells)}")
