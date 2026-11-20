#!/usr/bin/env python3
"""Train TAPOR-style gesture classifier on frozen pose backbones (or GT joints oracle)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model
from models.gesture_classifier import GESTURE_NAMES, GestureClassifier
from utils import GestureDataset, collate_gesture, load_yaml

POSE_EXPS = [
    ("vanilla", "vanilla_full"),
    ("hand3d", "hand3d_full"),
    ("mediapipe", "mediapipe_full"),
    ("nanotapor", "nanotapor_full"),
    ("A", "hybridA_full"),
    ("B", "hybridB_full"),
    ("C", "hybridC_full"),
    ("D", "hybridD_full"),
    ("A", "hybridA_nofwd_full"),
    ("B", "hybridB_nofwd_full"),
    ("C", "hybridC_nofwd_full"),
    ("D", "hybridD_nofwd_full"),
    ("modified_A", "modifiedA_full"),
    ("modified_A", "modifiedA_esc_full"),
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if k == "label":
            out[k] = v.to(device=device)
        else:
            out[k] = v.to(device=device, dtype=torch.float32)
    return out


def forward_pose(model, batch):
    kwargs = {}
    if getattr(model, "needs_hand_side", False):
        kwargs["hand_side"] = batch["hand_side"]
    if kwargs:
        return model(batch["thermal"], batch["ambient"], **kwargs)
    return model(batch["thermal"], batch["ambient"])


@torch.no_grad()
def predict_joints(pose_model, batch, use_gt: bool):
    if use_gt:
        return batch["joints"]
    out = forward_pose(pose_model, batch)
    return out["joints"]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def evaluate(pose_model, clf, loader, device, use_gt: bool, plug_thermal: bool):
    clf.eval()
    if pose_model is not None:
        pose_model.eval()
    ys, ps = [], []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, device)
            joints = predict_joints(pose_model, batch, use_gt=use_gt)
            thermal = batch["thermal"] if plug_thermal else None
            logits = clf(joints, thermal)
            pred = logits.argmax(dim=1)
            correct += int((pred == batch["label"]).sum().item())
            total += int(batch["label"].numel())
            ys.append(batch["label"].cpu().numpy())
            ps.append(pred.cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    acc = 100.0 * correct / max(total, 1)
    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1(y, p) * 100.0),
        "n": int(total),
        "y_true": y,
        "y_pred": p,
    }


def load_pose_backbone(model_name: str, ckpt_path: Path, device: torch.device, batch_size: int):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    num_history = int(ckpt.get("num_history", cfg.get("num_history", 0) if model_name == "vanilla" else 0))
    if model_name != "vanilla":
        num_history = 0
    build_kwargs = dict(
        batch_size=batch_size,
        device=device,
        num_history=num_history,
        detach_physics_for_pose=True,
        spatial_ch=int(cfg.get("spatial_ch", 128)),
        kp_dim=int(cfg.get("kp_dim", 96)),
        inverse_base=int(cfg.get("inverse_base", 64)),
        use_forward=bool(cfg.get("use_forward", True)),
    )
    if model_name == "modified_A":
        build_kwargs["keep_depth_attached"] = bool(cfg.get("keep_depth_attached", True))
        build_kwargs["soft_depth_attach"] = bool(cfg.get("soft_depth_attach", False))
        build_kwargs["use_mask_inv_skip"] = bool(cfg.get("use_mask_inv_skip", False))
    model = build_model(model_name, **build_kwargs).float().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, num_history, ckpt


def train_one(
    backbone: str,
    pose_exp: str | None,
    split_json: Path,
    cfg: dict,
    device: torch.device,
    out_root: Path,
    exp_suffix: str = "",
):
    use_gt = backbone == "gt_joints"
    suffix = exp_suffix or ""
    if use_gt:
        exp_id = f"gesture_{backbone}{suffix}"
    else:
        exp_id = f"gesture_{pose_exp}{suffix}"
    weight_dir = out_root / "weights" / exp_id
    log_dir = out_root / "log" / "tb" / exp_id
    # Preservation: never retrain over an existing gesture best.pt
    existing = weight_dir / "best.pt"
    summary_path = out_root / "log" / f"{exp_id}_summary.json"
    if existing.exists():
        print(f"[{exp_id}] SKIP existing {existing}", flush=True)
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        ck = torch.load(existing, map_location="cpu", weights_only=False)
        vm = ck.get("val_metrics") or {}
        return {
            "exp_id": exp_id,
            "backbone": backbone,
            "pose_exp": pose_exp,
            "best_epoch": ck.get("epoch"),
            "val_acc": vm.get("acc"),
            "val_macro_f1": vm.get("macro_f1"),
            "test_acc": None,
            "test_macro_f1": None,
            "status": "skipped_existing",
            "note": "best.pt already present; not overwritten",
        }
    weight_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    batch_size = int(cfg["batch_size"])
    # Heavy pose models: shrink batch to avoid OOM alongside other GPU users.
    if backbone in ("vanilla", "hand3d", "A", "C", "D", "modified_A"):
        batch_size = min(batch_size, 16)
    elif backbone in ("B", "mediapipe"):
        batch_size = min(batch_size, 24)
    plug_thermal = bool(cfg.get("plug_thermal_map", False))
    num_history = 0
    pose_model = None
    pose_ckpt_meta = {}

    if not use_gt:
        ckpt_path = Path("weights") / pose_exp / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing pose ckpt: {ckpt_path}")
        pose_model, num_history, pose_ckpt = load_pose_backbone(backbone, ckpt_path, device, batch_size)
        pose_ckpt_meta = {
            "pose_ckpt": str(ckpt_path),
            "pose_best_epoch": pose_ckpt.get("epoch"),
            "pose_val_mpjpe_cm": (pose_ckpt.get("val_metrics") or {}).get("mpjpe_cm"),
        }
        print(f"[{exp_id}] loaded pose {backbone} from {ckpt_path} (history={num_history})", flush=True)
    else:
        print(f"[{exp_id}] GT joints oracle (no pose model)", flush=True)

    ds_kwargs = dict(
        data_root=Path("data/tapor"),
        split_json=split_json,
        height=int(cfg.get("height", 24)),
        width=int(cfg.get("width", 32)),
        num_history=num_history,
        include_gt_joints=True,
    )
    train_ds = GestureDataset(split="train", **ds_kwargs)
    val_ds = GestureDataset(split="val", **ds_kwargs)
    test_ds = GestureDataset(split="test", **ds_kwargs)
    nw = int(cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, collate_fn=collate_gesture, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=nw, collate_fn=collate_gesture, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=nw, collate_fn=collate_gesture, drop_last=False
    )

    clf = GestureClassifier(
        expanding_vector=int(cfg.get("expanding_vector", 2)),
        num_classes=int(cfg.get("num_classes", 7)),
        plug_thermal_map=plug_thermal,
        map_hw=(int(cfg.get("height", 24)), int(cfg.get("width", 32))),
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_epoch = 0
    bad = 0
    patience = int(cfg.get("early_stop_patience", 12))
    t0 = time.time()
    history = []

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        clf.train()
        running = []
        for batch in train_loader:
            batch = prepare_batch(batch, device)
            with torch.no_grad():
                joints = predict_joints(pose_model, batch, use_gt=use_gt)
            thermal = batch["thermal"] if plug_thermal else None
            logits = clf(joints.detach(), thermal.detach() if thermal is not None else None)
            loss = criterion(logits, batch["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running.append(float(loss.item()))

        train_loss = float(np.mean(running)) if running else float("nan")
        val_m = evaluate(pose_model, clf, val_loader, device, use_gt=use_gt, plug_thermal=plug_thermal)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/acc", val_m["acc"], epoch)
        writer.add_scalar("val/macro_f1", val_m["macro_f1"], epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_m["acc"], "val_macro_f1": val_m["macro_f1"]})
        print(
            f"[{exp_id}] epoch={epoch} loss={train_loss:.4f} val_acc={val_m['acc']:.2f} "
            f"val_f1={val_m['macro_f1']:.2f}",
            flush=True,
        )

        ckpt = {
            "epoch": epoch,
            "exp_id": exp_id,
            "backbone": backbone,
            "pose_exp": pose_exp,
            "state_dict": clf.state_dict(),
            "optimizer": opt.state_dict(),
            "val_metrics": {"acc": val_m["acc"], "macro_f1": val_m["macro_f1"]},
            "cfg": cfg,
            "num_history": num_history,
            **pose_ckpt_meta,
        }
        torch.save(ckpt, weight_dir / "last.pt")
        if val_m["acc"] > best_val:
            best_val = val_m["acc"]
            best_epoch = epoch
            bad = 0
            torch.save(ckpt, weight_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[{exp_id}] early stop at epoch {epoch}", flush=True)
                break

    # test with best
    best_ck = torch.load(weight_dir / "best.pt", map_location=device, weights_only=False)
    clf.load_state_dict(best_ck["state_dict"])
    test_m = evaluate(pose_model, clf, test_loader, device, use_gt=use_gt, plug_thermal=plug_thermal)
    # confusion
    cm = np.zeros((7, 7), dtype=np.int64)
    for yt, yp in zip(test_m["y_true"], test_m["y_pred"]):
        cm[int(yt), int(yp)] += 1

    summary = {
        "exp_id": exp_id,
        "backbone": backbone,
        "pose_exp": pose_exp,
        "elapsed_sec": time.time() - t0,
        "best_epoch": best_epoch,
        "val_acc": best_val,
        "val_macro_f1": float(best_ck["val_metrics"]["macro_f1"]),
        "test_acc": test_m["acc"],
        "test_macro_f1": test_m["macro_f1"],
        "test_n": test_m["n"],
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "num_history": num_history,
        "class_names": GESTURE_NAMES,
        "confusion_matrix": cm.tolist(),
        **pose_ckpt_meta,
        "history_tail": history[-5:],
    }
    (weight_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "log" / f"{exp_id}_summary.json").write_text(json.dumps(summary, indent=2))
    # predictions for analysis
    np.savez_compressed(
        weight_dir / "test_preds.npz",
        y_true=test_m["y_true"],
        y_pred=test_m["y_pred"],
        confusion=cm,
    )
    writer.add_scalar("test/acc", test_m["acc"], best_epoch)
    writer.add_scalar("test/macro_f1", test_m["macro_f1"], best_epoch)
    writer.close()
    print(
        f"[{exp_id}] done test_acc={test_m['acc']:.2f} test_f1={test_m['macro_f1']:.2f} "
        f"best_epoch={best_epoch}",
        flush=True,
    )
    # free GPU
    del pose_model, clf
    torch.cuda.empty_cache()
    return summary


def write_results_md(summaries: list[dict], out_md: Path):
    lines = [
        "# Gesture recognition results",
        "",
        "TAPOR-style protocol: freeze pose backbone → train `GestureClassifier` on predicted 21×3 joints "
        "(7 classes: thumbs up … no-hand). Split: stratified 70/10/20 on `data/tapor/gesture/` "
        "(`data/tapor/gesture_split.json`, seed=0, skip first 20 frames).",
        "",
        "Paper reference (case study F1): Tapor **~94.9%**, NanoTapor **~87.2%**.",
        "",
        "| Backbone | Exp id | Test Acc (%) | Test macro-F1 (%) | Val Acc | Best ep | Test N | Status |",
        "|----------|--------|--------------|-------------------|---------|---------|--------|--------|",
    ]
    # Prefer v2 / longer-trained rows before legacy when sorting for readability
    def _sort_key(s):
        return (str(s.get("backbone") or ""), str(s.get("exp_id") or s.get("pose_exp") or ""))

    for s in sorted(summaries, key=_sort_key):
        st = s.get("status") or "ok"
        eid = s.get("exp_id") or (
            f"gesture_{s['backbone']}" if s.get("backbone") == "gt_joints" else f"gesture_{s.get('pose_exp')}"
        )
        if st not in ("ok", "skipped_existing") or s.get("test_acc") is None:
            lines.append(
                f"| {s.get('backbone')} | {eid} | - | - | "
                f"{s.get('val_acc') if s.get('val_acc') is not None else '-'} | "
                f"{s.get('best_epoch') or '-'} | - | {st} |"
            )
            continue
        lines.append(
            "| {bb} | {eid} | {acc:.2f} | {f1:.2f} | {va:.2f} | {ep} | {n} | {st} |".format(
                bb=s["backbone"],
                eid=eid,
                acc=float(s["test_acc"]),
                f1=float(s["test_macro_f1"]),
                va=float(s["val_acc"]),
                ep=s.get("best_epoch"),
                n=s.get("test_n", "-"),
                st=st,
            )
        )
    lines += [
        "",
        "Class order: `gesture0` thumbs_up, `gesture1` raised_fist, `gesture2` crossed_fingers, "
        "`gesture3` victory, `gesture4` OK, `gesture5` raised_hand, `gesture6` no_hand.",
        "",
        "Per-run summaries: `log/gesture_*_summary.json`; TB: `log/tb/gesture_*/`; weights: `weights/gesture_*/best.pt`.",
        "",
        f"Raw JSON: `log/gesture_results.json`",
    ]
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/gesture.yml"))
    ap.add_argument("--split-json", type=Path, default=Path("data/tapor/gesture_split.json"))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--backbones",
        nargs="*",
        default=None,
        help="Pose model names and/or 'gt_joints'. Default: all pose exps + gt_joints.",
    )
    ap.add_argument("--out-md", type=Path, default=Path("log/GESTURE_RESULTS.md"))
    ap.add_argument(
        "--exp-suffix",
        type=str,
        default="",
        help="Append to gesture exp id (e.g. _v2) so retrain does not clobber existing best.pt",
    )
    args = ap.parse_args()

    if not args.split_json.exists():
        raise SystemExit(f"Missing {args.split_json}; run scripts/make_gesture_split.py first")

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wanted = args.backbones
    jobs = []
    if wanted is None:
        # default: primary full exps only (first occurrence per model) + gt
        seen = set()
        for m, e in POSE_EXPS:
            if m in seen or "nofwd" in e:
                continue
            seen.add(m)
            jobs.append((m, e))
        jobs.append(("gt_joints", None))
    else:
        # Prefer exact pose-exp match; fall back to first model-name match
        exp_to_model = {e: m for m, e in POSE_EXPS}
        first_by_model: dict[str, str] = {}
        for m, e in POSE_EXPS:
            first_by_model.setdefault(m, e)
        for name in wanted:
            if name == "gt_joints":
                jobs.append(("gt_joints", None))
            elif name in exp_to_model:
                jobs.append((exp_to_model[name], name))
            elif name in first_by_model:
                jobs.append((name, first_by_model[name]))
            else:
                raise SystemExit(f"Unknown backbone {name}")

    summaries = []
    for backbone, pose_exp in jobs:
        try:
            s = train_one(
                backbone, pose_exp, args.split_json, cfg, device, ROOT, exp_suffix=args.exp_suffix
            )
            if "status" not in s:
                s["status"] = "ok"
            summaries.append(s)
        except Exception as e:
            print(f"[FAILED] {backbone}/{pose_exp}: {e}", flush=True)
            summaries.append({"backbone": backbone, "pose_exp": pose_exp, "status": f"error: {e}"})
            torch.cuda.empty_cache()

    # Merge any on-disk gesture summaries not in this batch (vanilla/gt etc.)
    by_id = {s.get("exp_id"): s for s in summaries if s.get("exp_id")}
    for p in sorted(Path("log").glob("gesture_*_summary.json")):
        try:
            row = json.loads(p.read_text())
        except Exception:
            continue
        eid = row.get("exp_id") or p.name.replace("_summary.json", "")
        row.setdefault("exp_id", eid)
        row.setdefault("status", "ok")
        if eid not in by_id:
            by_id[eid] = row
            summaries.append(row)
        elif by_id[eid].get("test_acc") is None and row.get("test_acc") is not None:
            by_id[eid].update(row)

    out_json = Path("log/gesture_results.json")
    out_json.write_text(json.dumps(summaries, indent=2))
    write_results_md(summaries, args.out_md)
    print(args.out_md.read_text())


if __name__ == "__main__":
    main()
