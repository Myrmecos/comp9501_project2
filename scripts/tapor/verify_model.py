#!/usr/bin/env python3
"""Engineering-only synthetic forward test for the unchanged official TAPOR base1 model."""
import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapor-root", type=Path, default=Path("/workspace/TAPOR"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise SystemExit("Use batch-size >= 2: BatchNorm in the official architecture needs more than one sample for train mode.")
    if not (args.tapor_root / "tapor_model_config.py").is_file():
        raise SystemExit(f"Not a TAPOR source root: {args.tapor_root}")
    sys.path.insert(0, str(args.tapor_root))
    import torch
    from models import Tapor
    from tapor_model_config import base1_setting

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    config = copy.deepcopy(base1_setting)
    config["temporal_keypoints_fusion_param"]["num_history"] = 0  # official CLI -fs 1 behavior
    model = Tapor(
        config["spatial_encoder_param"], config["keypoints_encoder_param"],
        config["cross_keypoints_fusion_param"], config["temporal_keypoints_fusion_param"],
        config["handpose_encoder_param"], input_width=32, input_height=24,
        batch_size=args.batch_size, train=True, device=device,
    ).to(device)
    checkpoint_result = "not requested"
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=True)
        checkpoint_result = "strict state_dict load succeeded"
    model.eval()
    synthetic_input = torch.randn(args.batch_size, 1, 24, 32, device=device)
    with torch.no_grad():
        outputs = model(synthetic_input)
    joints = outputs[0]
    if tuple(joints.shape) != (args.batch_size, 21, 3):
        raise RuntimeError(f"Unexpected official model output shape: {tuple(joints.shape)}")
    if not torch.isfinite(joints).all():
        raise RuntimeError("Non-finite model output")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "SYNTHETIC engineering smoke test only. This is not an evaluation and has no performance metric.",
        "tapor_root": str(args.tapor_root.resolve()),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "configuration": "official base1_setting with official -fs 1 override (num_history=0)",
        "synthetic_input_shape": list(synthetic_input.shape),
        "output_shapes": [list(item.shape) for item in outputs],
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_result": checkpoint_result,
        "finite_output": True,
    }
    report_path = args.report
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report: {report_path}")
    print("PASS: official TAPOR base1 synthetic forward test completed; no data evaluation was performed.")


if __name__ == "__main__":
    main()
