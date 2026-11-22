# TAPOR local data manifest

Downloaded 2026-11-16 into `data/tapor/`.

| Archive | Drive ID | Local path | Zip size | Extracted | Pickles |
|---------|----------|------------|----------|-----------|---------|
| Training Dataset (pose) | `12Ppoh15nomnVAKo2UFQRqgcLeDRh5hGk` | `Trainset/` | 415 MB | 1.6 GB | **123** |
| Test Dataset (pose) | `1cnwdelIiV4V15rq3ANN93YsNvAjpvV5O` | `Testset/` | 1.2 GB | 5.3 GB | **478** |
| Gesture Recognition | `1JmTD0N_9hdhCS8JOSd9t2bQ_NmB9BbdB` | `gesture/` (from `GR_Dataset/`) | 272 MB | 1.2 GB | **28** |

Raw zips kept under `raw/`.

## Shared pickle schema (all three)

- `thermal_array_temperature_maps` — per-frame **24×32** float (°C; zeros appear as background fill)
- `thermal_array_ambient_temperatures` — per-frame scalar
- `label_depth_maps` — per-frame **120×160** uint16 (RealSense-scale depth, mm-like)
- `label_2d_joints` — per-frame (21,3) or empty
- `label_3d_joints` — per-frame (21,3) or empty
- `label_hand_depth` — per-frame scalar (meters; train median ~0.34 m)

## Frame totals (approx)

- Trainset: 123 × 300 = **36,900** frames
- Testset: sum ≈ **124,400** frames (100–1000 per file)
- Gesture: 28 × 1000 = **28,000** frames; classes gesture0–gesture6 × 4 files each

See `inspection_summary.json` for detailed stats.
