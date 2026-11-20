# Candidate hypotheses (micro search)

**Preserve:** original `weights/hybridA_full` / `hybridA_nofwd_full` are never overwritten.
All work here uses `--artifact-root small_ws` and (mostly) `modified_A`.

Each candidate = paired `full` (Forward on) vs `nofwd` (`λ_recon=0`).

## Wave 1 (done)

| ID | Hypothesis | Result gap |
|----|------------|------------|
| C0 | ESC-S1 control | −1.18 |
| C1 | tiny λ_recon=0.05 | **+0.021** |
| C2 | recon warmup → phase2 λ=0 | **+0.036** (best w1) |
| C3 | keep_depth_attached=false | −0.68 |
| C4 | SSL→freeze→FT | −0.40 |
| C5 | strong depth / weak recon | −0.004 |

## Wave 2 (new)

| ID | Hypothesis | Knobs |
|----|------------|-------|
| **C6** | Push C1∩C2 | λ_recon p1=0.02, p2=0 |
| **C7** | Escalation hurt? | C2 schedule, **no** mask/inv-skip |
| **C8** | Partial depth grad | `depth_attach_alpha=0.2` + weak recon |
| **C9** | Longer joint physics | phase1=16/20, recon warmup 0.5→0 |
| **C10** | Recon cannot corrupt inverse | `recon_detach_primitives=true` |
| **C11** | Plain Hybrid **A** control | model=`A`, C2 recon schedule |

Promote micro gap ≥ **0.05** (stretch 0.08) to larger data. Fast screen: 20 epochs.
