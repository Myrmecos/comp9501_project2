# PhyPose family — four coupling variants

**PhyPose** = physics primitives (inverse ↔ forward) + TAPOR-style pose retrieval  
(keypoints query a spatial map → 21×3 joints).

The four models share the same two ingredients; they differ only in **where physics is fused into the pose pipeline**—at the input (**Early**), in shared mid-level features (**Trunk**), or at the final joints (**Late**). **Map** is physics-first spatial construction (joints query a map built from physics), not on the early–late axis.

| Name | Code (legacy) | Fusion point | One-liner |
|------|---------------|--------------|-----------|
| **PhyPose-Early** | Hybrid A / `hybrid_a.py` | Input | Physics maps stacked as multi-channel input to pose encoding & retrieval |
| **PhyPose-Trunk** | Hybrid B / `hybrid_b.py` | Shared trunk | Shared trunk features branch into physics and pose heads |
| **PhyPose-Late** | Hybrid C / `hybrid_c.py` | Joints | Parallel physics & pose; gated residual fusion on joints |
| **PhyPose-Map** | Hybrid D / `hybrid_d.py` | Spatial map | Spatial feature map built from physics maps; joints query that map |

Dashed edges in diagrams = optional **stop-grad** from physics → pose (phase-2 training).

```mermaid
flowchart LR
  subgraph axis["Fusion along the pose pipeline"]
    Early["Early<br/>fuse at input"]
    Trunk["Trunk<br/>fuse via shared F"]
    Late["Late<br/>fuse at joints"]
  end
  Early -.-> Trunk
  Trunk -.-> Late
  Map["Map<br/>physics builds spatial"] -.->|"physics-first"| Early
```

---

## Shared pieces

1. **Ambient normalize:** \(T_n = T - T_a\)
2. **Inverse:** thermal → primitives \(z\) (depth), \(\varepsilon\) (emissivity), \(T_t\) (surface temp), \(R\) (reflection), \(m\) (mask)
3. **Forward:** primitives → reconstructed thermal \(\hat{T}\) (reconstruction loss under hand mask)
4. **Pose path:** spatial features + 21 joint queries → `PositionRetrievalDecoder` → joints  
   (Early / Trunk / Late use `ThermoSpaceEncoder` + `KPFeatureExtractor`; Map uses a physics-driven spatial builder)

---

## PhyPose-Early — fuse at input

Physics attribute maps stacked with \(T_n\) as multi-channel input to pose encoding & retrieval.

```mermaid
flowchart TB
  Tn["T_n"] --> Inv["InverseUNet"]
  Inv --> Prim["z, ε, Tt, R, m"]
  Prim --> Fwd["ForwardThermal → T̂"]
  Prim -.->|"detach optional"| Pack["pack: Tn, z, ε, Tt"]
  Pack --> Thermo["ThermoSpaceEncoder"]
  Pack --> KP["KPFeatureExtractor"]
  Prim -.-> Cue["FiLM cue: mean(z,ε,Tt), Ta"]
  Cue --> FiLM["FiLM"]
  Thermo --> FiLM
  FiLM --> Dec["PositionRetrievalDecoder"]
  KP --> Dec
  Dec --> J["Joints 21×3"]
```

**Coupling:** physics enters pose as **extra input channels** (+ FiLM).

---

## PhyPose-Trunk — fuse via shared features

One ThermoSpace trunk; physics head and pose head branch from the same features.

```mermaid
flowchart TB
  Tn["T_n"] --> Thermo["ThermoSpaceEncoder → F"]
  Thermo --> Head["Primitive head → z, ε, Tt, R, m"]
  Head --> Fwd["ForwardThermal → T̂"]
  Thermo --> FiLM["FiLM + physics cue"]
  Head -.-> FiLM
  Head -.-> KPin["cat(Tn, z, ε)"]
  KPin --> KP["KPFeatureExtractor"]
  FiLM --> Dec["PositionRetrievalDecoder"]
  KP --> Dec
  Dec --> J["Joints 21×3"]
```

**Coupling:** physics and pose **share mid-level features**; primitives also cue FiLM / KP inputs.

---

## PhyPose-Late — fuse at joints

Parallel physics and pose paths; gated residual fusion on joints  
\(J = J_0 + g\cdot\Delta\).

```mermaid
flowchart TB
  Tn["T_n"] --> Inv["InverseUNet"]
  Tn --> Thermo["ThermoSpaceEncoder"]
  Tn --> KP["KPFeatureExtractor"]
  Inv --> Prim["primitives + T̂"]
  Thermo --> Dec["PositionRetrievalDecoder"]
  KP --> Dec
  Dec --> J0["Joints₀"]
  Prim -.-> Cue["depth / ε / Tt stats + Ta"]
  J0 --> Fuse["Gated fusion MLP"]
  Cue --> Fuse
  Fuse --> J["Joints = Joints₀ + g·Δ"]
```

**Coupling:** pose runs on \(T_n\) alone until the end; physics only **corrects joints** via a learned gate.

---

## PhyPose-Map — physics builds the spatial map

Spatial feature map constructed from physics maps (+ inverse feature residual); joints query that map.

```mermaid
flowchart TB
  Tn["T_n"] --> Inv["InverseUNet"]
  Inv --> Prim["z, ε, Tt, R, m + feat"]
  Prim --> Fwd["ForwardThermal → T̂"]
  Prim -.-> Pack["pack: Tn, z, ε, Tt, m"]
  Pack --> Spat["Spatial builder CNN"]
  Prim -.->|"inv feat + 1×1"| Spat
  Prim -.-> FiLM["FiLM cue"]
  Spat --> FiLM
  Prim -.-> KPin["cat(Tn, z)"]
  KPin --> KP["KPFeatureExtractor"]
  FiLM --> Dec["PositionRetrievalDecoder"]
  KP --> Dec
  Dec --> J["Joints 21×3"]
```

**Coupling:** physics **constructs** the spatial memory that pose retrieves from (not standard ThermoSpace on \(T_n\) alone).

---

## Comparison (coupling style)

```mermaid
flowchart LR
  Early["Early: physics maps<br/>as encoder input"]
  Trunk["Trunk: shared features<br/>branch prim + pose"]
  Late["Late: parallel paths<br/>fuse at joints"]
  Map["Map: physics builds<br/>spatial memory"]
```
