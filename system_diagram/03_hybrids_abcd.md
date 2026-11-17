# Hybrid structures A–D

Pose path always ends in **KPFeatureExtractor → PositionRetrievalDecoder → joints**.  
Dashed edges = optional **stop-grad** from physics → pose (phase-2 training).

## A — Inverse → pack maps → TAPOR-style retrieval

```mermaid
flowchart TB
  Tn["T_n"] --> InvA["InverseUNet"]
  InvA --> PrimA["z, ε, Tt, R, m"]
  PrimA --> FwdA["ForwardThermal → T̂"]
  PrimA -.->|"detach optional"| PackA["pack: T_n, z, ε, Tt"]
  PackA --> ThermoA["ThermoSpaceEncoder"]
  PackA --> KPA["KPFeatureExtractor"]
  PrimA -.-> CueA["FiLM cue: mean(z,ε,Tt), Ta"]
  CueA --> FiLMA["FiLM"]
  ThermoA --> FiLMA
  FiLMA --> DecA["PositionRetrievalDecoder"]
  KPA --> DecA
  DecA --> JA["Joints 21×3"]
```

## B — Shared ThermoSpace trunk → prim heads + pose

```mermaid
flowchart TB
  TnB["T_n"] --> ThermoB["ThermoSpaceEncoder"]
  ThermoB --> HeadB["Primitive head → z, ε, Tt, R, m"]
  HeadB --> FwdB["ForwardThermal → T̂"]
  ThermoB --> FiLMB["FiLM + physics cue"]
  HeadB -.-> FiLMB
  HeadB -.-> KPinB["cat(T_n, z, ε)"]
  KPinB --> KPB["KPFeatureExtractor"]
  FiLMB --> DecB["PositionRetrievalDecoder"]
  KPB --> DecB
  DecB --> JB["Joints 21×3"]
```

## C — Parallel physics + TAPOR, gated late fusion on joints

```mermaid
flowchart TB
  TnC["T_n"] --> InvC["InverseUNet"]
  TnC --> ThermoC["ThermoSpaceEncoder"]
  TnC --> KPC["KPFeatureExtractor"]
  InvC --> PrimC["primitives + T̂"]
  ThermoC --> DecC["PositionRetrievalDecoder"]
  KPC --> DecC
  DecC --> J0["Joints₀"]
  PrimC -.-> CueC["depth / ε / Tt stats + Ta"]
  J0 --> Fuse["Gated fusion MLP"]
  CueC --> Fuse
  Fuse --> JC["Joints = Joints₀ + g·Δ"]
```

## D — Physics maps build spatial space; KP queries it

```mermaid
flowchart TB
  TnD["T_n"] --> InvD["InverseUNet"]
  InvD --> PrimD["z, ε, Tt, R, m + feat"]
  PrimD --> FwdD["ForwardThermal → T̂"]
  PrimD -.-> PackD["pack: T_n, z, ε, Tt, m"]
  PackD --> SpatD["Spatial builder CNN"]
  PrimD -.->|"inv feat + 1×1"| SpatD
  PrimD -.-> FiLMD["FiLM cue"]
  SpatD --> FiLMD
  PrimD -.-> KPinD["cat(T_n, z)"]
  KPinD --> KPD["KPFeatureExtractor"]
  FiLMD --> DecD["PositionRetrievalDecoder"]
  KPD --> DecD
  DecD --> JD["Joints 21×3"]
```

## Comparison (coupling style)

```mermaid
flowchart LR
  A["A: physics maps<br/>as encoder input"]
  B["B: shared trunk<br/>branch prim + pose"]
  C["C: parallel paths<br/>fuse at joints"]
  D["D: physics builds<br/>spatial memory"]
```
