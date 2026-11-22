# Hybrid C — system diagram

**Inverse** = thermal map → physics primitives  
**Forward** = physics primitives → reconstructed thermal map  

Parallel paths: physics branch and TAPOR pose branch, then **gated late fusion** on joints.

```mermaid
flowchart TB
  subgraph Input["Input"]
    T["Thermal map T (°C)<br/>B×1×H×W"]
    Ta["Ambient Ta"]
  end

  AN["Ambient normalize<br/>Tn = T − Ta"]

  subgraph Inverse["Inverse model — thermal → physics<br/>InverseUNet (models/common.py)"]
    Inv["InverseUNet"]
    Z["Depth z"]
    Eps["Emissivity ε"]
    Tt["Surface temp Tt"]
    R["Reflection R"]
    M["Mask logits m"]
    Inv --> Z
    Inv --> Eps
    Inv --> Tt
    Inv --> R
    Inv --> M
  end

  subgraph Forward["Forward model — physics → thermal<br/>ForwardThermal (models/forward_thermal.py)"]
    Fwd["Stefan–Boltzmann + reflection<br/>+ depth attenuation + residual CNN"]
    That["Reconstructed thermal T̂"]
    Fwd --> That
  end

  subgraph Pose["Pose path — parallel TAPOR retrieval on Tn only"]
    Thermo["ThermoSpaceEncoder(Tn)"]
    KP["KPFeatureExtractor(Tn)"]
    Dec["PositionRetrievalDecoder"]
    J0["Joints₀ 21×3"]
  end

  subgraph Fuse["Gated late fusion"]
    Cue["Physics cue<br/>mean/masked/max z, mean ε, Tt, Ta"]
    MLP["Fusion MLP"]
    Gate["gate g = σ(·)"]
    Delta["Δ joints"]
    J["Joints = Joints₀ + g·Δ"]
  end

  subgraph Losses["Training losses"]
    LJ["Joint + bone"]
    LD["Depth / hand-depth"]
    LR["Recon ‖T̂ − T‖"]
    LM["Mask BCE"]
  end

  T --> AN
  Ta --> AN
  AN --> Inv
  AN --> Thermo
  AN --> KP

  Z --> Fwd
  Eps --> Fwd
  Tt --> Fwd
  R --> Fwd
  Ta --> Fwd

  Thermo --> Dec
  KP --> Dec
  Dec --> J0

  Z -.->|"stop-grad in phase 2"| Cue
  Eps -.-> Cue
  Tt -.-> Cue
  M -.-> Cue
  Ta --> Cue
  J0 --> MLP
  Cue --> MLP
  MLP --> Delta
  MLP --> Gate
  J0 --> J
  Delta --> J
  Gate --> J

  J --> LJ
  Z --> LD
  That --> LR
  M --> LM
  T --> LR
```

```mermaid
flowchart LR
  subgraph Direction["Physics direction of each head"]
    I["Inverse<br/>thermal map → physics<br/>(z, ε, Tt, R, m)"]
    F["Forward<br/>physics → thermal map<br/>(T̂)"]
  end
  subgraph Coupling["How C differs from A"]
    P["Pose runs on Tn alone<br/>physics does not pack into encoder"]
    G["Physics only enters via<br/>cue → gated residual on joints"]
  end
```
