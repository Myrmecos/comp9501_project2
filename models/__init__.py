from .tapor_full import TaporFull
from .hybrid_a import HybridA
from .hybrid_a_modified import HybridAModified
from .hybrid_b import HybridB
from .hybrid_c import HybridC
from .hybrid_d import HybridD
from .mediapipe3d import MediaPipe3D
from .nanotapor import NanoTapor
from .hand3d import Hand3D
from .resnet_pose import ResNetPose

MODEL_REGISTRY = {
    "vanilla": TaporFull,  # full-size official TAPOR base1 baseline
    "A": HybridA,
    "modified_A": HybridAModified,
    "B": HybridB,
    "C": HybridC,
    "D": HybridD,
    "mediapipe": MediaPipe3D,
    "nanotapor": NanoTapor,
    "hand3d": Hand3D,
    "resnet": ResNetPose,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name}; choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
