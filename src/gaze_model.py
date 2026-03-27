import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Shared loss / metric helpers
# ---------------------------------------------------------------------------

def gaze_to_3d(gaze):
    """Convert (pitch, yaw) in radians to unit 3D vector."""
    pitch, yaw = gaze[:, 0], gaze[:, 1]
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=1)


def angular_error(pred, target):
    """Mean angular error in degrees between predicted and target gaze vectors."""
    pred_3d   = gaze_to_3d(pred)
    target_3d = gaze_to_3d(target)
    pred_norm = nn.functional.normalize(pred_3d,   dim=1)
    tgt_norm  = nn.functional.normalize(target_3d, dim=1)
    cos_sim   = (pred_norm * tgt_norm).sum(dim=1).clamp(-1, 1)
    return torch.acos(cos_sim).mean() * (180.0 / torch.pi)


def angular_loss(pred, target):
    """Differentiable angular loss in radians."""
    pred_3d   = gaze_to_3d(pred)
    target_3d = gaze_to_3d(target)
    pred_norm = nn.functional.normalize(pred_3d,   dim=1)
    tgt_norm  = nn.functional.normalize(target_3d, dim=1)
    cos_sim   = (pred_norm * tgt_norm).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos_sim).mean()


# ---------------------------------------------------------------------------
# Shared regression head
# ---------------------------------------------------------------------------

def _make_head(in_features, hidden=256, dropout=0.3):
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.Hardswish(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, 64),
        nn.Hardswish(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(64, 2),
    )


# ---------------------------------------------------------------------------
# GazeNet — MobileNetV3-Small backbone  (original)
# ---------------------------------------------------------------------------

class GazeNet(nn.Module):
    """
    MobileNetV3-Small backbone.
    Inputs : eye_crop (B,1,32,64), head_pose (B,3)
    Output : (B,2) — (pitch, yaw) in radians
    """

    def __init__(self, freeze_backbone=False):
        super().__init__()
        mobilenet     = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = _make_head(576 + 3)

    def forward(self, eye_crop, head_pose):
        x     = eye_crop.expand(-1, 3, -1, -1)
        feats = self.pool(self.backbone(x)).flatten(1)
        return self.head(torch.cat([feats, head_pose], dim=1))


# ---------------------------------------------------------------------------
# GazeNetEfficientNet — EfficientNet-B0 backbone
# ---------------------------------------------------------------------------

class GazeNetEfficientNet(nn.Module):
    """
    EfficientNet-B0 backbone (1280-dim features).
    Inputs : eye_crop (B,1,32,64), head_pose (B,3)
    Output : (B,2) — (pitch, yaw) in radians
    """

    def __init__(self, freeze_backbone=False):
        super().__init__()
        eff            = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone  = eff.features
        self.pool      = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = _make_head(1280 + 3)

    def forward(self, eye_crop, head_pose):
        x     = eye_crop.expand(-1, 3, -1, -1)
        feats = self.pool(self.backbone(x)).flatten(1)
        return self.head(torch.cat([feats, head_pose], dim=1))


# ---------------------------------------------------------------------------
# GazeNetResNet — ResNet-18 backbone
# ---------------------------------------------------------------------------

class GazeNetResNet(nn.Module):
    """
    ResNet-18 backbone (512-dim features).
    Inputs : eye_crop (B,1,32,64), head_pose (B,3)
    Output : (B,2) — (pitch, yaw) in radians
    """

    def __init__(self, freeze_backbone=False):
        super().__init__()
        resnet         = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Drop the final FC layer
        self.backbone  = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = _make_head(512 + 3)

    def forward(self, eye_crop, head_pose):
        x     = eye_crop.expand(-1, 3, -1, -1)
        feats = self.backbone(x).flatten(1)
        return self.head(torch.cat([feats, head_pose], dim=1))


# ---------------------------------------------------------------------------
# GazeNetWithIris — MobileNetV3 + iris position ratios
# ---------------------------------------------------------------------------

class GazeNetWithIris(nn.Module):
    """
    MobileNetV3-Small + iris ratios from MediaPipe (4 extra values:
    left_iris_x, left_iris_y, right_iris_x, right_iris_y).

    Inputs : eye_crop   (B,1,32,64)
             head_pose  (B,3)
             iris_ratio (B,4)  — set to zeros if unavailable
    Output : (B,2) — (pitch, yaw) in radians
    """

    def __init__(self, freeze_backbone=False):
        super().__init__()
        mobilenet     = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = _make_head(576 + 3 + 4)

    def forward(self, eye_crop, head_pose, iris_ratio=None):
        x     = eye_crop.expand(-1, 3, -1, -1)
        feats = self.pool(self.backbone(x)).flatten(1)
        if iris_ratio is None:
            iris_ratio = torch.zeros(x.size(0), 4, device=x.device)
        return self.head(torch.cat([feats, head_pose, iris_ratio], dim=1))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "mobilenet":    GazeNet,
    "efficientnet": GazeNetEfficientNet,
    "resnet18":     GazeNetResNet,
    "iris":         GazeNetWithIris,
}


def build_model(arch="mobilenet", dropout=0.3, **kwargs):
    """
    Instantiate a gaze model by name.

    Args:
        arch    : one of 'mobilenet', 'efficientnet', 'resnet18', 'iris'
        dropout : dropout rate in the regression head (default 0.3)
        **kwargs: passed to the model constructor (e.g. freeze_backbone=True)

    Returns:
        nn.Module
    """
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(_REGISTRY)}")
    # Patch head dropout by temporarily overriding _make_head default
    import functools
    original = globals()["_make_head"]
    globals()["_make_head"] = functools.partial(original, dropout=dropout)
    model = _REGISTRY[arch](**kwargs)
    globals()["_make_head"] = original
    return model


def list_archs():
    return list(_REGISTRY.keys())
