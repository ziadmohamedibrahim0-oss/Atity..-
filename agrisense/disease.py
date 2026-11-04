"""Plant-disease module: class-name mapping, CNN inference (PyTorch) and a rule-based lesion estimator (OpenCV)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "tomato_resnet18.pt"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet statistics

_ALIASES = (("bacterial", "bacterial_spot"), ("early_blight", "early_blight"), ("late_blight", "late_blight"),
            ("mold", "leaf_mold"), ("septoria", "septoria_leaf_spot"), ("mite", "spider_mites"),
            ("target", "target_spot"), ("yellow", "yellow_leaf_curl_virus"), ("mosaic", "mosaic_virus"),
            ("healthy", "healthy"))


def canonical(label: str) -> str:
    """Map a class or folder name (e.g. 'Tomato___Late_blight') to a knowledge-base key."""
    s = label.lower().replace("-", "_").replace(" ", "_")
    for part, key in _ALIASES:
        if part in s:
            return key
    return "healthy" if s.replace("tomato", "").strip("_") in ("", "leaf") else s


def eval_transform(size=224):
    from torchvision import transforms as T
    return T.Compose([T.Resize(round(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


class Classifier:
    """Wraps a trained ResNet-18 and its class list."""

    def __init__(self, model, classes, size=224):
        self.model, self.classes, self.size = model, list(classes), size
        self._tf = eval_transform(size)

    def predict(self, img: Image.Image, topk=3):
        """Return [(class_name, probability), ...] sorted by probability."""
        import torch
        x = self._tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        vals, idx = probs.topk(min(topk, len(self.classes)))
        return [(self.classes[i], float(v)) for v, i in zip(vals.tolist(), idx.tolist())]


def load_classifier(path=MODEL_PATH):
    """Return a Classifier, or None when PyTorch or the trained checkpoint is not available."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        import torch
        from torchvision import models
    except ImportError:
        return None
    ckpt = torch.load(path, map_location="cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return Classifier(model, ckpt["classes"], ckpt.get("img_size", 224))


def estimate_severity(img: Image.Image, max_side=512):
    """Rule-based lesion estimate for ONE leaf photographed on a plain background.

    Green tissue is healthy; yellow / brown / reddish pixels inside the leaf outline count as lesions.
    This is a colour heuristic, not a diagnosis: it fails on cluttered backgrounds (soil, hands) and
    cannot tell diseases apart. Returns {"leaf_pct", "severity_pct", "overlay" (PIL image)}.
    """
    rgb = np.asarray(img.convert("RGB")).copy()
    scale = max_side / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))      # H 0-179, S and V 0-255
    green = (h >= 32) & (h <= 85) & (s >= 45) & (v >= 35)
    odd = ((h < 32) | (h > 170)) & (s >= 45) & (v >= 30)          # yellow / brown / red tissue
    near_green = cv2.dilate(green.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    cand = cv2.morphologyEx((green | (odd & near_green)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros(cand.shape, np.uint8)
    cv2.drawContours(region, [c for c in contours if cv2.contourArea(c) >= 0.005 * cand.size], -1, 1, thickness=-1)
    leaf = region > 0
    lesion = odd & leaf
    overlay = rgb.copy()
    overlay[lesion] = (0.4 * overlay[lesion] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    return {"leaf_pct": 100 * float(leaf.mean()), "severity_pct": 100 * float(lesion.sum()) / max(int(leaf.sum()), 1),
            "overlay": Image.fromarray(overlay)}
