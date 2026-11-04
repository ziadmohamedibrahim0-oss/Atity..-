"""Train the tomato-leaf disease classifier (ResNet-18 transfer learning).

Folder layout (torchvision ImageFolder):  <data>/<class_name>/*.jpg
The PlantVillage tomato folders work as they are, e.g. data/plantvillage_tomato/Tomato___Early_blight/

Examples:
  python scripts/train_disease.py --data data/plantvillage_tomato --epochs 8
  python scripts/train_disease.py --data data/plantvillage_tomato --external-test data/plantdoc_tomato
Use --workers 0 on Windows. --scratch skips the pretrained weights (offline smoke tests only).
The --external-test folder (field photos) is the number that tells you how the model behaves in the field.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision import transforms as T

from agrisense.disease import MEAN, STD, canonical, eval_transform


def train_transform(size):
    return T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomRotation(20), T.ColorJitter(0.25, 0.25, 0.25, 0.05), T.ToTensor(),
                      T.Normalize(MEAN, STD)])


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, ys, ps = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            out = model(x)
            loss = loss_fn(out, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * len(y)
        ys += y.tolist()
        ps += out.argmax(1).tolist()
    return total / len(loader.dataset), accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def evaluate(model, loader, class_names, device, prefix):
    """Write <prefix>_report.txt and <prefix>_confusion.csv; return the headline metrics."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ys += y.tolist()
            ps += model(x.to(device)).argmax(1).tolist()
    labels = list(range(len(class_names)))
    report = classification_report(ys, ps, labels=labels, target_names=class_names, zero_division=0)
    Path(f"{prefix}_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(f"{prefix}_confusion.csv", confusion_matrix(ys, ps, labels=labels), fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    print(f"\n[{Path(prefix).name}]\n{report}")
    return {"images": len(ys), "accuracy": accuracy_score(ys, ps), "macro_f1": f1_score(ys, ps, average="macro")}


def external_loader(path, classes, transform, batch, workers):
    """Field photos in ImageFolder layout; folders are matched to the training classes by name."""
    ds = datasets.ImageFolder(path, transform=transform)
    to_idx = {canonical(c): i for i, c in enumerate(classes)}
    ds.samples = [(p, to_idx[canonical(ds.classes[k])]) for p, k in ds.samples if canonical(ds.classes[k]) in to_idx]
    ds.targets = [k for _, k in ds.samples]
    if not ds.samples:
        raise SystemExit("No folder of --external-test matches a training class")
    return DataLoader(ds, batch_size=batch, num_workers=workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="training folder (ImageFolder layout)")
    ap.add_argument("--external-test", help="independent field-photo folder (ImageFolder layout)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scratch", action="store_true", help="no pretrained weights")
    ap.add_argument("--out", default="models/tomato_resnet18.pt")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds_train = datasets.ImageFolder(a.data, transform=train_transform(a.size))
    ds_eval = datasets.ImageFolder(a.data, transform=eval_transform(a.size))
    classes, y = ds_train.classes, np.array(ds_train.targets)
    idx_tr, idx_rest = train_test_split(np.arange(len(y)), test_size=0.30, stratify=y, random_state=a.seed)
    idx_va, idx_te = train_test_split(idx_rest, test_size=0.50, stratify=y[idx_rest], random_state=a.seed)

    def loader(ds, idx, shuffle):
        return DataLoader(Subset(ds, idx), batch_size=a.batch, shuffle=shuffle, num_workers=a.workers)

    dl_tr, dl_va, dl_te = loader(ds_train, idx_tr, True), loader(ds_eval, idx_va, False), loader(ds_eval, idx_te, False)

    weights = None if a.scratch else models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    best_f1, best_state = -1.0, None
    for epoch in range(1, a.epochs + 1):
        l_tr, acc_tr, _ = run_epoch(model, dl_tr, loss_fn, device, optimizer)
        l_va, acc_va, f1_va = run_epoch(model, dl_va, loss_fn, device)
        scheduler.step()
        print(f"epoch {epoch}/{a.epochs}  train loss {l_tr:.3f} acc {acc_tr:.3f} | "
              f"val loss {l_va:.3f} acc {acc_va:.3f} macro-F1 {f1_va:.3f}")
        if f1_va > best_f1:
            best_f1, best_state = f1_va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "resnet18", "classes": classes, "img_size": a.size, "state_dict": best_state}, out)

    prefix = str(out.with_suffix(""))
    results = {"classes": classes, "best_val_macro_f1": best_f1,
               "internal_test": evaluate(model, dl_te, classes, device, prefix + "_internal")}
    if a.external_test:
        ext = external_loader(a.external_test, classes, eval_transform(a.size), a.batch, a.workers)
        results["external_test"] = evaluate(model, ext, classes, device, prefix + "_external")
    Path(prefix + "_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved model to {out}\nmetrics: {json.dumps({k: v for k, v in results.items() if k != 'classes'}, indent=2)}")


if __name__ == "__main__":
    main()
