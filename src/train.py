"""Trains the product classification CNN.

Fine-tunes a pretrained ResNet18 (transfer learning, explicitly allowed by
the assignment brief) on the grocery product dataset and saves the best
checkpoint plus the class index mapping for use by classification.py.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # we only save figures, never show them; avoids GUI/Qt backend issues
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from src import config


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_datasets(data_dir: Path, val_split: float, seed: int):
    train_tf, eval_tf = build_transforms()
    train_base = datasets.ImageFolder(str(data_dir), transform=train_tf)
    eval_base = datasets.ImageFolder(str(data_dir), transform=eval_tf)

    targets = [label for _, label in train_base.samples]
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=val_split, stratify=targets, random_state=seed,
    )

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(eval_base, val_idx)
    return train_ds, val_ds, train_base.classes


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the product classifier CNN.")
    parser.add_argument("--data-dir", default=str(config.DATA_DIR))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, classes = build_datasets(Path(args.data_dir), args.val_split, args.seed)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            config.CLASSES_PATH.write_text(json.dumps(classes, indent=2))

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model to {config.MODEL_PATH}")

    _plot_training_curves(history)
    _evaluate_final(model, val_loader, classes, device)


def _plot_training_curves(history: dict):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path = config.OUTPUT_DIR / "training_curves.png"
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def _evaluate_final(model, val_loader, classes, device):
    import matplotlib.pyplot as plt

    # Reload the best checkpoint (last epoch isn't necessarily the best one).
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print("\nPer-class validation report:")
    print(report)
    (config.OUTPUT_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = config.OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
