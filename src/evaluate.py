from pathlib import Path
import json
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from data import make_loaders
from model import build_model


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, classes = make_loaders(
        cfg["paths"]["data_root"],
        test_root=cfg["paths"]["test_root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
    )

    checkpoint = torch.load(
        cfg["paths"]["model_path"],
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("PneumoVision Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PneumoVision ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curve.png", dpi=200)
    plt.close()

    with open("reports/metrics/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"classes": classes, "roc_auc": auc, "classification_report": report},
            f,
            indent=2,
        )

    print(json.dumps({"roc_auc": auc, "classification_report": report}, indent=2))


if __name__ == "__main__":
    main()
