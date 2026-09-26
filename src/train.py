from pathlib import Path
import json
import yaml
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from data import make_loaders, seed_everything
from model import build_model


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)
    losses, y_true, y_pred, y_prob = [], [], [], []

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        losses.append(loss.item())
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())

    metrics = {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, classes = make_loaders(
        cfg["paths"]["data_root"],
        test_root=cfg["paths"]["test_root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
    )

    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_f1 = -1
    patience_counter = 0
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(f"\nEpoch {epoch}")
        print("Train:", train_metrics)
        print("Val:", val_metrics)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            Path("models").mkdir(exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "config": cfg,
                },
                cfg["paths"]["model_path"],
            )
        else:
            patience_counter += 1

        if patience_counter >= cfg["training"]["patience"]:
            print("Early stopping.")
            break

    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nBest model saved to:", cfg["paths"]["model_path"])


if __name__ == "__main__":
    main()
