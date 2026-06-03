from pathlib import Path

import torch
import torch.nn as nn

def _unpack_batch(batch):
    if isinstance(batch, dict):
        image_keys = ("images", "image", "x", "inputs", "sequence")
        label_keys = ("labels", "label", "y", "target", "targets")

        x = next((batch[key] for key in image_keys if key in batch), None)
        y = next((batch[key] for key in label_keys if key in batch), None)
        if x is None or y is None:
            raise KeyError(
                "Batch dict must contain image sequence and label keys, "
                "for example 'images' and 'labels'."
            )
        return x, y

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError("Batch must be a tuple/list (images, labels) or a dict.")


def _to_device(x, y, device):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.float()

    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long)
    else:
        y = y.long()

    if y.ndim > 1 and y.size(-1) == 1:
        y = y.view(-1)

    return x.to(device), y.to(device)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion=None,
    device=None,
    grad_clip=None,
):
    criterion = criterion or nn.CrossEntropyLoss()
    device = device or next(model.parameters()).device
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        x, y = _unpack_batch(batch)
        x, y = _to_device(x, y, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate_model(model, data_loader, criterion=None, device=None):
    criterion = criterion or nn.CrossEntropyLoss()
    device = device or next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        x, y = _unpack_batch(batch)
        x, y = _to_device(x, y, device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    optimizer=None,
    criterion=None,
    device=None,
    grad_clip=1.0,
    scheduler=None,
    checkpoint_path=None,
):
    """
    Train CNN + LSTM classifier on skeleton image sequences.

    Expected batch:
        images shape = (batch_size, 10, 3, height, width)
        labels shape = (batch_size,)

    Labels:
        Activity1 -> 0
        Activity2 -> 1
        ...
        Activity11 -> 10
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }

        if val_loader is not None:
            val_metrics = evaluate_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
            )
            row["val_loss"] = val_metrics["loss"]
            row["val_accuracy"] = val_metrics["accuracy"]

            if checkpoint_path is not None and row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        if scheduler is not None:
            if "val_loss" in row:
                scheduler.step(row["val_loss"])
            else:
                scheduler.step()

        history.append(row)

        message = (
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f}"
        )
        if "val_loss" in row:
            message += (
                f" val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.4f}"
            )
        print(message)

    return history
