"""Training helpers for sequence-level skeleton-image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


SEQUENCE_KEYS = (
    "skeleton_paths",
)
LABEL_KEYS = (
    "label",
)


def _progress_iterator(data_loader, enabled: bool, desc: str | None):
    """Wrap a dataloader in tqdm when progress bars are enabled and available."""

    if not enabled or tqdm is None:
        return data_loader, None

    try:
        total = len(data_loader)
    except TypeError:
        total = None

    progress_bar = tqdm(
        data_loader,
        total=total,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )
    return progress_bar, progress_bar


def _unpack_batch(batch: Any) -> tuple[Any, Any]:
    """Extract sequence tensors and sequence labels from tuple/list or dict batches."""

    if isinstance(batch, dict):
        x = next((batch[key] for key in SEQUENCE_KEYS if key in batch), None) # next() returns the first matching value or None if no keys are found. batch ? is a dict containing the data for one batch, and SEQUENCE_KEYS is a tuple of possible keys that could contain the sequence data. The generator expression iterates over SEQUENCE_KEYS and checks if each key is present in the batch dict. If it finds a key that exists in the batch, it retrieves the corresponding value (which should be the sequence data) and returns it as x. If none of the keys in SEQUENCE_KEYS are found in the batch, x will be set to None.
        y = next((batch[key] for key in LABEL_KEYS if key in batch), None) # next() returns the first matching value or None if no keys are found. batch ? is a dict containing the data for one batch, and LABEL_KEYS is a tuple of possible keys that could contain the sequence labels. The generator expression iterates over LABEL_KEYS and checks if each key is present in the batch dict. If it finds a key that exists in the batch, it retrieves the corresponding value (which should be the sequence labels) and returns it as y. If none of the keys in LABEL_KEYS are found in the batch, y will be set to None.
        if x is None or y is None:
            raise KeyError(
                "Batch dict must contain a skeleton sequence and one sequence label, "
                "for example 'sequences' and 'sequence_labels'."
            )
        return x, y

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError(
        "Batch must be (sequences, sequence_labels) or a dict with sequence/label keys."
    )


def _as_sequence_tensor(x: Any) -> torch.Tensor:
    """Convert raw sequence input into a float tensor with shape B,T,C,H,W."""

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.float()

    if x.ndim != 5:
        raise ValueError(
            "Expected sequence batch with shape "
            "(batch, sequence_length, channels, height, width). "
            f"Got shape {tuple(x.shape)}."
        )
    return x


def _as_sequence_labels(y: Any) -> torch.Tensor:
    """Convert raw labels into one class id per sequence with shape B."""

    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long)
    else:
        y = y.long()

    if y.ndim > 1 and y.size(-1) == 1:
        y = y.view(-1)
    if y.ndim != 1:
        raise ValueError(
            "Expected one label per sequence with shape (batch,). "
            "Do not pass per-frame labels with shape (batch, sequence_length)."
        )
    return y


def _to_device(x: Any, y: Any, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one validated sequence batch and its labels onto the target device."""

    x = _as_sequence_tensor(x)
    y = _as_sequence_labels(y)
    if x.size(0) != y.size(0):
        raise ValueError(
            "Sequence batch and labels must have the same batch size. "
            f"Got {x.size(0)} sequences and {y.size(0)} labels."
        )
    return x.to(device), y.to(device)


def sequence_batch_size(x: torch.Tensor, y: torch.Tensor) -> int:
    """Return the number of sequences in a validated batch."""

    if x.size(0) != y.size(0):
        raise ValueError(
            "Sequence batch and labels must have the same batch size. "
            f"Got {x.size(0)} sequences and {y.size(0)} labels."
        )
    return x.size(0)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
    grad_clip: float | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Train for one epoch using complete skeleton sequences as training samples."""

    criterion = criterion or nn.CrossEntropyLoss()
    device = device or next(model.parameters()).device
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_sequences = 0

    batch_iter, progress_bar = _progress_iterator(
        train_loader,
        enabled=show_progress,
        desc=progress_desc or "Train",
    )

    for batch in batch_iter:
        x, y = _unpack_batch(batch)
        x, y = _to_device(x, y, device)

        optimizer.zero_grad(set_to_none=True) # zero_grad() sets the gradients of all model parameters to zero. This is necessary because by default, PyTorch accumulates gradients on subsequent backward passes. If you don't call zero_grad(), the gradients from previous batches will be added to the current gradients, which can lead to incorrect updates. The set_to_none=True argument is an optimization that sets the gradients to None instead of zero, which can save memory and slightly improve performance. When gradients are None, it indicates that they have not been computed yet, and PyTorch will allocate memory for them when needed during the backward pass.
        logits = model(x) 
        loss = criterion(logits, y)
        loss.backward() # loss.backward() computes the gradients of the loss with respect to the model parameters. This is done using backpropagation, which calculates how much each parameter contributed to the loss and updates the .grad attribute of each parameter with the computed gradients.

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # nn.utils.clip_grad_norm_() is a utility function that clips the gradients of the model parameters to a specified maximum norm. This is done to prevent the problem of exploding gradients, which can occur when the gradients become too large during training, leading to unstable updates and divergence. The function takes the model parameters and the maximum norm as arguments and scales down the gradients if their norm exceeds the specified threshold.

        optimizer.step() # optimizer.step() updates the model parameters based on the computed gradients. It uses the optimization algorithm defined by the optimizer (e.g., SGD, Adam) to adjust the parameters in the direction that minimizes the loss. The specific update rule depends on the chosen optimizer and its hyperparameters (e.g., learning rate, momentum). After calling step(), the model parameters are updated, and the training loop can proceed to the next batch of data.

        batch_sequences = sequence_batch_size(x, y) 
        total_loss += loss.item() * batch_sequences # loss.item() retrieves the scalar value of the loss tensor, and multiplying it by batch_sequences gives the total loss for that batch. This is done to accumulate the total loss across all batches, which can then be averaged over the total number of sequences at the end of the epoch to get the average loss per sequence.
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_sequences += batch_sequences
        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{total_loss / max(total_sequences, 1):.4f}",
                acc=f"{total_correct / max(total_sequences, 1):.4f}",
            )

    return {
        "loss": total_loss / max(total_sequences, 1),
        "accuracy": total_correct / max(total_sequences, 1),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Evaluate loss and accuracy over sequence-level samples."""

    criterion = criterion or nn.CrossEntropyLoss()
    device = device or next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_sequences = 0

    batch_iter, progress_bar = _progress_iterator(
        data_loader,
        enabled=show_progress,
        desc=progress_desc or "Eval",
    )

    for batch in batch_iter:
        x, y = _unpack_batch(batch)
        x, y = _to_device(x, y, device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_sequences = sequence_batch_size(x, y)
        total_loss += loss.item() * batch_sequences
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_sequences += batch_sequences
        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{total_loss / max(total_sequences, 1):.4f}",
                acc=f"{total_correct / max(total_sequences, 1):.4f}",
            )

    return {
        "loss": total_loss / max(total_sequences, 1),
        "accuracy": total_correct / max(total_sequences, 1),
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader=None,
    epochs: int = 20,
    lr: float = 1e-3,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
    grad_clip: float | None = 1.0,
    scheduler=None,
    checkpoint_path: Path | str | None = None,
    show_progress: bool = True,
) -> list[dict[str, float]]:
    """
    Train a CNN+LSTM classifier on skeleton-image sequences.

    Expected batch:
        sequences shape = (batch_size, sequence_length, 3, height, width)
        labels shape = (batch_size,)

    Each optimizer step treats one sliding-window sequence as one sample. Frames
    inside the window are encoded by the model and are not optimized as separate
    labels in the training loop.
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
            show_progress=show_progress,
            progress_desc=f"Epoch {epoch:03d}/{epochs:03d} train",
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
                show_progress=show_progress,
                progress_desc=f"Epoch {epoch:03d}/{epochs:03d} val",
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
