"""Frozen ViTPose-backbone embeddings + trainable LSTM classifier."""

from __future__ import annotations

import torch
from torch import nn

from common import configure_runtime_cache, resolve_hf_model_source


DEFAULT_VIT_MODEL = "usyd-community/vitpose-base-simple"


class FrozenVitPoseEmbeddingLSTMClassifier(nn.Module):
    """
    Classify sequences from frozen ViTPose backbone embeddings.

    Input:
        x shape = (batch, sequence_length, 3, 256, 192)

    Frozen part:
        ViTPose backbone only. The heatmap/keypoint head is never called.

    Trainable part:
        LSTM + dropout + linear classifier.
    """

    def __init__(
        self,
        vit_model_name: str = DEFAULT_VIT_MODEL,
        allow_download: bool = False,
        sequence_length: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 11,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        configure_runtime_cache()

        from transformers import VitPoseForPoseEstimation  # noqa: PLC0415

        model_source = resolve_hf_model_source(vit_model_name, allow_download)
        self.vitpose = VitPoseForPoseEstimation.from_pretrained(
            model_source,
            local_files_only=not allow_download,
        )
        self.vitpose.eval()
        for parameter in self.vitpose.parameters():
            parameter.requires_grad = False

        self.sequence_length = sequence_length
        self.embedding_dim = int(self.vitpose.config.backbone_config.hidden_size)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes),
        )

        trainable = sum(p.numel() for p in self.trainable_parameters())
        frozen = sum(p.numel() for p in self.vitpose.parameters())
        print(
            "Initialized FrozenVitPoseEmbeddingLSTMClassifier "
            f"embedding_dim={self.embedding_dim} trainable={trainable} frozen_vit={frozen}"
        )

    def trainable_parameters(self):
        yield from self.lstm.parameters()
        yield from self.classifier.parameters()

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "lstm": self.lstm.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_trainable_state_dict(self, state_dict: dict[str, object], strict: bool = True):
        if "lstm" in state_dict and "classifier" in state_dict:
            lstm_result = self.lstm.load_state_dict(state_dict["lstm"], strict=strict)
            classifier_result = self.classifier.load_state_dict(
                state_dict["classifier"],
                strict=strict,
            )
            return lstm_result, classifier_result

        trainable_state = {
            key: value
            for key, value in state_dict.items()
            if str(key).startswith(("lstm.", "classifier."))
        }
        return self.load_state_dict(trainable_state, strict=False)

    def extract_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        """Return one frozen ViTPose embedding per frame."""

        self.vitpose.eval()
        with torch.no_grad():
            outputs = self.vitpose.backbone(pixel_values=frames)
            feature_map = outputs.feature_maps[0]
            return feature_map.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "Expected x with shape (batch, sequence, channels, height, width)."
            )

        batch_size, sequence_length, channels, height, width = x.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {sequence_length}."
            )

        frames = x.reshape(batch_size * sequence_length, channels, height, width)
        embeddings = self.extract_embeddings(frames)
        sequence_embeddings = embeddings.reshape(batch_size, sequence_length, -1)

        _, (hidden, _) = self.lstm(sequence_embeddings)
        if self.lstm.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        return self.classifier(last_hidden)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        class_ids = probabilities.argmax(dim=1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "class_ids": class_ids,
        }
