from pathlib import Path

import torch
import torch.nn as nn
from utils import _to_device, _unpack_batch, evaluate_model, train_model

class SkeletonFrameEncoder(nn.Module):
    """Small CNN that converts one skeleton image into a feature vector."""

    def __init__(self, in_channels=3, embedding_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
        )   

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.cnn(x)
        return self.projection(x)


class SkeletonImageLSTMClassifier(nn.Module):
    """
    Classify HAR-UP action from 10 skeleton images.

    Input:
        x shape = (batch_size, sequence_length, channels, height, width)
        example = (batch_size, 10, 3, 224, 224)

    Output:
        logits shape = (batch_size, num_classes)
    """

    def __init__(
        self,
        sequence_length=10,
        in_channels=3,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=11,
        dropout=0.3,
        bidirectional=False,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        
        self.encoder = SkeletonFrameEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers, # number of stacked LSTM layers. If num_layers=1, there is only one layer. If num_layers>1, the output of each layer is fed as input to the next layer, and dropout is applied between layers (except after the last layer).
            batch_first=True,  # batch_first=True means the input and output tensors are expected to have the shape (batch, sequence, feature). If False, the expected shape is (sequence, batch, feature).
            dropout=lstm_dropout, 
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes),
        )

    def forward(self, x):
        if x.ndim != 5: #ndim returns the number of dimensions of the input tensor x. In this case, we expect x to have 5 dimensions: (batch_size, sequence_length, channels, height, width). If x does not have 5 dimensions, it means the input shape is incorrect for our model, and we raise a ValueError with a message indicating the expected shape.
            raise ValueError(
                "Expected x with shape (batch, sequence, channels, height, width)."
            )

        batch_size, sequence_length, channels, height, width = x.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {sequence_length}."
            )

        frames = x.reshape(batch_size * sequence_length, channels, height, width)
        frame_features = self.encoder(frames)
        sequence_features = frame_features.reshape(batch_size, sequence_length, -1)

        _, (hidden, _) = self.lstm(sequence_features)
        if self.lstm.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        return self.classifier(last_hidden)

    @torch.no_grad()
    def predict(self, x):
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        class_ids = probabilities.argmax(dim=1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "class_ids": class_ids,
        }
