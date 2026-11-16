"""A from-scratch CNN + bidirectional LSTM baseline for line-level CTC.

Random initialization is not a usable handwriting recognizer. No training or
pretrained weights are downloaded by this module.
"""
from dataclasses import asdict, dataclass

from PIL import Image
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence, pad_sequence,
)


@dataclass(frozen=True)
class CNNLSTMCTCConfig:
    """Initial architecture settings, not experimentally selected optima."""

    num_classes: int
    input_height: int = 64
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    hidden_size: int = 128
    lstm_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("num_classes", "input_height", "hidden_size", "lstm_layers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_classes < 2 or self.input_height < 8:
            raise ValueError("Require at least two classes and input height >= 8")
        if len(self.cnn_channels) != 3 or any(type(c) is not int or c < 1 for c in self.cnn_channels):
            raise ValueError("cnn_channels must contain three positive integers")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


def _lengths(lengths: torch.Tensor, batch_size: int, maximum: int) -> torch.Tensor:
    """Validate integer lengths and return CPU int64 for packed sequences/CTC."""
    if lengths.ndim != 1 or len(lengths) != batch_size:
        raise ValueError("Provide one sequence length per batch example")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Sequence lengths must be integer tensors")
    lengths = lengths.detach().to(device="cpu", dtype=torch.long)
    if bool((lengths < 1).any()) or bool((lengths > maximum).any()):
        raise ValueError(f"Sequence lengths must be between 1 and {maximum}")
    return lengths


class CNNLSTMCTC(nn.Module):
    """Extract visual features, read their sequence, predict character scores.

    CNN processing crops away each image's batch padding first. This prevents
    another image's width from changing convolution boundary features. Packed
    sequences keep padding out of both directions of the LSTM. Per-image CNN
    execution favors consistent inputs over maximum training throughput.
    """

    def __init__(self, config: CNNLSTMCTCConfig):
        super().__init__()
        self.config = config
        first, second, third = config.cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, first, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(first, second, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(second, third, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.lstm = nn.LSTM(
            input_size=third, hidden_size=config.hidden_size,
            num_layers=config.lstm_layers, bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_classes)

    @staticmethod
    def output_lengths(image_widths: torch.Tensor) -> torch.Tensor:
        """Two width-halving pools produce floor(width / 4) time steps."""
        if image_widths.ndim != 1 or image_widths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Image widths must be a one-dimensional integer tensor")
        if bool((image_widths < 4).any()):
            raise ValueError("Prepared images need width >= 4 for the CNN pools")
        return image_widths.detach().to(device="cpu", dtype=torch.long) // 4

    def forward(self, images: torch.Tensor, image_widths: torch.Tensor) -> tuple:
        """Return log probabilities [time, batch, classes] and CTC lengths."""
        if images.ndim != 4 or images.shape[1] != 1 or images.shape[2] != self.config.input_height:
            raise ValueError(f"Expected [batch, 1, {self.config.input_height}, width] images")
        if images.shape[0] < 1 or not images.is_floating_point():
            raise ValueError("Provide a nonempty floating-point image batch")
        widths = _lengths(image_widths, images.shape[0], images.shape[3])
        lengths = self.output_lengths(widths)
        sequences = []
        for image, width in zip(images, widths.tolist()):
            features = self.cnn(image[None, :, :, :width])
            # Average only the remaining height: left-to-right positions survive.
            sequences.append(features.mean(dim=2).squeeze(0).transpose(0, 1))
        padded = pad_sequence(sequences)  # [time, batch, CNN channels]
        packed = pack_padded_sequence(padded, lengths, enforce_sorted=False)
        recurrent, _ = self.lstm(packed)
        recurrent, _ = pad_packed_sequence(recurrent)
        return self.classifier(recurrent).log_softmax(dim=2), lengths

    def to_config(self) -> dict:
        return {"architecture": "CNN-BiLSTM-CTC", **asdict(self.config),
                "width_downsampling": 4, "height_reduction": "mean after CNN",
                "cnn_padding_policy": "crop to each actual width before CNN",
                "lstm_padding_policy": "packed sequences",
                "output": "time,batch,class log probabilities"}


def ctc_batch_loss(log_probs: torch.Tensor, input_lengths: torch.Tensor,
                   batch: dict, blank_id: int = 0) -> torch.Tensor:
    """Compute CTC loss; explicitly reject impossible repeated-letter alignments.

    No samples are silently dropped and infinite losses are not zeroed out.
    Targets are the concatenated labels produced by collate_ctc_lines.
    """
    if log_probs.ndim != 3:
        raise ValueError("Expected log probabilities [time, batch, classes]")
    time, size, classes = log_probs.shape
    if not 0 <= blank_id < classes:
        raise ValueError("Blank ID must be a valid output class")
    lengths = _lengths(input_lengths, size, time)
    targets = batch["targets"]
    if targets.ndim != 1 or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("Targets must be concatenated integer character IDs")
    target_lengths = _lengths(batch["target_lengths"], size, len(targets))
    if int(target_lengths.sum()) != len(targets):
        raise ValueError("Target lengths must sum to the concatenated label count")
    if bool(((targets < 0) | (targets >= classes) | (targets == blank_id)).any()):
        raise ValueError("Target IDs must be in vocabulary and exclude the CTC blank")
    offset = 0
    for index, target_length in enumerate(target_lengths.tolist()):
        label = targets[offset:offset + target_length]
        required = target_length + int((label[1:] == label[:-1]).sum())
        if required > int(lengths[index]):
            sample = batch.get("sample_ids", list(range(size)))[index]
            raise ValueError(f"Impossible CTC alignment for {sample}: need {required} time steps, "
                             f"CNN supplies {int(lengths[index])}; revise input/model using training data")
        offset += target_length
    return nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=False)(
        log_probs, targets.to(log_probs.device), lengths, target_lengths,
    )


def greedy_ctc_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor,
                      vocabulary) -> list[str]:
    """Merge adjacent repeats first, then remove blanks; l-blank-l stays ll."""
    if log_probs.ndim != 3 or log_probs.shape[2] != vocabulary.num_classes:
        raise ValueError("Expected [time, batch, vocabulary classes] scores")
    lengths = _lengths(input_lengths, log_probs.shape[1], log_probs.shape[0])
    paths = log_probs.detach().argmax(dim=2).transpose(0, 1).cpu()
    predictions = []
    for path, length in zip(paths, lengths.tolist()):
        previous, label = None, []
        for index in path[:length].tolist():
            if index != previous and index != vocabulary.blank_id:
                label.append(index)
            previous = index
        predictions.append(vocabulary.decode_labels(label))
    return predictions


class CNNLSTMCTCRecognizer:
    """Adapter for existing predict(image)/to_config() evaluation routines.

    Set trained=True only after loading or training actual model weights.
    This flag prevents accidental accuracy reports for random initialization.
    """

    def __init__(self, model: CNNLSTMCTC, vocabulary, preparation=None,
                 device: str = "cpu", trained: bool = False):
        # Local import keeps models/evaluation/data modules free of import cycles.
        from src.training.ctc_data import CTCImagePreparation

        self.model = model
        self.vocabulary = vocabulary
        self.preparation = preparation or CTCImagePreparation(model.config.input_height)
        if vocabulary.num_classes != model.config.num_classes:
            raise ValueError("Vocabulary and model class counts must match")
        if self.preparation.height != model.config.input_height:
            raise ValueError("Preparation height must match the CNN input height")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.trained = trained

    def predict(self, image: Image.Image) -> str:
        if not self.trained:
            raise RuntimeError("Train or load trained weights before reporting recognition predictions")
        prepared = self.preparation(image).unsqueeze(0).to(self.device)
        width = torch.tensor([prepared.shape[3]], dtype=torch.long)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                scores, lengths = self.model(prepared, width)
                return greedy_ctc_decode(scores, lengths, self.vocabulary)[0]
        finally:
            self.model.train(was_training)

    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu"):
        """Restore weights, character IDs, and mandatory inputs together."""
        from src.training.ctc_training import load_recognizer
        return load_recognizer(path, device=device)

    def to_config(self) -> dict:
        return {**self.model.to_config(), "vocabulary": self.vocabulary.to_config(),
                "mandatory_input": self.preparation.to_config(),
                "device": str(self.device), "decoding": "greedy CTC",
                "checkpoint": getattr(self, "checkpoint_path", None),
                "checkpoint_sha256": getattr(self, "checkpoint_sha256", None),
                "checkpoint_epoch": getattr(self, "checkpoint_epoch", None),
                "weights_status": "trained" if self.trained else "random initialization"}
