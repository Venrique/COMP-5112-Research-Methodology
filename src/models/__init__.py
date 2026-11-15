"""Interchangeable handwriting recognizers."""
from .base import HandwritingRecognizer
from .trocr import TrOCRRecognizer
from .cnn_lstm_ctc import CNNLSTMCTC, CNNLSTMCTCConfig, CNNLSTMCTCRecognizer

__all__ = ["HandwritingRecognizer", "TrOCRRecognizer", "CNNLSTMCTC",
           "CNNLSTMCTCConfig", "CNNLSTMCTCRecognizer"]
