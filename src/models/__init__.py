"""Interchangeable handwriting recognizers."""
from .base import HandwritingRecognizer
from .trocr import TrOCRRecognizer

__all__ = ["HandwritingRecognizer", "TrOCRRecognizer"]
