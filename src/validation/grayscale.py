"""Validate the existing grayscale transform against untreated inputs.

Grayscale has no tunable numeric parameters. Test its enabled setting once
rather than treating repeated identical executions as parameter configurations.
"""
from src.preprocessing.grayscale import Grayscale
from src.validation.denoising import evaluate, summarize


def configurations() -> dict:
    """One meaningful grayscale configuration plus baseline."""
    return {'baseline': None, 'grayscale': Grayscale()}
