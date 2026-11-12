"""Five controlled deskew contrasts using the existing validation runner.

Labels record maximum correction angle and search step, both in degrees.
The shared runner applies the same input adaptation, batched inference,
checkpoint validation and corpus scoring used by the denoising study.
"""
from src.preprocessing.deskew import Deskew
from src.validation.denoising import evaluate, summarize


def configurations() -> dict:
    """Vary one parameter relative to max_angle=10, angle_step=0.5."""
    return {
        'baseline': None,
        'deskew_max5_step0.5': Deskew(max_angle=5, angle_step=0.5),
        'deskew_max10_step0.5': Deskew(max_angle=10, angle_step=0.5),
        'deskew_max15_step0.5': Deskew(max_angle=15, angle_step=0.5),
        'deskew_max10_step0.25': Deskew(max_angle=10, angle_step=0.25),
        'deskew_max10_step1': Deskew(max_angle=10, angle_step=1),
    }
